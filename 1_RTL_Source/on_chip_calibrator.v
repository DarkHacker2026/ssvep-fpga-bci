`timescale 1ns / 1ps
// =============================================================================
// on_chip_calibrator.v
//
// Assumptions used for this architecture pass:
//   - BTN_YES collects calibration anchors for class 0.
//   - BTN_NO  collects calibration anchors for class 1.
//   - Classes 2 and 3 keep their offline bootstrap weights initially and are
//     refined later through confident pseudo-label output bias updates.
//
// The calibrator performs:
//   - spatial-weight learning from per-window 3-channel means
//   - LDA-style feature direction using means + within-class variance
//   - full-network RAM updates (fc1/fc2/fc3) using low-rank LDA backprop
//   - online pseudo-label bias updates every 32 confident decisions
// =============================================================================

module on_chip_calibrator #(
    parameter integer N_FEATURES   = 12,
    parameter integer N_H1         = 32,
    parameter integer N_H2         = 8,
    parameter integer N_OUT        = 4,
    parameter integer N_CAL        = 16,
    parameter integer DEBOUNCE_MAX = 2_000_000
)(
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire [N_FEATURES*18-1:0]     feature_flat,
    input  wire                         features_valid,
    input  wire signed [23:0]           mean_ch1,
    input  wire signed [23:0]           mean_ch2,
    input  wire signed [23:0]           mean_ch3,
    input  wire [N_H1*18-1:0]           h1_flat,
    input  wire [N_H2*18-1:0]           h2_flat,
    input  wire [N_OUT*18-1:0]          logit_flat,
    input  wire [1:0]                   pred_class,
    input  wire [17:0]                  pred_margin,
    input  wire                         output_valid,
    input  wire                         btn_yes,
    input  wire                         btn_no,

    output reg                          nn_wr_en,
    output reg  [9:0]                   nn_wr_addr,
    output reg  signed [17:0]           nn_wr_data,
    output reg  [9:0]                   nn_rd_addr,
    input  wire signed [17:0]           nn_rd_data,

    output reg  signed [17:0]           spatial_w_ch1,
    output reg  signed [17:0]           spatial_w_ch2,
    output reg  signed [17:0]           spatial_w_ch3,
    output reg  [7:0]                   gain_ch1_q4,
    output reg  [7:0]                   gain_ch2_q4,
    output reg  [7:0]                   gain_ch3_q4,
    output reg                          calibrated,
    output reg                          led_cal_yes,
    output reg                          led_cal_no,
    output reg                          led_computing,
    output reg                          led_done
);

localparam integer FC1_W_COUNT = N_FEATURES * N_H1;
localparam integer FC1_B_BASE  = FC1_W_COUNT;
localparam integer FC1_B_COUNT = N_H1;
localparam integer FC2_W_BASE  = FC1_B_BASE + FC1_B_COUNT;
localparam integer FC2_W_COUNT = N_H1 * N_H2;
localparam integer FC2_B_BASE  = FC2_W_BASE + FC2_W_COUNT;
localparam integer FC2_B_COUNT = N_H2;
localparam integer FC3_W_BASE  = FC2_B_BASE + FC2_B_COUNT;
localparam integer FC3_W_COUNT = N_H2 * N_OUT;
localparam integer FC3_B_BASE  = FC3_W_BASE + FC3_W_COUNT;
localparam integer FC3_B_COUNT = N_OUT;
localparam integer TOTAL_COUNT = FC3_B_BASE + FC3_B_COUNT;

localparam [3:0]
    ST_IDLE         = 4'd0,
    ST_CAL_YES      = 4'd1,
    ST_WAIT_NO      = 4'd2,
    ST_CAL_NO       = 4'd3,
    ST_COMPUTE      = 4'd4,
    ST_WRITE_READ   = 4'd5,
    ST_WRITE_COMMIT = 4'd6,
    ST_RUN          = 4'd7,
    ST_ONLINE_READ  = 4'd8,
    ST_ONLINE_WRITE = 4'd9;

localparam [17:0] ONLINE_MARGIN = 18'd8192;

reg [3:0] state;
reg [5:0] calc_idx;
reg [9:0] wr_cursor;
reg signed [17:0] pending_update;
reg [5:0] online_count;
reg [2:0] online_bias_idx;
reg [7:0] pseudo_hits [0:N_OUT-1];

reg [$clog2(DEBOUNCE_MAX+1)-1:0] db_yes_cnt, db_no_cnt;
reg db_yes_q, db_no_q;
reg db_yes_prev, db_no_prev;
reg btn_yes_rise, btn_no_rise;

reg [4:0] yes_count;
reg [4:0] no_count;

reg [N_FEATURES*18-1:0] pending_feature_flat;
reg signed [23:0]       pending_mean_ch1, pending_mean_ch2, pending_mean_ch3;

reg signed [47:0] feat_sum_yes [0:N_FEATURES-1];
reg signed [47:0] feat_sum_no  [0:N_FEATURES-1];
reg [63:0]        feat_sumsq_yes [0:N_FEATURES-1];
reg [63:0]        feat_sumsq_no  [0:N_FEATURES-1];
reg signed [17:0] lda_feat_w [0:N_FEATURES-1];

reg signed [47:0] h1_sum_yes [0:N_H1-1];
reg signed [47:0] h1_sum_no  [0:N_H1-1];
reg signed [17:0] h1_diff    [0:N_H1-1];

reg signed [47:0] h2_sum_yes [0:N_H2-1];
reg signed [47:0] h2_sum_no  [0:N_H2-1];
reg signed [17:0] h2_diff    [0:N_H2-1];

reg signed [47:0] ch_sum_yes [0:2];
reg signed [47:0] ch_sum_no  [0:2];

integer i;
integer row_idx;
integer col_idx;
reg signed [17:0] feat_v;
reg signed [17:0] h1_v;
reg signed [17:0] h2_v;
reg signed [47:0] mu_yes;
reg signed [47:0] mu_no;
reg [63:0]        var_yes;
reg [63:0]        var_no;
reg [17:0]        lda_den;

function signed [17:0] sat18;
    input signed [47:0] v;
    begin
        if (v > 48'sd131071)
            sat18 = 18'sd131071;
        else if (v < -48'sd131072)
            sat18 = -18'sd131072;
        else
            sat18 = v[17:0];
    end
endfunction

function [17:0] abs18;
    input signed [17:0] v;
    begin
        abs18 = v[17] ? (~v + 18'd1) : v[17:0];
    end
endfunction

function [7:0] gain_from_weight;
    input signed [17:0] w;
    reg [17:0] aw;
    reg [7:0] g;
    begin
        aw = abs18(w);
        g  = 8'd16 + aw[15:12];
        if (g < 8'd8)
            gain_from_weight = 8'd8;
        else if (g > 8'd31)
            gain_from_weight = 8'd31;
        else
            gain_from_weight = g;
    end
endfunction

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        db_yes_cnt   <= 0; db_no_cnt <= 0;
        db_yes_q     <= 0; db_no_q   <= 0;
        db_yes_prev  <= 0; db_no_prev<= 0;
        btn_yes_rise <= 0; btn_no_rise <= 0;
    end else begin
        btn_yes_rise <= 0;
        btn_no_rise  <= 0;

        if (btn_yes == db_yes_q)
            db_yes_cnt <= 0;
        else if (db_yes_cnt == DEBOUNCE_MAX - 1) begin
            db_yes_q   <= btn_yes;
            db_yes_cnt <= 0;
        end else
            db_yes_cnt <= db_yes_cnt + 1;

        if (btn_no == db_no_q)
            db_no_cnt <= 0;
        else if (db_no_cnt == DEBOUNCE_MAX - 1) begin
            db_no_q   <= btn_no;
            db_no_cnt <= 0;
        end else
            db_no_cnt <= db_no_cnt + 1;

        db_yes_prev <= db_yes_q;
        db_no_prev  <= db_no_q;
        if (db_yes_q && !db_yes_prev) btn_yes_rise <= 1'b1;
        if (db_no_q  && !db_no_prev)  btn_no_rise  <= 1'b1;
    end
end

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        pending_feature_flat <= {(N_FEATURES*18){1'b0}};
        pending_mean_ch1 <= 24'sd0;
        pending_mean_ch2 <= 24'sd0;
        pending_mean_ch3 <= 24'sd0;
    end else if (features_valid) begin
        pending_feature_flat <= feature_flat;
        pending_mean_ch1 <= mean_ch1;
        pending_mean_ch2 <= mean_ch2;
        pending_mean_ch3 <= mean_ch3;
    end
end

always @(*) begin
    pending_update = 18'sd0;
    if (wr_cursor < FC1_W_COUNT) begin
        row_idx = wr_cursor / N_FEATURES;
        col_idx = wr_cursor % N_FEATURES;
        pending_update = sat18(($signed(h1_diff[row_idx]) * $signed(lda_feat_w[col_idx])) >>> 12);
    end else if (wr_cursor < FC1_B_BASE + FC1_B_COUNT) begin
        row_idx = wr_cursor - FC1_B_BASE;
        pending_update = h1_diff[row_idx] >>> 2;
    end else if (wr_cursor < FC2_W_BASE + FC2_W_COUNT) begin
        row_idx = (wr_cursor - FC2_W_BASE) / N_H1;
        col_idx = (wr_cursor - FC2_W_BASE) % N_H1;
        pending_update = sat18(($signed(h2_diff[row_idx]) * $signed(h1_diff[col_idx])) >>> 12);
    end else if (wr_cursor < FC2_B_BASE + FC2_B_COUNT) begin
        row_idx = wr_cursor - FC2_B_BASE;
        pending_update = h2_diff[row_idx] >>> 2;
    end else if (wr_cursor < FC3_W_BASE + FC3_W_COUNT) begin
        row_idx = (wr_cursor - FC3_W_BASE) / N_H2;
        col_idx = (wr_cursor - FC3_W_BASE) % N_H2;
        if (row_idx == 0)
            pending_update = h2_diff[col_idx] >>> 1;
        else if (row_idx == 1)
            pending_update = -(h2_diff[col_idx] >>> 1);
        else
            pending_update = 18'sd0;
    end else if (wr_cursor < TOTAL_COUNT) begin
        row_idx = wr_cursor - FC3_B_BASE;
        if (row_idx == 0)
            pending_update = 18'sd4096;
        else if (row_idx == 1)
            pending_update = -18'sd4096;
        else
            pending_update = 18'sd0;
    end
end

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state <= ST_IDLE;
        yes_count <= 5'd0;
        no_count  <= 5'd0;
        calc_idx  <= 6'd0;
        wr_cursor <= 10'd0;
        nn_wr_en  <= 1'b0;
        nn_wr_addr<= 10'd0;
        nn_wr_data<= 18'sd0;
        nn_rd_addr<= 10'd0;
        // Zero-sum Laplacian prior so CAR does not collapse the default
        // spatial combination to zero before online calibration runs.
        spatial_w_ch1 <= -18'sd32768;
        spatial_w_ch2 <= 18'sd65536;
        spatial_w_ch3 <= -18'sd32768;
        gain_ch1_q4   <= 8'd16;
        gain_ch2_q4   <= 8'd16;
        gain_ch3_q4   <= 8'd16;
        calibrated    <= 1'b0;
        led_cal_yes   <= 1'b0;
        led_cal_no    <= 1'b0;
        led_computing <= 1'b0;
        led_done      <= 1'b0;
        online_count  <= 6'd0;
        online_bias_idx <= 3'd0;
        for (i = 0; i < N_FEATURES; i = i + 1) begin
            feat_sum_yes[i]   <= 48'sd0;
            feat_sum_no[i]    <= 48'sd0;
            feat_sumsq_yes[i] <= 64'd0;
            feat_sumsq_no[i]  <= 64'd0;
            lda_feat_w[i]     <= 18'sd0;
        end
        for (i = 0; i < N_H1; i = i + 1) begin
            h1_sum_yes[i] <= 48'sd0;
            h1_sum_no[i]  <= 48'sd0;
            h1_diff[i]    <= 18'sd0;
        end
        for (i = 0; i < N_H2; i = i + 1) begin
            h2_sum_yes[i] <= 48'sd0;
            h2_sum_no[i]  <= 48'sd0;
            h2_diff[i]    <= 18'sd0;
        end
        for (i = 0; i < 3; i = i + 1) begin
            ch_sum_yes[i] <= 48'sd0;
            ch_sum_no[i]  <= 48'sd0;
        end
        for (i = 0; i < N_OUT; i = i + 1)
            pseudo_hits[i] <= 8'd0;
    end else begin
        nn_wr_en <= 1'b0;

        case (state)
            ST_IDLE: begin
                led_cal_yes <= 1'b0;
                led_cal_no  <= 1'b0;
                led_computing <= 1'b0;
                if (btn_yes_rise) begin
                    yes_count <= 5'd0;
                    no_count  <= 5'd0;
                    calibrated <= 1'b0;
                    led_done   <= 1'b0;
                    for (i = 0; i < N_FEATURES; i = i + 1) begin
                        feat_sum_yes[i]   <= 48'sd0;
                        feat_sum_no[i]    <= 48'sd0;
                        feat_sumsq_yes[i] <= 64'd0;
                        feat_sumsq_no[i]  <= 64'd0;
                    end
                    for (i = 0; i < N_H1; i = i + 1) begin
                        h1_sum_yes[i] <= 48'sd0;
                        h1_sum_no[i]  <= 48'sd0;
                    end
                    for (i = 0; i < N_H2; i = i + 1) begin
                        h2_sum_yes[i] <= 48'sd0;
                        h2_sum_no[i]  <= 48'sd0;
                    end
                    for (i = 0; i < 3; i = i + 1) begin
                        ch_sum_yes[i] <= 48'sd0;
                        ch_sum_no[i]  <= 48'sd0;
                    end
                    led_cal_yes <= 1'b1;
                    state <= ST_CAL_YES;
                end else begin
                    state <= ST_RUN;
                end
            end

            ST_CAL_YES: begin
                if (output_valid) begin
                    ch_sum_yes[0] <= ch_sum_yes[0] + {{24{pending_mean_ch1[23]}}, pending_mean_ch1};
                    ch_sum_yes[1] <= ch_sum_yes[1] + {{24{pending_mean_ch2[23]}}, pending_mean_ch2};
                    ch_sum_yes[2] <= ch_sum_yes[2] + {{24{pending_mean_ch3[23]}}, pending_mean_ch3};
                    for (i = 0; i < N_FEATURES; i = i + 1) begin
                        feat_v = pending_feature_flat[i*18 +: 18];
                        feat_sum_yes[i]   <= feat_sum_yes[i] + {{30{feat_v[17]}}, feat_v};
                        feat_sumsq_yes[i] <= feat_sumsq_yes[i] + ($signed(feat_v) * $signed(feat_v));
                    end
                    for (i = 0; i < N_H1; i = i + 1) begin
                        h1_v = h1_flat[i*18 +: 18];
                        h1_sum_yes[i] <= h1_sum_yes[i] + {{30{h1_v[17]}}, h1_v};
                    end
                    for (i = 0; i < N_H2; i = i + 1) begin
                        h2_v = h2_flat[i*18 +: 18];
                        h2_sum_yes[i] <= h2_sum_yes[i] + {{30{h2_v[17]}}, h2_v};
                    end

                    if (yes_count == N_CAL - 1) begin
                        led_cal_yes <= 1'b0;
                        state <= ST_WAIT_NO;
                    end else begin
                        yes_count <= yes_count + 5'd1;
                    end
                end
            end

            ST_WAIT_NO: begin
                if (btn_no_rise) begin
                    led_cal_no <= 1'b1;
                    state <= ST_CAL_NO;
                end
            end

            ST_CAL_NO: begin
                if (output_valid) begin
                    ch_sum_no[0] <= ch_sum_no[0] + {{24{pending_mean_ch1[23]}}, pending_mean_ch1};
                    ch_sum_no[1] <= ch_sum_no[1] + {{24{pending_mean_ch2[23]}}, pending_mean_ch2};
                    ch_sum_no[2] <= ch_sum_no[2] + {{24{pending_mean_ch3[23]}}, pending_mean_ch3};
                    for (i = 0; i < N_FEATURES; i = i + 1) begin
                        feat_v = pending_feature_flat[i*18 +: 18];
                        feat_sum_no[i]   <= feat_sum_no[i] + {{30{feat_v[17]}}, feat_v};
                        feat_sumsq_no[i] <= feat_sumsq_no[i] + ($signed(feat_v) * $signed(feat_v));
                    end
                    for (i = 0; i < N_H1; i = i + 1) begin
                        h1_v = h1_flat[i*18 +: 18];
                        h1_sum_no[i] <= h1_sum_no[i] + {{30{h1_v[17]}}, h1_v};
                    end
                    for (i = 0; i < N_H2; i = i + 1) begin
                        h2_v = h2_flat[i*18 +: 18];
                        h2_sum_no[i] <= h2_sum_no[i] + {{30{h2_v[17]}}, h2_v};
                    end

                    if (no_count == N_CAL - 1) begin
                        led_cal_no <= 1'b0;
                        led_computing <= 1'b1;
                        calc_idx <= 6'd0;
                        state <= ST_COMPUTE;
                    end else begin
                        no_count <= no_count + 5'd1;
                    end
                end
            end

            ST_COMPUTE: begin
                if (calc_idx < N_FEATURES) begin
                    mu_yes = feat_sum_yes[calc_idx] >>> 4;
                    mu_no  = feat_sum_no[calc_idx]  >>> 4;
                    // Guard against negative variance from fixed-point rounding:
                    // E[X^2] - E[X]^2 can underflow in unsigned arithmetic.
                    var_yes = (feat_sumsq_yes[calc_idx] >>> 4);
                    var_no  = (feat_sumsq_no[calc_idx]  >>> 4);
                    begin : variance_guard
                        reg [63:0] mean_sq_yes, mean_sq_no;
                        mean_sq_yes = ($signed(mu_yes[17:0]) * $signed(mu_yes[17:0])) >>> 4;
                        mean_sq_no  = ($signed(mu_no[17:0])  * $signed(mu_no[17:0]))  >>> 4;
                        var_yes = (var_yes >= mean_sq_yes) ? (var_yes - mean_sq_yes) : 64'd0;
                        var_no  = (var_no  >= mean_sq_no)  ? (var_no  - mean_sq_no)  : 64'd0;
                    end
                    if ((var_yes + var_no + 64'd1) > 64'd131071)
                        lda_den = 18'd131071;
                    else
                        lda_den = (var_yes + var_no + 64'd1);
                    lda_feat_w[calc_idx] <= sat18((($signed(mu_yes - mu_no) <<< 8) / $signed(lda_den)));
                    calc_idx <= calc_idx + 6'd1;
                end else if (calc_idx < N_FEATURES + N_H1) begin
                    h1_diff[calc_idx - N_FEATURES] <=
                        sat18((h1_sum_yes[calc_idx - N_FEATURES] >>> 4) - (h1_sum_no[calc_idx - N_FEATURES] >>> 4));
                    calc_idx <= calc_idx + 6'd1;
                end else if (calc_idx < N_FEATURES + N_H1 + N_H2) begin
                    h2_diff[calc_idx - N_FEATURES - N_H1] <=
                        sat18((h2_sum_yes[calc_idx - N_FEATURES - N_H1] >>> 4) -
                              (h2_sum_no[calc_idx - N_FEATURES - N_H1] >>> 4));
                    calc_idx <= calc_idx + 6'd1;
                end else begin
                    spatial_w_ch1 <= sat18((ch_sum_yes[0] >>> 4) - (ch_sum_no[0] >>> 4));
                    spatial_w_ch2 <= sat18((ch_sum_yes[1] >>> 4) - (ch_sum_no[1] >>> 4));
                    spatial_w_ch3 <= sat18((ch_sum_yes[2] >>> 4) - (ch_sum_no[2] >>> 4));
                    gain_ch1_q4   <= gain_from_weight(sat18((ch_sum_yes[0] >>> 4) - (ch_sum_no[0] >>> 4)));
                    gain_ch2_q4   <= gain_from_weight(sat18((ch_sum_yes[1] >>> 4) - (ch_sum_no[1] >>> 4)));
                    gain_ch3_q4   <= gain_from_weight(sat18((ch_sum_yes[2] >>> 4) - (ch_sum_no[2] >>> 4)));
                    wr_cursor     <= 10'd0;
                    state         <= ST_WRITE_READ;
                end
            end

            ST_WRITE_READ: begin
                nn_rd_addr <= wr_cursor;
                state <= ST_WRITE_COMMIT;
            end

            ST_WRITE_COMMIT: begin
                nn_wr_en   <= 1'b1;
                nn_wr_addr <= wr_cursor;
                nn_wr_data <= nn_rd_data + pending_update;
                if (wr_cursor == TOTAL_COUNT - 1) begin
                    led_computing <= 1'b0;
                    calibrated    <= 1'b1;
                    led_done      <= 1'b1;
                    state         <= ST_RUN;
                end else begin
                    wr_cursor <= wr_cursor + 10'd1;
                    state     <= ST_WRITE_READ;
                end
            end

            ST_RUN: begin
                if (btn_yes_rise) begin
                    yes_count <= 5'd0;
                    no_count  <= 5'd0;
                    calibrated <= 1'b0;
                    led_done   <= 1'b0;
                    for (i = 0; i < N_FEATURES; i = i + 1) begin
                        feat_sum_yes[i]   <= 48'sd0;
                        feat_sum_no[i]    <= 48'sd0;
                        feat_sumsq_yes[i] <= 64'd0;
                        feat_sumsq_no[i]  <= 64'd0;
                    end
                    for (i = 0; i < N_H1; i = i + 1) begin
                        h1_sum_yes[i] <= 48'sd0;
                        h1_sum_no[i]  <= 48'sd0;
                    end
                    for (i = 0; i < N_H2; i = i + 1) begin
                        h2_sum_yes[i] <= 48'sd0;
                        h2_sum_no[i]  <= 48'sd0;
                    end
                    for (i = 0; i < 3; i = i + 1) begin
                        ch_sum_yes[i] <= 48'sd0;
                        ch_sum_no[i]  <= 48'sd0;
                    end
                    led_cal_yes <= 1'b1;
                    state <= ST_CAL_YES;
                end else if (output_valid && (pred_margin > ONLINE_MARGIN)) begin
                    pseudo_hits[pred_class] <= pseudo_hits[pred_class] + 8'd1;
                    if (online_count == 6'd31) begin
                        online_count <= 6'd0;
                        online_bias_idx <= 3'd0;
                        state <= ST_ONLINE_READ;
                    end else begin
                        online_count <= online_count + 6'd1;
                    end
                end
            end

            ST_ONLINE_READ: begin
                nn_rd_addr <= FC3_B_BASE + online_bias_idx;
                state <= ST_ONLINE_WRITE;
            end

            ST_ONLINE_WRITE: begin
                nn_wr_en   <= 1'b1;
                nn_wr_addr <= FC3_B_BASE + online_bias_idx;
                nn_wr_data <= sat18($signed(nn_rd_data) + $signed({{10{1'b0}}, pseudo_hits[online_bias_idx]}) - $signed(18'sd4));
                pseudo_hits[online_bias_idx] <= 8'd0;
                if (online_bias_idx == N_OUT-1) begin
                    state <= ST_RUN;
                end else begin
                    online_bias_idx <= online_bias_idx + 3'd1;
                    state <= ST_ONLINE_READ;
                end
            end

            default: state <= ST_IDLE;
        endcase
    end
end

endmodule
