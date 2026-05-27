`timescale 1ns / 1ps
`include "filter_coeffs.vh"
// =============================================================================
// matched_filter_bank.v
//
// 3-channel coherent detector with:
//   - 4 targets x 3 harmonics = 12 features
//   - separate per-channel coherent accumulators
//   - learned spatial combination via channel weights
//   - 512-sample windows with 256-sample hop
//   - a small 4-stage internal MAC pipeline
//
// The engine is intentionally time-multiplexed: EEG samples arrive at only
// 250 Hz, so the detector can spend thousands of 100 MHz cycles evaluating a
// window and still remain comfortably real time.
// =============================================================================

module matched_filter_bank #(
    parameter integer WINDOW_LEN   = 512,
    parameter integer HOP_LEN      = 256,
    parameter integer N_FEATURES   = 12,
    parameter integer COEFF_SHIFT  = 12,
    parameter integer SPATIAL_SHIFT= 16
)(
    input  wire                        clk,
    input  wire                        rst_n,
    input  wire signed [23:0]          sample_ch1,
    input  wire signed [23:0]          sample_ch2,
    input  wire signed [23:0]          sample_ch3,
    input  wire                        sample_valid,
    input  wire signed [17:0]          spatial_w_ch1,
    input  wire signed [17:0]          spatial_w_ch2,
    input  wire signed [17:0]          spatial_w_ch3,
    output reg  [N_FEATURES*18-1:0]    feature_flat,
    output reg                         features_valid,
    output reg  signed [23:0]          mean_ch1,
    output reg  signed [23:0]          mean_ch2,
    output reg  signed [23:0]          mean_ch3,
    output reg  [15:0]                 window_index
);

localparam [2:0]
    ST_IDLE    = 3'd0,
    ST_ACCUM   = 3'd1,
    ST_FLUSH   = 3'd2,
    ST_COMBINE = 3'd3,
    ST_PACK    = 3'd4;

reg [2:0] state;

reg signed [23:0] ring_ch1 [0:WINDOW_LEN-1];
reg signed [23:0] ring_ch2 [0:WINDOW_LEN-1];
reg signed [23:0] ring_ch3 [0:WINDOW_LEN-1];

reg signed [15:0] coeff_sin [0:(N_FEATURES*WINDOW_LEN)-1];
reg signed [15:0] coeff_cos [0:(N_FEATURES*WINDOW_LEN)-1];

reg signed [63:0] sin_acc_ch1 [0:N_FEATURES-1];
reg signed [63:0] sin_acc_ch2 [0:N_FEATURES-1];
reg signed [63:0] sin_acc_ch3 [0:N_FEATURES-1];
reg signed [63:0] cos_acc_ch1 [0:N_FEATURES-1];
reg signed [63:0] cos_acc_ch2 [0:N_FEATURES-1];
reg signed [63:0] cos_acc_ch3 [0:N_FEATURES-1];

reg signed [17:0] feature_reg [0:N_FEATURES-1];

reg [8:0]  wr_ptr;
reg [9:0]  sample_count;
reg [7:0]  hop_count;

reg [8:0]  tap_idx;
reg [3:0]  feat_idx;
reg [3:0]  pipe_valid;

reg signed [23:0] s0_ch1, s0_ch2, s0_ch3;
reg signed [15:0] s0_sin, s0_cos;
reg [3:0]         s0_feat;
reg               s0_sum_en;

reg signed [39:0] s1_mul_sin_ch1, s1_mul_sin_ch2, s1_mul_sin_ch3;
reg signed [39:0] s1_mul_cos_ch1, s1_mul_cos_ch2, s1_mul_cos_ch3;
reg signed [23:0] s1_raw_ch1, s1_raw_ch2, s1_raw_ch3;
reg [3:0]         s1_feat;
reg               s1_sum_en;

reg signed [31:0] s2_add_sin_ch1, s2_add_sin_ch2, s2_add_sin_ch3;
reg signed [31:0] s2_add_cos_ch1, s2_add_cos_ch2, s2_add_cos_ch3;
reg signed [23:0] s2_raw_ch1, s2_raw_ch2, s2_raw_ch3;
reg [3:0]         s2_feat;
reg               s2_sum_en;

reg signed [63:0] sum_ch1;
reg signed [63:0] sum_ch2;
reg signed [63:0] sum_ch3;

reg signed [63:0] mix_sin;
reg signed [63:0] mix_cos;
reg signed [63:0] mag_sq;

integer i;

initial begin
`INIT_FILTER_COEFFS
end

function signed [17:0] sat18;
    input signed [63:0] v;
    begin
        if (v > 64'sd131071)
            sat18 = 18'sd131071;
        else if (v < -64'sd131072)
            sat18 = -18'sd131072;
        else
            sat18 = v[17:0];
    end
endfunction

function [63:0] abs64;
    input signed [63:0] v;
    begin
        abs64 = v[63] ? (~v + 64'd1) : v[63:0];
    end
endfunction

function [31:0] isqrt64;
    input [63:0] x;
    reg [63:0] rem;
    reg [31:0] root;
    reg [33:0] trial;
    integer bitn;
    begin
        rem  = 64'd0;
        root = 32'd0;
        for (bitn = 0; bitn < 32; bitn = bitn + 1) begin
            rem  = {rem[61:0], x[63-(bitn*2)], x[62-(bitn*2)]};
            trial = {root, 2'b01};
            if (rem >= trial) begin
                rem  = rem - trial;
                root = {root[30:0], 1'b1};
            end else begin
                root = {root[30:0], 1'b0};
            end
        end
        isqrt64 = root;
    end
endfunction

wire [8:0] ring_rd_idx = wr_ptr + tap_idx;
wire [13:0] coeff_idx  = (feat_idx * WINDOW_LEN) + tap_idx;
wire signed [63:0] mix_sin_w =
    (($signed(sin_acc_ch1[feat_idx]) * $signed(spatial_w_ch1)) >>> SPATIAL_SHIFT) +
    (($signed(sin_acc_ch2[feat_idx]) * $signed(spatial_w_ch2)) >>> SPATIAL_SHIFT) +
    (($signed(sin_acc_ch3[feat_idx]) * $signed(spatial_w_ch3)) >>> SPATIAL_SHIFT);
wire signed [63:0] mix_cos_w =
    (($signed(cos_acc_ch1[feat_idx]) * $signed(spatial_w_ch1)) >>> SPATIAL_SHIFT) +
    (($signed(cos_acc_ch2[feat_idx]) * $signed(spatial_w_ch2)) >>> SPATIAL_SHIFT) +
    (($signed(cos_acc_ch3[feat_idx]) * $signed(spatial_w_ch3)) >>> SPATIAL_SHIFT);
wire [63:0] mag_sq_w =
    (($signed(mix_sin_w >>> 8) * $signed(mix_sin_w >>> 8)) >>> 8) +
    (($signed(mix_cos_w >>> 8) * $signed(mix_cos_w >>> 8)) >>> 8);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state        <= ST_IDLE;
        wr_ptr       <= 9'd0;
        sample_count <= 10'd0;
        hop_count    <= 8'd0;
        tap_idx      <= 9'd0;
        feat_idx     <= 4'd0;
        pipe_valid   <= 4'd0;
        sum_ch1      <= 64'sd0;
        sum_ch2      <= 64'sd0;
        sum_ch3      <= 64'sd0;
        features_valid <= 1'b0;
        feature_flat <= {(N_FEATURES*18){1'b0}};
        mean_ch1     <= 24'sd0;
        mean_ch2     <= 24'sd0;
        mean_ch3     <= 24'sd0;
        window_index <= 16'd0;
        for (i = 0; i < WINDOW_LEN; i = i + 1) begin
            ring_ch1[i] <= 24'sd0;
            ring_ch2[i] <= 24'sd0;
            ring_ch3[i] <= 24'sd0;
        end
        for (i = 0; i < N_FEATURES; i = i + 1) begin
            sin_acc_ch1[i] <= 64'sd0;
            sin_acc_ch2[i] <= 64'sd0;
            sin_acc_ch3[i] <= 64'sd0;
            cos_acc_ch1[i] <= 64'sd0;
            cos_acc_ch2[i] <= 64'sd0;
            cos_acc_ch3[i] <= 64'sd0;
            feature_reg[i] <= 18'sd0;
        end
        s0_ch1 <= 24'sd0; s0_ch2 <= 24'sd0; s0_ch3 <= 24'sd0;
        s0_sin <= 16'sd0; s0_cos <= 16'sd0; s0_feat <= 4'd0; s0_sum_en <= 1'b0;
        s1_mul_sin_ch1 <= 40'sd0; s1_mul_sin_ch2 <= 40'sd0; s1_mul_sin_ch3 <= 40'sd0;
        s1_mul_cos_ch1 <= 40'sd0; s1_mul_cos_ch2 <= 40'sd0; s1_mul_cos_ch3 <= 40'sd0;
        s1_raw_ch1 <= 24'sd0; s1_raw_ch2 <= 24'sd0; s1_raw_ch3 <= 24'sd0; s1_feat <= 4'd0; s1_sum_en <= 1'b0;
        s2_add_sin_ch1 <= 32'sd0; s2_add_sin_ch2 <= 32'sd0; s2_add_sin_ch3 <= 32'sd0;
        s2_add_cos_ch1 <= 32'sd0; s2_add_cos_ch2 <= 32'sd0; s2_add_cos_ch3 <= 32'sd0;
        s2_raw_ch1 <= 24'sd0; s2_raw_ch2 <= 24'sd0; s2_raw_ch3 <= 24'sd0; s2_feat <= 4'd0; s2_sum_en <= 1'b0;
    end else begin
        features_valid <= 1'b0;

        if (sample_valid) begin
            ring_ch1[wr_ptr] <= sample_ch1;
            ring_ch2[wr_ptr] <= sample_ch2;
            ring_ch3[wr_ptr] <= sample_ch3;
            wr_ptr <= wr_ptr + 9'd1;

            if (sample_count < WINDOW_LEN)
                sample_count <= sample_count + 10'd1;

            if (sample_count >= WINDOW_LEN) begin
                if (hop_count == HOP_LEN - 1)
                    hop_count <= 8'd0;
                else
                    hop_count <= hop_count + 8'd1;
            end else begin
                hop_count <= 8'd0;
            end
        end

        case (state)
            ST_IDLE: begin
                pipe_valid <= 4'b0000;
                if (sample_valid &&
                    (((sample_count + 10'd1) == WINDOW_LEN) ||
                     ((sample_count >= WINDOW_LEN) && (hop_count == HOP_LEN - 1)))) begin
                    tap_idx <= 9'd0;
                    feat_idx <= 4'd0;
                    sum_ch1 <= 64'sd0;
                    sum_ch2 <= 64'sd0;
                    sum_ch3 <= 64'sd0;
                    for (i = 0; i < N_FEATURES; i = i + 1) begin
                        sin_acc_ch1[i] <= 64'sd0;
                        sin_acc_ch2[i] <= 64'sd0;
                        sin_acc_ch3[i] <= 64'sd0;
                        cos_acc_ch1[i] <= 64'sd0;
                        cos_acc_ch2[i] <= 64'sd0;
                        cos_acc_ch3[i] <= 64'sd0;
                    end
                    state <= ST_ACCUM;
                end
            end

            ST_ACCUM: begin
                // Stage 0: fetch ring-buffer samples and detector coefficients.
                s0_ch1    <= ring_ch1[ring_rd_idx];
                s0_ch2    <= ring_ch2[ring_rd_idx];
                s0_ch3    <= ring_ch3[ring_rd_idx];
                s0_sin    <= coeff_sin[coeff_idx];
                s0_cos    <= coeff_cos[coeff_idx];
                s0_feat   <= feat_idx;
                s0_sum_en <= (feat_idx == 4'd0);
                pipe_valid<= {pipe_valid[2:0], 1'b1};

                // Stage 1: multiply.
                s1_mul_sin_ch1 <= $signed(s0_ch1) * $signed(s0_sin);
                s1_mul_sin_ch2 <= $signed(s0_ch2) * $signed(s0_sin);
                s1_mul_sin_ch3 <= $signed(s0_ch3) * $signed(s0_sin);
                s1_mul_cos_ch1 <= $signed(s0_ch1) * $signed(s0_cos);
                s1_mul_cos_ch2 <= $signed(s0_ch2) * $signed(s0_cos);
                s1_mul_cos_ch3 <= $signed(s0_ch3) * $signed(s0_cos);
                s1_raw_ch1     <= s0_ch1;
                s1_raw_ch2     <= s0_ch2;
                s1_raw_ch3     <= s0_ch3;
                s1_feat        <= s0_feat;
                s1_sum_en      <= s0_sum_en;

                // Stage 2: rescale products.
                s2_add_sin_ch1 <= s1_mul_sin_ch1 >>> COEFF_SHIFT;
                s2_add_sin_ch2 <= s1_mul_sin_ch2 >>> COEFF_SHIFT;
                s2_add_sin_ch3 <= s1_mul_sin_ch3 >>> COEFF_SHIFT;
                s2_add_cos_ch1 <= s1_mul_cos_ch1 >>> COEFF_SHIFT;
                s2_add_cos_ch2 <= s1_mul_cos_ch2 >>> COEFF_SHIFT;
                s2_add_cos_ch3 <= s1_mul_cos_ch3 >>> COEFF_SHIFT;
                s2_raw_ch1     <= s1_raw_ch1;
                s2_raw_ch2     <= s1_raw_ch2;
                s2_raw_ch3     <= s1_raw_ch3;
                s2_feat        <= s1_feat;
                s2_sum_en      <= s1_sum_en;

                // Stage 3: commit.
                if (pipe_valid[3]) begin
                    sin_acc_ch1[s2_feat] <= sin_acc_ch1[s2_feat] + {{32{s2_add_sin_ch1[31]}}, s2_add_sin_ch1};
                    sin_acc_ch2[s2_feat] <= sin_acc_ch2[s2_feat] + {{32{s2_add_sin_ch2[31]}}, s2_add_sin_ch2};
                    sin_acc_ch3[s2_feat] <= sin_acc_ch3[s2_feat] + {{32{s2_add_sin_ch3[31]}}, s2_add_sin_ch3};
                    cos_acc_ch1[s2_feat] <= cos_acc_ch1[s2_feat] + {{32{s2_add_cos_ch1[31]}}, s2_add_cos_ch1};
                    cos_acc_ch2[s2_feat] <= cos_acc_ch2[s2_feat] + {{32{s2_add_cos_ch2[31]}}, s2_add_cos_ch2};
                    cos_acc_ch3[s2_feat] <= cos_acc_ch3[s2_feat] + {{32{s2_add_cos_ch3[31]}}, s2_add_cos_ch3};
                    if (s2_sum_en) begin
                        sum_ch1 <= sum_ch1 + {{40{s2_raw_ch1[23]}}, s2_raw_ch1};
                        sum_ch2 <= sum_ch2 + {{40{s2_raw_ch2[23]}}, s2_raw_ch2};
                        sum_ch3 <= sum_ch3 + {{40{s2_raw_ch3[23]}}, s2_raw_ch3};
                    end
                end

                if (feat_idx == N_FEATURES-1) begin
                    feat_idx <= 4'd0;
                    if (tap_idx == WINDOW_LEN-1)
                        state <= ST_FLUSH;
                    else
                        tap_idx <= tap_idx + 9'd1;
                end else begin
                    feat_idx <= feat_idx + 4'd1;
                end
            end

            ST_FLUSH: begin
                pipe_valid <= {pipe_valid[2:0], 1'b0};
                if (pipe_valid[3]) begin
                    sin_acc_ch1[s2_feat] <= sin_acc_ch1[s2_feat] + {{32{s2_add_sin_ch1[31]}}, s2_add_sin_ch1};
                    sin_acc_ch2[s2_feat] <= sin_acc_ch2[s2_feat] + {{32{s2_add_sin_ch2[31]}}, s2_add_sin_ch2};
                    sin_acc_ch3[s2_feat] <= sin_acc_ch3[s2_feat] + {{32{s2_add_sin_ch3[31]}}, s2_add_sin_ch3};
                    cos_acc_ch1[s2_feat] <= cos_acc_ch1[s2_feat] + {{32{s2_add_cos_ch1[31]}}, s2_add_cos_ch1};
                    cos_acc_ch2[s2_feat] <= cos_acc_ch2[s2_feat] + {{32{s2_add_cos_ch2[31]}}, s2_add_cos_ch2};
                    cos_acc_ch3[s2_feat] <= cos_acc_ch3[s2_feat] + {{32{s2_add_cos_ch3[31]}}, s2_add_cos_ch3};
                    if (s2_sum_en) begin
                        sum_ch1 <= sum_ch1 + {{40{s2_raw_ch1[23]}}, s2_raw_ch1};
                        sum_ch2 <= sum_ch2 + {{40{s2_raw_ch2[23]}}, s2_raw_ch2};
                        sum_ch3 <= sum_ch3 + {{40{s2_raw_ch3[23]}}, s2_raw_ch3};
                    end
                end
                if (pipe_valid == 4'b0000) begin
                    feat_idx <= 4'd0;
                    state <= ST_COMBINE;
                end
            end

            ST_COMBINE: begin
                mix_sin <= mix_sin_w;
                mix_cos <= mix_cos_w;
                mag_sq  <= mag_sq_w;
                feature_reg[feat_idx] <= sat18(isqrt64(mag_sq_w));

                if (feat_idx == N_FEATURES-1) begin
                    feat_idx <= 4'd0;
                    state <= ST_PACK;
                end else begin
                    feat_idx <= feat_idx + 4'd1;
                end
            end

            ST_PACK: begin
                for (i = 0; i < N_FEATURES; i = i + 1)
                    feature_flat[i*18 +: 18] <= feature_reg[i];

                mean_ch1 <= sum_ch1 >>> 9;
                mean_ch2 <= sum_ch2 >>> 9;
                mean_ch3 <= sum_ch3 >>> 9;
                features_valid <= 1'b1;
                window_index <= window_index + 16'd1;
                state <= ST_IDLE;
            end

            default: state <= ST_IDLE;
        endcase
    end
end

endmodule
