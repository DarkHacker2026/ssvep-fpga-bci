`timescale 1ns / 1ps
`include "neural_network_weights.vh"
// =============================================================================
// neural_network.v
//
// RAM-backed 12 -> 32 -> 8 -> 4 MLP with:
//   - leaky ReLU in the hidden layers
//   - soft-sign / Pade-style tanh approximation on the logits
//   - a flat calibration write address space for all weights and biases
//   - a calibration readback port so the on-chip calibrator can propagate
//     its LDA direction through the whole network
// =============================================================================

module neural_network #(
    parameter integer N_IN   = 12,
    parameter integer N_H1   = 32,
    parameter integer N_H2   = 8,
    parameter integer N_OUT  = 4,
    parameter integer Q_SHIFT= 16
)(
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     input_valid,
    input  wire [N_IN*18-1:0]       feature_flat,

    input  wire                     cal_wr_en,
    input  wire [9:0]               cal_wr_addr,
    input  wire signed [17:0]       cal_wr_data,
    input  wire [9:0]               cal_rd_addr,
    output reg  signed [17:0]       cal_rd_data,

    output reg  [N_H1*18-1:0]       h1_flat,
    output reg  [N_H2*18-1:0]       h2_flat,
    output reg  [N_OUT*18-1:0]      logit_flat,
    output reg                      output_valid,
    output reg  [1:0]               class_idx,
    output reg  [17:0]              class_margin
);

localparam integer FC1_W_COUNT = N_IN * N_H1;
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
    ST_IDLE   = 4'd0,
    ST_LOAD   = 4'd1,
    ST_FC1    = 4'd2,
    ST_ACT1   = 4'd3,
    ST_FC2    = 4'd4,
    ST_ACT2   = 4'd5,
    ST_FC3    = 4'd6,
    ST_LOGITS = 4'd7,
    ST_DONE   = 4'd8;

reg [3:0] state;

reg signed [17:0] fc1_w [0:FC1_W_COUNT-1];
reg signed [17:0] fc1_b [0:N_H1-1];
reg signed [17:0] fc2_w [0:FC2_W_COUNT-1];
reg signed [17:0] fc2_b [0:N_H2-1];
reg signed [17:0] fc3_w [0:FC3_W_COUNT-1];
reg signed [17:0] fc3_b [0:N_OUT-1];

reg signed [17:0] x_vec [0:N_IN-1];
reg signed [17:0] h1    [0:N_H1-1];
reg signed [17:0] h2    [0:N_H2-1];
reg signed [17:0] logit [0:N_OUT-1];

reg [5:0] row_idx;
reg [5:0] col_idx;
reg signed [47:0] acc;
reg signed [17:0] best_logit;
reg signed [17:0] second_logit;
reg [1:0]         best_class;
integer i;

initial begin
`INIT_NN_WEIGHTS
end

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

function signed [17:0] leaky_relu18;
    input signed [17:0] x;
    begin
        if (x[17])
            leaky_relu18 = x >>> 5;
        else
            leaky_relu18 = x;
    end
endfunction

function signed [17:0] pade_tanh18;
    input signed [17:0] x;
    reg [17:0] ax;
    reg signed [17:0] recip;
    reg signed [35:0] prod;
    begin
        ax = x[17] ? (~x + 18'd1) : x[17:0];
        if (ax < 18'd8192)       recip = 18'sd61166;  // ~0.933
        else if (ax < 18'd16384) recip = 18'sd54613;  // ~0.833
        else if (ax < 18'd32768) recip = 18'sd46811;  // ~0.714
        else if (ax < 18'd49152) recip = 18'sd39322;  // ~0.600
        else if (ax < 18'd65536) recip = 18'sd32768;  // ~0.500
        else if (ax < 18'd81920) recip = 18'sd28399;  // ~0.433
        else if (ax < 18'd98304) recip = 18'sd24904;  // ~0.380
        else                     recip = 18'sd21845;  // ~0.333
        prod = $signed(x) * $signed(recip);
        pade_tanh18 = sat18(prod >>> 16);
    end
endfunction

always @(*) begin
    if (cal_rd_addr < FC1_W_COUNT)
        cal_rd_data = fc1_w[cal_rd_addr];
    else if (cal_rd_addr < FC1_B_BASE + FC1_B_COUNT)
        cal_rd_data = fc1_b[cal_rd_addr - FC1_B_BASE];
    else if (cal_rd_addr < FC2_W_BASE + FC2_W_COUNT)
        cal_rd_data = fc2_w[cal_rd_addr - FC2_W_BASE];
    else if (cal_rd_addr < FC2_B_BASE + FC2_B_COUNT)
        cal_rd_data = fc2_b[cal_rd_addr - FC2_B_BASE];
    else if (cal_rd_addr < FC3_W_BASE + FC3_W_COUNT)
        cal_rd_data = fc3_w[cal_rd_addr - FC3_W_BASE];
    else if (cal_rd_addr < TOTAL_COUNT)
        cal_rd_data = fc3_b[cal_rd_addr - FC3_B_BASE];
    else
        cal_rd_data = 18'sd0;
end

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state <= ST_IDLE;
        row_idx <= 6'd0;
        col_idx <= 6'd0;
        acc <= 48'sd0;
        class_idx <= 2'd0;
        class_margin <= 18'd0;
        output_valid <= 1'b0;
        h1_flat <= {(N_H1*18){1'b0}};
        h2_flat <= {(N_H2*18){1'b0}};
        logit_flat <= {(N_OUT*18){1'b0}};
        for (i = 0; i < N_IN; i = i + 1)
            x_vec[i] <= 18'sd0;
        for (i = 0; i < N_H1; i = i + 1)
            h1[i] <= 18'sd0;
        for (i = 0; i < N_H2; i = i + 1)
            h2[i] <= 18'sd0;
        for (i = 0; i < N_OUT; i = i + 1)
            logit[i] <= 18'sd0;
    end else begin
        output_valid <= 1'b0;

        if (cal_wr_en) begin
            if (cal_wr_addr < FC1_W_COUNT)
                fc1_w[cal_wr_addr] <= cal_wr_data;
            else if (cal_wr_addr < FC1_B_BASE + FC1_B_COUNT)
                fc1_b[cal_wr_addr - FC1_B_BASE] <= cal_wr_data;
            else if (cal_wr_addr < FC2_W_BASE + FC2_W_COUNT)
                fc2_w[cal_wr_addr - FC2_W_BASE] <= cal_wr_data;
            else if (cal_wr_addr < FC2_B_BASE + FC2_B_COUNT)
                fc2_b[cal_wr_addr - FC2_B_BASE] <= cal_wr_data;
            else if (cal_wr_addr < FC3_W_BASE + FC3_W_COUNT)
                fc3_w[cal_wr_addr - FC3_W_BASE] <= cal_wr_data;
            else if (cal_wr_addr < TOTAL_COUNT)
                fc3_b[cal_wr_addr - FC3_B_BASE] <= cal_wr_data;
        end

        case (state)
            ST_IDLE: begin
                if (input_valid)
                    state <= ST_LOAD;
            end

            ST_LOAD: begin
                for (i = 0; i < N_IN; i = i + 1)
                    x_vec[i] <= feature_flat[i*18 +: 18];
                row_idx <= 6'd0;
                col_idx <= 6'd0;
                acc <= {{30{fc1_b[0][17]}}, fc1_b[0]};
                state <= ST_FC1;
            end

            ST_FC1: begin
                acc <= acc + (($signed(x_vec[col_idx]) * $signed(fc1_w[row_idx*N_IN + col_idx])) >>> Q_SHIFT);
                if (col_idx == N_IN-1) begin
                    state <= ST_ACT1;
                end else begin
                    col_idx <= col_idx + 6'd1;
                end
            end

            ST_ACT1: begin
                h1[row_idx] <= leaky_relu18(sat18(acc));
                if (row_idx == N_H1-1) begin
                    for (i = 0; i < N_H1; i = i + 1)
                        h1_flat[i*18 +: 18] <= (i == row_idx) ? leaky_relu18(sat18(acc)) : h1[i];
                    row_idx <= 6'd0;
                    col_idx <= 6'd0;
                    acc <= {{30{fc2_b[0][17]}}, fc2_b[0]};
                    state <= ST_FC2;
                end else begin
                    h1_flat[row_idx*18 +: 18] <= leaky_relu18(sat18(acc));
                    row_idx <= row_idx + 6'd1;
                    col_idx <= 6'd0;
                    acc <= {{30{fc1_b[row_idx+1][17]}}, fc1_b[row_idx+1]};
                    state <= ST_FC1;
                end
            end

            ST_FC2: begin
                acc <= acc + (($signed(h1[col_idx]) * $signed(fc2_w[row_idx*N_H1 + col_idx])) >>> Q_SHIFT);
                if (col_idx == N_H1-1) begin
                    state <= ST_ACT2;
                end else begin
                    col_idx <= col_idx + 6'd1;
                end
            end

            ST_ACT2: begin
                h2[row_idx] <= leaky_relu18(sat18(acc));
                h2_flat[row_idx*18 +: 18] <= leaky_relu18(sat18(acc));
                if (row_idx == N_H2-1) begin
                    row_idx <= 6'd0;
                    col_idx <= 6'd0;
                    acc <= {{30{fc3_b[0][17]}}, fc3_b[0]};
                    state <= ST_FC3;
                end else begin
                    row_idx <= row_idx + 6'd1;
                    col_idx <= 6'd0;
                    acc <= {{30{fc2_b[row_idx+1][17]}}, fc2_b[row_idx+1]};
                    state <= ST_FC2;
                end
            end

            ST_FC3: begin
                acc <= acc + (($signed(h2[col_idx]) * $signed(fc3_w[row_idx*N_H2 + col_idx])) >>> Q_SHIFT);
                if (col_idx == N_H2-1) begin
                    state <= ST_LOGITS;
                end else begin
                    col_idx <= col_idx + 6'd1;
                end
            end

            ST_LOGITS: begin
                logit[row_idx] <= pade_tanh18(sat18(acc));
                logit_flat[row_idx*18 +: 18] <= pade_tanh18(sat18(acc));
                if (row_idx == N_OUT-1) begin
                    best_logit   = logit[0];
                    second_logit = -18'sd131072;
                    best_class   = 2'd0;

                    for (i = 0; i < N_OUT; i = i + 1) begin
                        if ((i == row_idx ? pade_tanh18(sat18(acc)) : logit[i]) > best_logit) begin
                            second_logit = best_logit;
                            best_logit   = (i == row_idx ? pade_tanh18(sat18(acc)) : logit[i]);
                            best_class   = i[1:0];
                        end else if ((i == row_idx ? pade_tanh18(sat18(acc)) : logit[i]) > second_logit) begin
                            second_logit = (i == row_idx ? pade_tanh18(sat18(acc)) : logit[i]);
                        end
                    end

                    class_idx    <= best_class;
                    class_margin <= best_logit - second_logit;
                    state        <= ST_DONE;
                end else begin
                    row_idx <= row_idx + 6'd1;
                    col_idx <= 6'd0;
                    acc <= {{30{fc3_b[row_idx+1][17]}}, fc3_b[row_idx+1]};
                    state <= ST_FC3;
                end
            end

            ST_DONE: begin
                output_valid <= 1'b1;
                state <= ST_IDLE;
            end

            default: state <= ST_IDLE;
        endcase
    end
end

endmodule
