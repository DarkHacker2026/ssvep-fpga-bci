`timescale 1ns / 1ps

module top_ssvep #(
    parameter integer SIM_BYPASS          = 0,
    parameter integer DEBOUNCE_MAX        = 2_000_000,
    parameter integer N_CAL               = 16,
    parameter integer SHIFT               = 4,
    parameter integer USE_PREFILTER       = 0,
    parameter integer USE_HYBRID_FUSER    = 1,
    parameter integer ARTIFACT_THRESHOLD  = 24'd800000,
    parameter integer UNCERTAIN_LIMIT     = 4,
    parameter integer SIM_REPLAY_SHIFT    = 8
)(
    input  wire               clk,
    input  wire               rst_n,

    output wire               ads_sclk,
    output wire               ads_mosi,
    input  wire               ads_miso,
    output wire               ads_cs_n,
    output wire               ads_pwdn_n,
    output wire               ads_start,
    input  wire               ads_drdy_n,

    input  wire signed [23:0] sim_sample_in,
    input  wire               sim_sample_valid,

    input  wire               btn_yes,
    input  wire               btn_no,

    output reg  [1:0]         decision,
    output reg                decision_valid,
    output wire               calibrated,
    output wire               LED_ready,
    output wire               led_cal_yes,
    output wire               led_cal_no,
    output wire               led_computing,
    output wire               led_done,
    output reg                fault_alert
);

// ---------------------------------------------------------------------------
// Acquisition front-end
// ---------------------------------------------------------------------------
wire signed [23:0] ads_ch1, ads_ch2, ads_ch3;
wire               ads_sample_valid;
wire               artifact_active;

wire signed [17:0] spatial_w_ch1, spatial_w_ch2, spatial_w_ch3;
wire [7:0]         gain_ch1_q4, gain_ch2_q4, gain_ch3_q4;
wire signed [17:0] spatial_w_ch1_eff = SIM_BYPASS ? -18'sd32768 : spatial_w_ch1;
wire signed [17:0] spatial_w_ch2_eff = SIM_BYPASS ?  18'sd65536 : spatial_w_ch2;
wire signed [17:0] spatial_w_ch3_eff = SIM_BYPASS ? -18'sd32768 : spatial_w_ch3;

ads1299_interface ads_if_inst (
    .clk(clk),
    .rst_n(rst_n),
    .ads_sclk(ads_sclk),
    .ads_mosi(ads_mosi),
    .ads_miso(ads_miso),
    .ads_cs_n(ads_cs_n),
    .ads_pwdn_n(ads_pwdn_n),
    .ads_start(ads_start),
    .ads_drdy_n(ads_drdy_n),
    .gain_ch1_q4(gain_ch1_q4),
    .gain_ch2_q4(gain_ch2_q4),
    .gain_ch3_q4(gain_ch3_q4),
    .artifact_threshold(ARTIFACT_THRESHOLD),
    .sample_ch1(ads_ch1),
    .sample_ch2(ads_ch2),
    .sample_ch3(ads_ch3),
    .sample_valid(ads_sample_valid),
    .artifact_active(artifact_active)
);

// In simulation replay mode we inject a pre-combined single stream. Route it
// onto CH2 only, with an RTL-only gain boost, so tiny replay integers still
// survive the fixed-point MAC stages. The default spatial prior (+2 on CH2)
// reconstructs the effective single-channel stream.
wire signed [23:0] in_ch1 = SIM_BYPASS ? 24'sd0 : ads_ch1;
wire signed [23:0] in_ch2 = SIM_BYPASS ? (sim_sample_in <<< SIM_REPLAY_SHIFT) : ads_ch2;
wire signed [23:0] in_ch3 = SIM_BYPASS ? 24'sd0 : ads_ch3;
wire               in_valid = SIM_BYPASS ? sim_sample_valid : ads_sample_valid;

// ---------------------------------------------------------------------------
// Coherent multichannel filter bank
// ---------------------------------------------------------------------------
wire [12*18-1:0] feature_flat;
wire             features_valid;
wire signed [23:0] mean_ch1, mean_ch2, mean_ch3;
wire [15:0]      window_index;

matched_filter_bank filter_bank_inst (
    .clk(clk),
    .rst_n(rst_n),
    .sample_ch1(in_ch1),
    .sample_ch2(in_ch2),
    .sample_ch3(in_ch3),
    .sample_valid(in_valid),
    .spatial_w_ch1(spatial_w_ch1_eff),
    .spatial_w_ch2(spatial_w_ch2_eff),
    .spatial_w_ch3(spatial_w_ch3_eff),
    .feature_flat(feature_flat),
    .features_valid(features_valid),
    .mean_ch1(mean_ch1),
    .mean_ch2(mean_ch2),
    .mean_ch3(mean_ch3),
    .window_index(window_index)
);

// Debug aliases for the active 15 Hz / 20 Hz demo classes.
wire [17:0] feat15_h1 = feature_flat[6*18 +: 18];
wire [17:0] feat15_h2 = feature_flat[7*18 +: 18];
wire [17:0] feat15_h3 = feature_flat[8*18 +: 18];
wire [17:0] feat20_h1 = feature_flat[9*18 +: 18];
wire [17:0] feat20_h2 = feature_flat[10*18 +: 18];
wire [17:0] feat20_h3 = feature_flat[11*18 +: 18];
wire [19:0] demo_score_yes = {2'b00, feat15_h1} + {2'b00, feat15_h2} + {2'b00, feat15_h3};
wire [19:0] demo_score_no  = {2'b00, feat20_h1} + {2'b00, feat20_h2} + {2'b00, feat20_h3};
wire [17:0] p15_norm = demo_score_yes[17:0];
wire [17:0] p20_norm = demo_score_no[17:0];
wire [17:0] p30_norm = feat15_h1;
wire [17:0] p40_norm = feat20_h1;
wire [1:0]  replay_vote_class = (demo_score_no > demo_score_yes) ? 2'd1 : 2'd0;
wire [19:0] replay_margin = (demo_score_yes >= demo_score_no) ?
                            (demo_score_yes - demo_score_no) :
                            (demo_score_no - demo_score_yes);

// ---------------------------------------------------------------------------
// RAM-backed neural network
// ---------------------------------------------------------------------------
wire [32*18-1:0] h1_flat;
wire [8*18-1:0]  h2_flat;
wire [4*18-1:0]  logit_flat;
wire             raw_decision_valid;
wire [1:0]       raw_decision;
wire [17:0]      logit_margin;
wire             nn_wr_en;
wire [9:0]       nn_wr_addr;
wire signed [17:0] nn_wr_data;
wire [9:0]       nn_rd_addr;
wire signed [17:0] nn_rd_data;

neural_network nn_inst (
    .clk(clk),
    .rst_n(rst_n),
    .input_valid(features_valid),
    .feature_flat(feature_flat),
    .cal_wr_en(nn_wr_en),
    .cal_wr_addr(nn_wr_addr),
    .cal_wr_data(nn_wr_data),
    .cal_rd_addr(nn_rd_addr),
    .cal_rd_data(nn_rd_data),
    .h1_flat(h1_flat),
    .h2_flat(h2_flat),
    .logit_flat(logit_flat),
    .output_valid(raw_decision_valid),
    .class_idx(raw_decision),
    .class_margin(logit_margin)
);

// ---------------------------------------------------------------------------
// On-chip calibrator / online adaptation
// ---------------------------------------------------------------------------
on_chip_calibrator #(
    .N_FEATURES(12),
    .N_H1(32),
    .N_H2(8),
    .N_OUT(4),
    .N_CAL(N_CAL),
    .DEBOUNCE_MAX(DEBOUNCE_MAX)
) cal_inst (
    .clk(clk),
    .rst_n(rst_n),
    .feature_flat(feature_flat),
    .features_valid(features_valid),
    .mean_ch1(mean_ch1),
    .mean_ch2(mean_ch2),
    .mean_ch3(mean_ch3),
    .h1_flat(h1_flat),
    .h2_flat(h2_flat),
    .logit_flat(logit_flat),
    .pred_class(raw_decision),
    .pred_margin(logit_margin),
    .output_valid(raw_decision_valid),
    .btn_yes(btn_yes),
    .btn_no(btn_no),
    .nn_wr_en(nn_wr_en),
    .nn_wr_addr(nn_wr_addr),
    .nn_wr_data(nn_wr_data),
    .nn_rd_addr(nn_rd_addr),
    .nn_rd_data(nn_rd_data),
    .spatial_w_ch1(spatial_w_ch1),
    .spatial_w_ch2(spatial_w_ch2),
    .spatial_w_ch3(spatial_w_ch3),
    .gain_ch1_q4(gain_ch1_q4),
    .gain_ch2_q4(gain_ch2_q4),
    .gain_ch3_q4(gain_ch3_q4),
    .calibrated(calibrated),
    .led_cal_yes(led_cal_yes),
    .led_cal_no(led_cal_no),
    .led_computing(led_computing),
    .led_done(led_done)
);

assign LED_ready = rst_n && !led_cal_yes && !led_cal_no && !led_computing;

// ---------------------------------------------------------------------------
// EMA + overlap-aware reject logic
//   - EMA integrates the 4 output logits
//   - 5-window history requires a stable majority before asserting valid
//   - if the history is split or EMA margin is weak, hold the previous output
// ---------------------------------------------------------------------------
reg signed [19:0] ema_logit [0:3];
reg [1:0]         hist0, hist1, hist2, hist3, hist4;
reg [2:0]         hist_count;
integer           cls;
integer           c0, c1, c2, c3;
reg signed [19:0] best_ema;
reg signed [19:0] second_ema;
reg [1:0]         best_cls;
integer           best_count;
reg               accept_decision;
reg [7:0]         uncertain_streak;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        decision <= 2'b00;
        decision_valid <= 1'b0;
        fault_alert <= 1'b0;
        hist0 <= 2'b00; hist1 <= 2'b00; hist2 <= 2'b00; hist3 <= 2'b00; hist4 <= 2'b00;
        hist_count <= 3'd0;
        uncertain_streak <= 8'd0;
        for (cls = 0; cls < 4; cls = cls + 1)
            ema_logit[cls] <= 20'sd0;
    end else begin
        decision_valid <= 1'b0;

        if (raw_decision_valid) begin
            for (cls = 0; cls < 4; cls = cls + 1) begin
                ema_logit[cls] <= ema_logit[cls] +
                    (($signed({{2{logit_flat[cls*18+17]}}, logit_flat[cls*18 +: 18]}) - ema_logit[cls]) >>> 2);
            end

            hist4 <= hist3;
            hist3 <= hist2;
            hist2 <= hist1;
            hist1 <= hist0;
            hist0 <= raw_decision;
            if (hist_count < 3'd5)
                hist_count <= hist_count + 3'd1;

            best_ema   = ema_logit[0];
            second_ema = -20'sd524288;
            best_cls   = 2'd0;
            for (cls = 0; cls < 4; cls = cls + 1) begin
                if (ema_logit[cls] > best_ema) begin
                    second_ema = best_ema;
                    best_ema   = ema_logit[cls];
                    best_cls   = cls[1:0];
                end else if (ema_logit[cls] > second_ema) begin
                    second_ema = ema_logit[cls];
                end
            end

            c0 = (raw_decision == 2'd0) + (hist0 == 2'd0) + (hist1 == 2'd0) + (hist2 == 2'd0) + (hist3 == 2'd0);
            c1 = (raw_decision == 2'd1) + (hist0 == 2'd1) + (hist1 == 2'd1) + (hist2 == 2'd1) + (hist3 == 2'd1);
            c2 = (raw_decision == 2'd2) + (hist0 == 2'd2) + (hist1 == 2'd2) + (hist2 == 2'd2) + (hist3 == 2'd2);
            c3 = (raw_decision == 2'd3) + (hist0 == 2'd3) + (hist1 == 2'd3) + (hist2 == 2'd3) + (hist3 == 2'd3);

            best_count = c0;
            if (c1 > best_count) best_count = c1;
            if (c2 > best_count) best_count = c2;
            if (c3 > best_count) best_count = c3;
            accept_decision = (hist_count >= 3'd4) &&
                              (best_count >= 3) &&
                              ((best_ema - second_ema) > 20'sd4096);

            if (accept_decision) begin
                decision <= best_cls;
                decision_valid <= 1'b1;
                uncertain_streak <= 8'd0;
                fault_alert <= 1'b0;
            end else begin
                if (uncertain_streak < 8'hff)
                    uncertain_streak <= uncertain_streak + 8'd1;
                if ((uncertain_streak + 8'd1) >= UNCERTAIN_LIMIT)
                    fault_alert <= 1'b1;
            end
        end
    end
end

endmodule
