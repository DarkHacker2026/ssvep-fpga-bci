`timescale 1ns / 1ps
// =============================================================================
// ads1299_interface.v
//
// 3-channel ADS1299 front-end with:
//   - independent CH1/CH2/CH3 outputs
//   - common-average reference (CAR)
//   - per-channel Q4 gain multipliers
//   - 1st-order high-pass + 2nd-order 50 Hz notch
//   - artifact blanking when the rolling 16-sample peak crosses threshold
//
// Notes:
//   * The runtime gains are provided by the on-chip calibrator.
//   * The blanking logic suppresses sample_valid for 32 EEG samples while the
//     internal filter states keep running, which avoids filter discontinuities.
// =============================================================================

module ads1299_interface (
    input  wire               clk,
    input  wire               rst_n,

    output reg                ads_sclk,
    output reg                ads_mosi,
    input  wire               ads_miso,
    output reg                ads_cs_n,
    output reg                ads_pwdn_n,
    output reg                ads_start,
    input  wire               ads_drdy_n,

    input  wire [7:0]         gain_ch1_q4,
    input  wire [7:0]         gain_ch2_q4,
    input  wire [7:0]         gain_ch3_q4,
    input  wire [23:0]        artifact_threshold,

    output reg  signed [23:0] sample_ch1,
    output reg  signed [23:0] sample_ch2,
    output reg  signed [23:0] sample_ch3,
    output reg                sample_valid,
    output reg                artifact_active
);

localparam HALF_PER  = 6'd50;
localparam N_INIT    = 6'd33;
localparam N_FRAME   = 5'd27;
localparam DELAY_1MS = 20'd100_000;

// HPF alpha ~= 0.98 in Q1.15.
localparam signed [15:0] HPF_ALPHA = 16'sd32113;

// 50 Hz notch, fs = 250 Hz, Q1.15.
localparam signed [15:0] NOTCH_B0 = 16'sd32767;
localparam signed [15:0] NOTCH_B1 = -16'sd20252;
localparam signed [15:0] NOTCH_B2 = 16'sd32767;
localparam signed [15:0] NOTCH_A1 = -16'sd19239;
localparam signed [15:0] NOTCH_A2 = 16'sd29573;

localparam [2:0]
    S_RESET = 3'd0,
    S_WAIT  = 3'd1,
    S_INIT  = 3'd2,
    S_IDLE  = 3'd3,
    S_READ  = 3'd4,
    S_PROC  = 3'd5;

reg [2:0]  state;
reg [19:0] delay_cnt;
reg [5:0]  init_idx;

reg [8:0] init_rom [0:32];
initial begin
    init_rom[ 0] = {1'b1, 8'h11};
    init_rom[ 1] = {1'b0, 8'h41};
    init_rom[ 2] = {1'b0, 8'h00};
    init_rom[ 3] = {1'b1, 8'h96};
    init_rom[ 4] = {1'b0, 8'h43};
    init_rom[ 5] = {1'b0, 8'h00};
    init_rom[ 6] = {1'b1, 8'hE0};
    init_rom[ 7] = {1'b0, 8'h45};
    init_rom[ 8] = {1'b0, 8'h00};
    init_rom[ 9] = {1'b1, 8'h60};
    init_rom[10] = {1'b0, 8'h46};
    init_rom[11] = {1'b0, 8'h00};
    init_rom[12] = {1'b1, 8'h60};
    init_rom[13] = {1'b0, 8'h47};
    init_rom[14] = {1'b0, 8'h00};
    init_rom[15] = {1'b1, 8'h60};
    init_rom[16] = {1'b0, 8'h48};
    init_rom[17] = {1'b0, 8'h00};
    init_rom[18] = {1'b1, 8'h81};
    init_rom[19] = {1'b0, 8'h49};
    init_rom[20] = {1'b0, 8'h00};
    init_rom[21] = {1'b1, 8'h81};
    init_rom[22] = {1'b0, 8'h4A};
    init_rom[23] = {1'b0, 8'h00};
    init_rom[24] = {1'b1, 8'h81};
    init_rom[25] = {1'b0, 8'h4B};
    init_rom[26] = {1'b0, 8'h00};
    init_rom[27] = {1'b1, 8'h81};
    init_rom[28] = {1'b0, 8'h4C};
    init_rom[29] = {1'b0, 8'h00};
    init_rom[30] = {1'b1, 8'h81};
    init_rom[31] = {1'b1, 8'h08};
    init_rom[32] = {1'b0, 8'h10};
end

reg [5:0] half_cnt;
reg [2:0] bit_cnt;
reg       sclk_ph;
reg       byte_busy;
reg       byte_done;
reg [7:0] tx_byte;
reg [7:0] rx_byte;
reg [7:0] rx_shift;

reg [4:0] frame_byte_cnt;
reg [7:0] frame [0:26];

reg signed [23:0] ch1_raw;
reg signed [23:0] ch2_raw;
reg signed [23:0] ch3_raw;

reg drdy_meta, drdy_sync, drdy_sync_prev;

reg proc_d;
reg [5:0] blank_count;
reg [23:0] peak_hist [0:15];
integer peak_i;

reg signed [23:0] hpf_x1 [0:2];
reg signed [31:0] hpf_y1 [0:2];
reg signed [23:0] notch_x1 [0:2];
reg signed [23:0] notch_x2 [0:2];
reg signed [31:0] notch_y1 [0:2];
reg signed [31:0] notch_y2 [0:2];

reg signed [23:0] car0, car1, car2;
reg signed [31:0] gain0, gain1, gain2;
reg signed [31:0] hpf0, hpf1, hpf2;
reg signed [47:0] notch_acc0, notch_acc1, notch_acc2;
reg signed [23:0] notch0, notch1, notch2;
reg [23:0] local_peak;
integer ch;

function signed [23:0] sat24;
    input signed [47:0] v;
    begin
        if (v > 48'sd8388607)
            sat24 = 24'sd8388607;
        else if (v < -48'sd8388608)
            sat24 = -24'sd8388608;
        else
            sat24 = v[23:0];
    end
endfunction

function [23:0] abs24;
    input signed [23:0] v;
    begin
        abs24 = v[23] ? (~v + 24'd1) : v[23:0];
    end
endfunction

function [23:0] max3u24;
    input [23:0] a;
    input [23:0] b;
    input [23:0] c;
    reg [23:0] m;
    begin
        m = (a > b) ? a : b;
        max3u24 = (m > c) ? m : c;
    end
endfunction

function [23:0] rolling_max16;
    input unused;
    integer idx;
    reg [23:0] m;
    begin
        m = 24'd0;
        for (idx = 0; idx < 16; idx = idx + 1) begin
            if (peak_hist[idx] > m)
                m = peak_hist[idx];
        end
        rolling_max16 = m;
    end
endfunction

wire signed [25:0] ch_sum =
    {{2{ch1_raw[23]}}, ch1_raw} +
    {{2{ch2_raw[23]}}, ch2_raw} +
    {{2{ch3_raw[23]}}, ch3_raw};
wire signed [41:0] avg_full = ch_sum * $signed(18'sd21845);
wire signed [23:0] car_avg = avg_full[39:16];

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        ads_sclk  <= 1'b0;
        ads_mosi  <= 1'b0;
        half_cnt  <= 6'd0;
        bit_cnt   <= 3'd7;
        sclk_ph   <= 1'b0;
        byte_done <= 1'b0;
        rx_byte   <= 8'd0;
        rx_shift  <= 8'd0;
    end else begin
        byte_done <= 1'b0;
        if (byte_busy) begin
            half_cnt <= half_cnt + 6'd1;
            if (!sclk_ph) begin
                ads_sclk <= 1'b0;
                if (half_cnt == 6'd0)
                    ads_mosi <= tx_byte[bit_cnt];
                if (half_cnt == HALF_PER - 1) begin
                    sclk_ph  <= 1'b1;
                    half_cnt <= 6'd0;
                end
            end else begin
                ads_sclk <= 1'b1;
                if (half_cnt == 6'd0)
                    rx_shift[bit_cnt] <= ads_miso;
                if (half_cnt == HALF_PER - 1) begin
                    sclk_ph  <= 1'b0;
                    half_cnt <= 6'd0;
                    if (bit_cnt == 3'd0) begin
                        byte_done <= 1'b1;
                        rx_byte   <= {rx_shift[7:1], ads_miso};
                        bit_cnt   <= 3'd7;
                    end else begin
                        bit_cnt <= bit_cnt - 3'd1;
                    end
                end
            end
        end else begin
            half_cnt <= 6'd0;
            bit_cnt  <= 3'd7;
            sclk_ph  <= 1'b0;
        end
    end
end

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state          <= S_RESET;
        delay_cnt      <= 20'd0;
        init_idx       <= 6'd0;
        ads_cs_n       <= 1'b1;
        ads_pwdn_n     <= 1'b0;
        ads_start      <= 1'b0;
        byte_busy      <= 1'b0;
        tx_byte        <= 8'd0;
        frame_byte_cnt <= 5'd0;
        drdy_meta      <= 1'b1;
        drdy_sync      <= 1'b1;
        drdy_sync_prev <= 1'b1;
        ch1_raw        <= 24'sd0;
        ch2_raw        <= 24'sd0;
        ch3_raw        <= 24'sd0;
    end else begin
        drdy_meta      <= ads_drdy_n;
        drdy_sync      <= drdy_meta;
        drdy_sync_prev <= drdy_sync;

        case (state)
            S_RESET: begin
                ads_pwdn_n <= 1'b0;
                delay_cnt  <= delay_cnt + 20'd1;
                if (delay_cnt == DELAY_1MS - 1) begin
                    ads_pwdn_n <= 1'b1;
                    delay_cnt  <= 20'd0;
                    state      <= S_WAIT;
                end
            end

            S_WAIT: begin
                delay_cnt <= delay_cnt + 20'd1;
                if (delay_cnt == DELAY_1MS - 1) begin
                    delay_cnt <= 20'd0;
                    init_idx  <= 6'd0;
                    state     <= S_INIT;
                end
            end

            S_INIT: begin
                if (!byte_busy && !byte_done) begin
                    tx_byte   <= init_rom[init_idx][7:0];
                    ads_cs_n  <= 1'b0;
                    byte_busy <= 1'b1;
                end else if (byte_done) begin
                    byte_busy <= 1'b0;
                    if (init_rom[init_idx][8])
                        ads_cs_n <= 1'b1;
                    if (init_idx == N_INIT - 1) begin
                        ads_cs_n  <= 1'b0;
                        ads_start <= 1'b1;
                        state     <= S_IDLE;
                    end else begin
                        init_idx <= init_idx + 6'd1;
                    end
                end
            end

            S_IDLE: begin
                if (drdy_sync_prev && !drdy_sync) begin
                    frame_byte_cnt <= 5'd0;
                    byte_busy      <= 1'b0;
                    state          <= S_READ;
                end
            end

            S_READ: begin
                if (!byte_busy && !byte_done) begin
                    tx_byte   <= 8'hFF;
                    byte_busy <= 1'b1;
                end else if (byte_done) begin
                    byte_busy             <= 1'b0;
                    frame[frame_byte_cnt] <= rx_byte;
                    if (frame_byte_cnt == N_FRAME - 1)
                        state <= S_PROC;
                    else
                        frame_byte_cnt <= frame_byte_cnt + 5'd1;
                end
            end

            S_PROC: begin
                ch1_raw <= {frame[3],  frame[4],  frame[5]};
                ch2_raw <= {frame[6],  frame[7],  frame[8]};
                ch3_raw <= {frame[9],  frame[10], frame[11]};
                state   <= S_IDLE;
            end

            default: state <= S_IDLE;
        endcase
    end
end

always @(posedge clk or negedge rst_n) begin : proc_pipe
    if (!rst_n) begin
        proc_d         <= 1'b0;
        sample_ch1     <= 24'sd0;
        sample_ch2     <= 24'sd0;
        sample_ch3     <= 24'sd0;
        sample_valid   <= 1'b0;
        blank_count    <= 6'd0;
        artifact_active<= 1'b0;
        for (ch = 0; ch < 3; ch = ch + 1) begin
            hpf_x1[ch]   <= 24'sd0;
            hpf_y1[ch]   <= 32'sd0;
            notch_x1[ch] <= 24'sd0;
            notch_x2[ch] <= 24'sd0;
            notch_y1[ch] <= 32'sd0;
            notch_y2[ch] <= 32'sd0;
        end
        for (ch = 0; ch < 16; ch = ch + 1)
            peak_hist[ch] <= 24'd0;
    end else begin
        proc_d       <= (state == S_PROC);
        sample_valid <= 1'b0;

        if (proc_d) begin
            car0 = ch1_raw - car_avg;
            car1 = ch2_raw - car_avg;
            car2 = ch3_raw - car_avg;

            gain0 = ($signed(car0) * $signed({1'b0, gain_ch1_q4})) >>> 4;
            gain1 = ($signed(car1) * $signed({1'b0, gain_ch2_q4})) >>> 4;
            gain2 = ($signed(car2) * $signed({1'b0, gain_ch3_q4})) >>> 4;

            hpf0 = ($signed(HPF_ALPHA) * ($signed(gain0) - $signed({{8{hpf_x1[0][23]}}, hpf_x1[0]}) + hpf_y1[0])) >>> 15;
            hpf1 = ($signed(HPF_ALPHA) * ($signed(gain1) - $signed({{8{hpf_x1[1][23]}}, hpf_x1[1]}) + hpf_y1[1])) >>> 15;
            hpf2 = ($signed(HPF_ALPHA) * ($signed(gain2) - $signed({{8{hpf_x1[2][23]}}, hpf_x1[2]}) + hpf_y1[2])) >>> 15;

            hpf_x1[0] <= sat24(gain0); hpf_y1[0] <= hpf0;
            hpf_x1[1] <= sat24(gain1); hpf_y1[1] <= hpf1;
            hpf_x1[2] <= sat24(gain2); hpf_y1[2] <= hpf2;

            notch_acc0 =
                ($signed(NOTCH_B0) * $signed(hpf0)) +
                ($signed(NOTCH_B1) * $signed({{8{notch_x1[0][23]}}, notch_x1[0]})) +
                ($signed(NOTCH_B2) * $signed({{8{notch_x2[0][23]}}, notch_x2[0]})) -
                ($signed(NOTCH_A1) * $signed(notch_y1[0])) -
                ($signed(NOTCH_A2) * $signed(notch_y2[0]));
            notch_acc1 =
                ($signed(NOTCH_B0) * $signed(hpf1)) +
                ($signed(NOTCH_B1) * $signed({{8{notch_x1[1][23]}}, notch_x1[1]})) +
                ($signed(NOTCH_B2) * $signed({{8{notch_x2[1][23]}}, notch_x2[1]})) -
                ($signed(NOTCH_A1) * $signed(notch_y1[1])) -
                ($signed(NOTCH_A2) * $signed(notch_y2[1]));
            notch_acc2 =
                ($signed(NOTCH_B0) * $signed(hpf2)) +
                ($signed(NOTCH_B1) * $signed({{8{notch_x1[2][23]}}, notch_x1[2]})) +
                ($signed(NOTCH_B2) * $signed({{8{notch_x2[2][23]}}, notch_x2[2]})) -
                ($signed(NOTCH_A1) * $signed(notch_y1[2])) -
                ($signed(NOTCH_A2) * $signed(notch_y2[2]));

            notch0 = sat24(notch_acc0 >>> 15);
            notch1 = sat24(notch_acc1 >>> 15);
            notch2 = sat24(notch_acc2 >>> 15);

            notch_x2[0] <= notch_x1[0]; notch_x1[0] <= sat24(hpf0); notch_y2[0] <= notch_y1[0]; notch_y1[0] <= {{8{notch0[23]}}, notch0};
            notch_x2[1] <= notch_x1[1]; notch_x1[1] <= sat24(hpf1); notch_y2[1] <= notch_y1[1]; notch_y1[1] <= {{8{notch1[23]}}, notch1};
            notch_x2[2] <= notch_x1[2]; notch_x1[2] <= sat24(hpf2); notch_y2[2] <= notch_y1[2]; notch_y1[2] <= {{8{notch2[23]}}, notch2};

            sample_ch1 <= notch0;
            sample_ch2 <= notch1;
            sample_ch3 <= notch2;

            for (peak_i = 15; peak_i > 0; peak_i = peak_i - 1)
                peak_hist[peak_i] <= peak_hist[peak_i-1];

            local_peak  = max3u24(abs24(notch0), abs24(notch1), abs24(notch2));
            peak_hist[0] <= local_peak;

            if ((artifact_threshold != 24'd0) &&
                ((local_peak > artifact_threshold) || (rolling_max16(1'b0) > artifact_threshold))) begin
                blank_count <= 6'd32;
            end else if (blank_count != 6'd0) begin
                blank_count <= blank_count - 6'd1;
            end

            artifact_active <= (blank_count != 6'd0);
            if (blank_count == 6'd0)
                sample_valid <= 1'b1;
        end
    end
end

endmodule
