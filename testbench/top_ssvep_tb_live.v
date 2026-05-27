`timescale 1ns / 1ps
// =============================================================================
// top_ssvep_tb_live.v  --  Live testbench for Emotiv integration
//
// HOW IT WORKS:
//   Python writes EEG samples to  vivado_sim/live_eeg.txt  (one number/line)
//   Python writes button commands to  vivado_sim/cmd.txt   (BTN_YES / BTN_NO)
//   This testbench reads both files and drives the DUT.
//   Decisions are printed as:
//   DECISION|raw_class|stable_valid|stable_class|fault|p15|p20|p30|p40|cal|cal_yes|cal_no
//   Python parses those lines from the simulation log.
//
// FLOW PER WINDOW:
//   1. Python writes 512 samples to live_eeg.txt
//   2. Python writes "RUN" to cmd.txt
//   3. This testbench detects "RUN", injects 512 samples, waits for window output
//   4. Prints a machine-readable DECISION line for the new window
//   5. Python reads decision, updates stimulus display
//   6. Repeat
//
// CALIBRATION:
//   Python writes "BTN_YES" to cmd.txt  → testbench pulses btn_yes
//   Python writes "BTN_NO"  to cmd.txt  → testbench pulses btn_no
// =============================================================================

module top_ssvep_tb_live;

    reg clk, rst_n;
    always #5 clk = ~clk;   // 100 MHz

    reg  signed [23:0] sim_sample_in;
    reg                sim_sample_valid;
    reg                btn_yes, btn_no;

    wire [1:0]  decision;
    wire        decision_valid;
    wire        calibrated;
    wire        LED_ready;
    wire        led_cal_yes, led_cal_no, led_computing, led_done;
    wire        fault_alert;

    // ── DUT ──────────────────────────────────────────────────────────────────
    top_ssvep #(
        .SIM_BYPASS  (1),
        .DEBOUNCE_MAX(10),
        .N_CAL       (16),
        .SHIFT       (3)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .ads_sclk(), .ads_mosi(), .ads_miso(1'b1),
        .ads_cs_n(), .ads_pwdn_n(), .ads_start(),
        .ads_drdy_n(1'b1),
        .sim_sample_in(sim_sample_in),
        .sim_sample_valid(sim_sample_valid),
        .btn_yes(btn_yes), .btn_no(btn_no),
        .decision(decision), .decision_valid(decision_valid),
        .calibrated(calibrated), .LED_ready(LED_ready),
        .led_cal_yes(led_cal_yes), .led_cal_no(led_cal_no),
        .led_computing(led_computing), .led_done(led_done),
        .fault_alert(fault_alert)
    );

    // ── File handles ──────────────────────────────────────────────────────────
    integer eeg_file, cmd_file;
    integer status, val;
    reg [3*8-1:0] cmd_result;   // holds: "RUN", "YES", "NOO", or "NIL"

    // ── Simulation delay between samples (fast sim, not real 250SPS) ─────────
    localparam SIM_DELAY = 9;   // 10 cycles per sample

    // ── Inject one sample ─────────────────────────────────────────────────────
    task inject_sample;
        input integer s;
        begin
            sim_sample_in    = s;
            sim_sample_valid = 1'b1;
            @(posedge clk);
            sim_sample_valid = 1'b0;
            repeat(SIM_DELAY) @(posedge clk);
        end
    endtask

    // ── Pulse a button (> DEBOUNCE_MAX=10 cycles) ─────────────────────────────
    task pulse_btn_yes;
        begin
            btn_yes = 1'b1;
            repeat(25) @(posedge clk);
            btn_yes = 1'b0;
            repeat(5)  @(posedge clk);
            $display("STATUS|BTN_YES|pressed");
        end
    endtask

    task pulse_btn_no;
        begin
            btn_no = 1'b1;
            repeat(25) @(posedge clk);
            btn_no = 1'b0;
            repeat(5)  @(posedge clk);
            $display("STATUS|BTN_NO|pressed");
        end
    endtask

    // ── Decision monitor ──────────────────────────────────────────────────────
    reg prev_raw_valid;
    always @(posedge clk) begin
        prev_raw_valid <= dut.raw_decision_valid;
        if (dut.raw_decision_valid && !prev_raw_valid) begin
            // Print machine-readable decision line
            $strobe("DECISION|%0d|%0d|%0d|%0d|%0d|%0d|%0d|%0d|%0d|%0d|%0d",
                dut.raw_decision,
                decision_valid,
                decision,
                fault_alert,
                dut.p15_norm,
                dut.p20_norm,
                dut.p30_norm,
                dut.p40_norm,
                calibrated,
                led_cal_yes,
                led_cal_no
            );
        end
    end

    // ── LED state changes ─────────────────────────────────────────────────────
    reg prev_cal_yes, prev_cal_no, prev_computing, prev_done, prev_calibrated, prev_fault;
    always @(posedge clk) begin
        prev_cal_yes   <= led_cal_yes;
        prev_cal_no    <= led_cal_no;
        prev_computing <= led_computing;
        prev_done      <= led_done;
        prev_calibrated<= calibrated;
        prev_fault     <= fault_alert;

        if ( led_cal_yes  && !prev_cal_yes)    $display("STATUS|LED|CAL_YES_ON");
        if (!led_cal_yes  &&  prev_cal_yes)    $display("STATUS|LED|CAL_YES_OFF");
        if ( led_cal_no   && !prev_cal_no)     $display("STATUS|LED|CAL_NO_ON");
        if (!led_cal_no   &&  prev_cal_no)     $display("STATUS|LED|CAL_NO_OFF");
        if ( led_computing&& !prev_computing)  $display("STATUS|LED|COMPUTING");
        if ( led_done     && !prev_done)       $display("STATUS|LED|DONE");
        if ( calibrated   && !prev_calibrated) $display("STATUS|CALIBRATED");
        if ( fault_alert  && !prev_fault)      $display("STATUS|FAULT|TOO_MANY_UNCERTAIN");
    end

    // ── Main loop ─────────────────────────────────────────────────────────────
    integer  window_num;
    integer  samples_read;

    // File-based string comparison helper
    // Reads cmd.txt, returns what command is in it, then clears it
    task read_cmd;
        output reg [3*8-1:0] result;  // "RUN", "YES", "NOO", "NIL"
        integer f, c;
        integer idx;
        reg [8*8-1:0] cmd_buf;   // renamed from buf - buf is a reserved Verilog keyword
        begin
            result = "NIL";
            f = $fopen("D:/vivado_proj/ssvep_fpga/vivado_sim/cmd.txt", "r");
            if (f != 0) begin
                cmd_buf = 0;
                for (idx = 0; idx < 8; idx = idx+1) begin
                    c = $fgetc(f);
                    if (c != -1 && c != "\n" && c != "\r")
                        cmd_buf[idx*8 +: 8] = c[7:0];
                end
                $fclose(f);
                // Check first chars to identify command
                if (cmd_buf[0+:8]=="R" && cmd_buf[8+:8]=="U" && cmd_buf[16+:8]=="N")
                    result = "RUN";
                else if (cmd_buf[0+:8]=="B" && cmd_buf[8+:8]=="T" && cmd_buf[16+:8]=="N" &&
                         cmd_buf[24+:8]=="_" && cmd_buf[32+:8]=="Y")
                    result = "YES";
                else if (cmd_buf[0+:8]=="B" && cmd_buf[8+:8]=="T" && cmd_buf[16+:8]=="N" &&
                         cmd_buf[24+:8]=="_" && cmd_buf[32+:8]=="N")
                    result = "NOO";
            end
        end
    endtask

    initial begin
        clk = 0; rst_n = 0;
        sim_sample_in = 0; sim_sample_valid = 0;
        btn_yes = 0; btn_no = 0;
        window_num = 0;
        prev_raw_valid = 0;
        prev_cal_yes=0; prev_cal_no=0; prev_computing=0;
        prev_done=0; prev_calibrated=0; prev_fault=0;

        #100; rst_n = 1; #200;

        $display("READY");   // Python waits for this line before sending data

        // ── Main polling loop ──────────────────────────────────────────────
        forever begin
            // Poll cmd.txt every 100 clock cycles
            repeat(100) @(posedge clk);

            read_cmd(cmd_result);

            if (cmd_result == "YES") begin
                // Clear cmd file
                cmd_file = $fopen("D:/vivado_proj/ssvep_fpga/vivado_sim/cmd.txt", "w"); $fclose(cmd_file);
                pulse_btn_yes;

            end else if (cmd_result == "NOO") begin
                cmd_file = $fopen("D:/vivado_proj/ssvep_fpga/vivado_sim/cmd.txt", "w"); $fclose(cmd_file);
                pulse_btn_no;

            end else if (cmd_result == "RUN") begin
                // Clear cmd file
                cmd_file = $fopen("D:/vivado_proj/ssvep_fpga/vivado_sim/cmd.txt", "w"); $fclose(cmd_file);

                // Open the EEG sample file
                eeg_file = $fopen("D:/vivado_proj/ssvep_fpga/vivado_sim/live_eeg.txt", "r");
                if (eeg_file == 0) begin
                    $display("ERROR|Cannot open vivado_sim/live_eeg.txt");
                end else begin
                    samples_read = 0;
                    while (!$feof(eeg_file) && samples_read < 512) begin
                        status = $fscanf(eeg_file, "%d\n", val);
                        if (status == 1) begin
                            inject_sample(val);
                            samples_read = samples_read + 1;
                        end
                    end
                    $fclose(eeg_file);
                    window_num = window_num + 1;
                    $display("WINDOW_DONE|%0d|%0d", window_num, samples_read);
                    // Give pipeline time to finish (MAC + gate cycles)
                    repeat(200) @(posedge clk);
                end
            end
            // else: cmd is NIL, nothing to do, keep polling
        end
    end

endmodule
