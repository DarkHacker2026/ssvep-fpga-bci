`timescale 1ns / 1ps
// =============================================================================
// top_ssvep_tb.v  --  FIXED simulation testbench
//
// Three bugs fixed vs previous version:
//
//   FIX 1 - STATUS monitor spam:
//     Old: fired every clock cycle that led_cal_yes was HIGH → thousands of
//          identical lines printed, appeared to be stuck.
//     New: fires only on RISING EDGE of each LED (using $past).
//
//   FIX 2 - repeat(399) = 53 ms simulation = minutes to run:
//     Old: 400 cycles between samples → 8 windows = 16,384,000 ns.
//     New: SIM_DELAY=9 → 10 cycles/sample → 8 windows = 409,600 ns.
//          The 250 Hz timing is not needed for functional simulation.
//          Keep repeat(399) only in real hardware (top_ssvep with ADS1299).
//
//   FIX 3 - inject_file ran out of samples (patient_data.txt has 1285 samples,
//            8 windows needs 4096):
//     New: inject_file task LOOPS the file automatically when exhausted.
//          N_CAL overridden to 1 in TB (1 window per class = 512 samples).
//          Phase 4 injects 6 windows (loops file ~2.4×) to trigger majority vote.
// =============================================================================

module top_ssvep_tb;

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

    // ── DUT: SIM_BYPASS=1, DEBOUNCE_MAX=10, N_CAL=1 for fast simulation ──
    top_ssvep #(
        .SIM_BYPASS(1),
        .DEBOUNCE_MAX(10),
        .N_CAL(1),          // 1 window per class = 512 samples (fits in file)
        .SHIFT(0)           // log2(1) = 0; mean = acc >> 0 = acc itself
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

    // ── Counters ─────────────────────────────────────────────────────────────
    integer sample_count, window_count;

    // ── AI window monitor ─────────────────────────────────────────────────────
    always @(posedge clk) begin
        if (dut.raw_decision_valid) begin
            window_count = window_count + 1;
            $display("--------------------------------------------------");
            $display("[%0t ns] WINDOW #%0d  (sample #%0d)  calibrated=%0b",
                     $time, window_count, sample_count, calibrated);
            $display("   p15=%0d (%0d%%)  p20=%0d (%0d%%)  p30=%0d (%0d%%)  p40=%0d (%0d%%)",
                     dut.p15_norm, (dut.p15_norm * 100) / 65535,
                     dut.p20_norm, (dut.p20_norm * 100) / 65535,
                     dut.p30_norm, (dut.p30_norm * 100) / 65535,
                     dut.p40_norm, (dut.p40_norm * 100) / 65535);
            if (dut.p15_norm < 16'd100 && dut.p20_norm < 16'd100 &&
                dut.p30_norm < 16'd100 && dut.p40_norm < 16'd100) begin
                $display("   *** WARNING: all scores near zero ***");
                $display("   *** Run prepare_data.py and copy patient_15hz.txt");
                $display("   *** + patient_20hz.txt to the xsim folder ***");
            end
            if      (fault_alert)
                $display("   >>> FAULT: TOO MANY UNCERTAIN WINDOWS <<<");
            else if (decision_valid && decision == 2'b00)
                $display("   >>> STABLE DECISION: [ YES ] <<<");
            else if (decision_valid && decision == 2'b01)
                $display("   >>> STABLE DECISION: [ NO  ] <<<");
            else
                $display("   Waiting for majority...");
            $display("--------------------------------------------------\n");
        end
    end

    // =========================================================================
    // PIPELINE PROBES - prints structured debug lines parseable by dashboard
    // Format: PROBE|stage|signal|value  (one line per event)
    // =========================================================================

    // -- PROBE 1: every injected sample (first 10 only to avoid spam) ---------
    integer probe_sample_cnt;
    initial probe_sample_cnt = 0;
    always @(posedge clk) begin
        if (sim_sample_valid) begin
            if (probe_sample_cnt < 10)
                $display("PROBE|INJECT|sample|%0d", sim_sample_in);
            probe_sample_cnt = probe_sample_cnt + 1;
        end
    end

    // -- PROBE 2: filter bank output each window ------------------------------
    always @(posedge clk) begin
        if (dut.features_valid) begin
            $display("PROBE|FILTER|p15_norm|%0d", dut.p15_norm);
            $display("PROBE|FILTER|p20_norm|%0d", dut.p20_norm);
            $display("PROBE|FILTER|p30_norm|%0d", dut.p30_norm);
            $display("PROBE|FILTER|p40_norm|%0d", dut.p40_norm);
        end
    end

    // -- PROBE 3: divider output each window ----------------------------------
    always @(posedge clk) begin
        if (dut.raw_decision_valid) begin
            $display("PROBE|DIVIDER|p15_norm|%0d", dut.p15_norm);
            $display("PROBE|DIVIDER|p20_norm|%0d", dut.p20_norm);
            $display("PROBE|DIVIDER|p30_norm|%0d", dut.p30_norm);
            $display("PROBE|DIVIDER|p40_norm|%0d", dut.p40_norm);
        end
    end

    // Legacy divider / raw-sample probes removed: the current architecture no
    // longer instantiates that path. Keep only probes that match the active RTL.

    // -- PROBE 5: calibrator state transitions --------------------------------
    always @(posedge clk) begin
        if (dut.cal_inst.cal_wr_en)
            $display("PROBE|CAL|wr_addr|%0d|wr_data|%0d",
                dut.cal_inst.cal_wr_addr,
                $signed(dut.cal_inst.cal_wr_data));
    end

    // ── FIX 1: Status LED monitor - RISING EDGE ONLY (plain Verilog-2001) ────
    // $past() is SystemVerilog only and not supported by xvlog in Verilog mode.
    // Use explicit prev_* registers updated every cycle to detect edges.
    reg prev_cal_yes, prev_cal_no, prev_computing, prev_done;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            prev_cal_yes  <= 0; prev_cal_no  <= 0;
            prev_computing<= 0; prev_done    <= 0;
        end else begin
            prev_cal_yes   <= led_cal_yes;
            prev_cal_no    <= led_cal_no;
            prev_computing <= led_computing;
            prev_done      <= led_done;
        end
    end

    always @(posedge clk) begin
        if ( led_cal_yes   && !prev_cal_yes)
            $display("[%0t ns] STATUS: Collecting YES windows...", $time);
        if (!led_cal_yes   &&  prev_cal_yes)
            $display("[%0t ns] STATUS: YES collection done.", $time);
        if ( led_cal_no    && !prev_cal_no)
            $display("[%0t ns] STATUS: Collecting NO windows...", $time);
        if (!led_cal_no    &&  prev_cal_no)
            $display("[%0t ns] STATUS: NO collection done.", $time);
        if ( led_computing && !prev_computing)
            $display("[%0t ns] STATUS: Computing LDA weights...", $time);
        if ( led_done      && !prev_done)
            $display("[%0t ns] *** CALIBRATION COMPLETE - patient weights loaded ***",
                     $time);
    end

    // ── FIX 2+3: inject_file task - FAST delay, LOOPS file ───────────────────
    // SIM_DELAY: cycles between samples. 9 = 10 cycles/sample for fast sim.
    // (Real hardware uses ADS1299 at 250Hz = 400,000 cycles/sample - not needed here.)
    localparam integer SIM_DELAY = 9;

    task inject_file_looped;
        input [255:0] filename;
        input integer n_windows;   // total windows to inject (loops file if needed)
        integer fptr, status, val, samples_needed, injected;
        begin
            samples_needed = n_windows * 512;
            injected = 0;

            // Loop until we have injected enough samples
            while (injected < samples_needed) begin
                fptr = $fopen(filename, "r");
                if (fptr == 0) begin
                    $display("ERROR: cannot open %s", filename);
                    $finish;
                end
                while (!$feof(fptr) && injected < samples_needed) begin
                    status = $fscanf(fptr, "%d\n", val);
                    if (status == 1) begin
                        sim_sample_in    = val;
                        sim_sample_valid = 1'b1;
                        @(posedge clk);
                        sim_sample_valid = 1'b0;
                        sample_count = sample_count + 1;
                        injected = injected + 1;
                        repeat(SIM_DELAY) @(posedge clk);
                    end
                end
                $fclose(fptr);
            end
        end
    endtask

    // ── Button press task - holds button > DEBOUNCE_MAX=10 cycles ────────────
    task press_button;
        input integer which;   // 0=yes, 1=no
        begin
            if (which == 0) btn_yes = 1'b1;
            else            btn_no  = 1'b1;
            repeat(25) @(posedge clk);   // well above DEBOUNCE_MAX=10
            if (which == 0) btn_yes = 1'b0;
            else            btn_no  = 1'b0;
            repeat(5) @(posedge clk);
        end
    endtask

    // ── Main test sequence ────────────────────────────────────────────────────
    initial begin
        clk=0; rst_n=0; sim_sample_in=0; sim_sample_valid=0;
        btn_yes=0; btn_no=0;
        sample_count=0; window_count=0;

        $display("=======================================================");
        $display(" SSVEP FPGA Simulation - On-chip Calibration Test");
        $display(" N_CAL=1  SIM_DELAY=%0d  Window=512", SIM_DELAY);
        $display("=======================================================\n");

        #100; rst_n = 1; #200;

        // ── Phase 1: BASE WEIGHTS - no calibration yet ───────────────────
        // Test that the trained cross-subject weights already produce
        // YES/NO decisions on 15Hz data.  Majority vote needs 5 windows.
        $display("[PHASE 1] Testing BASE cross-subject weights (no calibration)");
        $display("          Expect STABLE YES for 15Hz data after 5 windows.\n");
        inject_file_looped("patient_15hz.txt", 7);   // 7 windows → majority at 5
        repeat(200) @(posedge clk);

        if (window_count >= 5)
            $display("[PHASE 1] Done. %0d windows processed.\n", window_count);

        // ── Phase 2: YES calibration ─────────────────────────────────────
        $display("[PHASE 2] BTN_YES - collecting 1 YES calibration window");
        press_button(0);
        inject_file_looped("patient_15hz.txt", 1);
        repeat(100) @(posedge clk);
        $display("[PHASE 2] Done.\n");

        // ── Phase 3: NO calibration ──────────────────────────────────────
        $display("[PHASE 3] BTN_NO - collecting 1 NO calibration window");
        press_button(1);
        inject_file_looped("patient_20hz.txt", 1);
        repeat(100) @(posedge clk);
        $display("[PHASE 3] Done.\n");

        // ── Phase 4: Wait for LDA compute ────────────────────────────────
        $display("[PHASE 4] Waiting for LDA (~66 cycles)...");
        repeat(500) @(posedge clk);
        $display("[PHASE 4] calibrated=%0b\n", calibrated);

        // ── Phase 5: Post-calibration inference ──────────────────────────
        $display("[PHASE 5] Post-calibration - injecting 15Hz data");
        $display("          Expect YES decisions (calibrated for this patient).\n");
        inject_file_looped("patient_15hz.txt", 7);
        repeat(200) @(posedge clk);

        $display("\n=======================================================");
        $display(" Done.  Samples=%0d  Windows=%0d  Calibrated=%0b",
                 sample_count, window_count, calibrated);
        $display("=======================================================");
        $finish;
    end

endmodule
