# Hardware-Accelerated SSVEP Brain-Computer Interface
**Hackathon Final Submission**

Welcome to our project submission! This directory contains the complete end-to-end framework for a fully hardware-accelerated, self-contained BCI capable of on-chip patient calibration, runtime bias-adaptation, clinical-grade biological noise filtering, and algorithmic classification entirely inside Verilog.

Please refer to the enclosed **`Final_Technical_Report.md`** for extremely detailed algorithmic and mathematical breakdowns of the logic, as well as the overarching clinical goals.

## Directory Structure

### `1_RTL_Source\`
This folder contains all heavily optimized, hand-written RTL logic blocks running precisely under a 100 MHz clock rate.
*   **`ads1299_interface.v`:** The primary analog frontend bridge managing SPI traffic. It natively calculates instantaneous Common Average Referencing (CAR), a 50Hz notch filter, and dynamic hardware artifact blanking.
*   **`matched_filter_bank.v`:** A time-multiplexed DSP48 pipeline engine parsing exactly 12 harmonic SSVEP stimuli signatures and yielding vector magnitude utilizing internal iterative Inverse Square Root (`isqrt`) logic.
*   **`neural_network.v`:** A highly constrained 716-parameter Multi-Layer Perceptron formatted across `18-bit (Q16)` depth processing forward propagation via arithmetic bit-shifted Leaky ReLU and Hardware Tanh Padé Approximants.
*   **`on_chip_calibrator.v`:** An active feedback calibrator utilizing unsupervised online pseudo-label learning to battle physiological fatigue continuously via bias correction.
*   **`top_ssvep.v`:** The macro-integrator operating robust Exponential Moving Average (EMA) smoothing and fault-alert protocols over neural logits before rendering terminal outputs. 
*   **`.vh` Header Files:** Instantiated arrays containing statically compiled coefficient logic mapping directly into Block RAM logic.

### `2_Testbench\`
Includes testing platforms bridging simulated temporal environments toward real-data implementations.
*   **`top_ssvep_tb_live.v`:** Our premier testing utility. Instead of standard analytical mathematical evaluations, this testbench streams *real, clinical offline EEG Data* (represented via `demo_eeg.txt`) exactly mimicking biological artifacts and variable physical impedances into the SPI framework to evaluate EMA Fault Tolerance accurately. 
*   **`.txt` Files:** Authentic representations of human occipital electrical responses formatting the basis of the Live Simulation.

### `3_Python_Toolchain\`
Enclosed are parallel mathematical veritas tools ensuring software parity with fixed-point Verilog calculations.
*   **`bootstrap_network.py`:** Initiates the neural structure and actively resolves Harmonic De-Duplication constraints utilizing intelligent topological shuffling. Generates Verilog `.vh` ROM maps natively.
*   **`verify_pipeline_accuracy.py`:** Validated execution matrices processing the complete 40 GB Human SSVEP Dataset through isolated `Q16` software approximations. Proved offline trial independence metrics equating to **95.83%** functionality avoiding overfitting.

### `4_Synthesis_Reports\`
Post-synthesis metrics explicitly derived utilizing AMD Vivado targeting the standard `Zynq-7020` deployment FPGA.
*   Highlights incredible pipeline throughput bounding hardware implementations heavily inside 98% boundary limits while dynamically running successfully at roughly 0.6W overhead configurations. 

---
**Thank you for your time reviewing our work—this project aspires to lay foundational constraints for entirely portable, long-term operational BCI platforms directly restoring locked-in patients' independence.**
