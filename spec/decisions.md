# Decisions log (append only)
D1 2026-11-03  HAI = primary dataset (training, held-out test runs, probe tests). HAI 23.05 main; 22.04 cross-time check.
D2 2026-11-03  QAPPD (folder data/raw/EUROCAST, Quanser Aero 2) = second plant for generalisation. Commands: Voltage0/1, Enable. Responses: currents, motor speeds, pitch/yaw.
D3 2026-11-03  DeceptIIoT excluded (network flows only, no process values).
D4 2026-11-03  SWaT/EPIC not needed; QAPPD fills the second-plant role.
D5 2026-11-03  Structure approved: one script per phase (00 protocol ... 07 manuscript).
D6 2026-11-03  QAPPD facts (Zenodo 10.5281/zenodo.20287835, v1.0, CC BY-SA 4.0): 100 Hz; 10 runs x ~182,001 samples (~30 min);
               train_* = normal only; test_* = normal + anomalies; test_label_* = point-wise labels. Enable=1 in 99.9% of samples.
D7 2026-11-03  Known limitation: QAPPD voltages come from the rig's own feedback controller (no reference trajectory recorded),
               so training data shows closed-loop behaviour only. Protocol must measure how well training data covers probe inputs.
D8 2026-11-03  QAPPD anomalies are designed process faults (arXiv 2605.27486): pick-up position outside 3-5 deg, pause outside 0.3-0.7 s,
               motor voltage outside +-18 V (long segment, one file). Only the voltage type resembles a command probe.
D9 2026-11-03  QAPPD trajectories were generated with the tool from Sassnick et al. (EUROCAST 2024 generative OPC UA honeypot), which
               left interaction as future work. That gap is the one this work addresses.
D10 2026-11-03 Sampling rate conflict: Zenodo says 100 Hz, paper says 50 Hz. Resolved from data (PitchDot vs dPitch). See next entry.
D11 2026-11-03 HAI 23.05 confirmed: 1 Hz, no gaps; train 4 files = 896,400 s (~10.4 days) normal; test 2 files = 284,400 s;
               52 attack segments (14 + 38), 56-2052 s each. 6 clean command/response pairs: P1_{FCV01,FCV02,FCV03,LCV01,PCV01,PCV02} D->Z,
               plus 7 controller setpoint outputs (x1001_05 ... x1003_24).
D12 2026-11-03 QAPPD sampling rate = 100 Hz, measured from data (yaw 99.5-102.1 Hz, pitch 97.8-107.1 Hz); matches Zenodo, not the paper's 50 Hz.
               ~30 min per file, cycle ~49 s (~37 cycles/file). Short anomalies 2-17 s (within one cycle); test_7 voltage anomaly 194 s.
D13 2026-11-03 HAI 23.05: 52 attacks (14 + 38). summary_label2 header says 40 but lists 38; we use 38 (matches labels and README).
               Licence stated as CC BY-SA 4.0 (README) and CC BY 4.0 (metadata); follow BY-SA for any derived release.
               Attack targets must come from hai_dataset_technical_details.pdf before HAI attacks are used as probe tests.
D14 2026-11-03 HAI 23.05 attacks classified from technical PDF pp.31-32: 14 command-only (probe ground truth, spec/hai2305_command_attacks.csv),
               8 setpoint-only, 12 with sensor masking (excluded), 8 internal-logic (later), 10 combined (excluded). Decoy scope: P1 loops PC, FC, LC, TC, CC.
D15 2026-11-03 All 14 command-attack onsets match PDF start times. Pre-window z-score invalid when tag constant (PP04); use training-range
               normalisation. Short-term attacks show valve position lagging demand (A105 8.2 vs 2.6, A218 5.8 vs 1.2 sd): real dynamics a decoy must
               reproduce. haiend PP04-SP-OUT available -> A213, A234 secondary setpoint probes. Inclusion rule fixed in PROTOCOL 3.3 before training.
D16 2026-11-03 Metrics restricted to established ones: FIT% (Ljung), discriminative score |acc-0.5| (Yoon et al. 2019), TSTR, Context-FID, DTW.
               Invented "violation rate" removed; physics checks are an attacker-side detector (detection rate). Section 7 renamed fingerprinting attacks.
               Chapter metrics may differ across thesis chapters; link is narrative (Ch2 skepticism consequence, Ch3 cause).
