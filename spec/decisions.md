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
D17 2026-11-03 Aligned with TimeGAN (discriminative score uses its post-hoc RNN classifier; TSTR), SimProcess (CPSS 2025: tsfresh+RF attacker,
               fidelity score M2b; models noise only and filters out windows with process changes = our gap), NeuPot (IEEE TII 2023, seq2seq+Huber
               Modbus honeypot = closest method competitor, baseline B10, full text needed: O5).
D18 2026-11-03 D17 committed before PROTOCOL patch applied (anchor mismatch); patch applied in this commit. Borrowed papers supply the evaluation attacker and baselines only; the decoy method is our own.
D19 2026-11-03 O1 closed. Russo, Marasco, Colajanni (ITASEC 2026, CEUR Vol-4198 paper23): LLM values + 4 rule checks, no physical process model;
               own limitation: "actuator-sensor feedback loops ... multi-minute transients are not explicitly captured"; future work = process dynamics.
               Supports our gap. Threat-model citation: Vetterl & Clayton, WOOT 2018 (actuator/sensor inconsistency fingerprinting). New item O6: LLMPot.
D20 2026-11-03 O6 closed: LLMPot (Vasilatos et al., arXiv 2405.05999v3) requires live-PLC probing; related work only, not a baseline.
D21 2026-11-03 O2 closed (HAI 22.04 tag names match). O3: C2 relation forms from hardware type (DC motor V~I,w; first-order lags; rate/range limits), parameters fitted on training split only; forms checked on train_1 / hai-train1.
D22 2026-11-03 O3/O4/O5 closed. QAPPD: V = R*I + k*w holds (R2 0.972/0.983); speed tau ~240 ms both motors (1-step R2=1.0 uninformative at 100 Hz).
               HAI: first-order lag rejected (R2 0.005-0.41); position tracks demand (corr 0.976-0.999) with rate limit p99.9
               (FCV01 2.96, FCV03 1.08, PCV01 1.15, LCV01 1.15 per s); PCV02 near-static, excluded. FCV03 training range 49.0-66.9 (coverage gap).
               Qin et al. = topology-image CGAN -> related work; B5 = RCGAN. NeuPot B10 from published description.
D23 2026-11-03 Phase 1 rules fixed before running 01_preprocess.py. Processed data kept in raw physical units; normalisation stats from training
               split only, applied from Phase 2. QAPPD: trim leading/trailing Enable=0 rows, stop if Enable=0 inside a run; test normal mask =
               label 0 and not within one cycle after an anomaly (cycle from training Pitch autocorrelation). HAI: scope = all P1_ tags; roles by rule:
               command = 5 attacked CVs; exogenous = other P1 tags ending in D or containing SP; response = remaining P1 tags; constant-in-training
               tags marked constant. HAI test normal mask excludes attacks + 600 s after each. Non-finite values stop the run (no imputation).
               Correction: earlier "QAPPD voltage anomaly outside +-18 V in test_7" is not supported (test_7 |V| <= 15.9; test_3 reaches 18.66).
               P-anom definition to be fixed from qappd_anomaly_segments.csv (command channels only) before Phase 2.
D24 2026-11-03 Phase 1 data facts. (1) HAI 23.05 label-test2.csv (and haiend copy) has timestamps truncated to minutes; rows equal (230,400), every row matches data time floored to minute, 38 segments (14+38=52 total, per PDF), all 10 test2 command attacks start on the PDF second with matching durations -> labels aligned by position; 01_preprocess accepts "minute" alignment only under these conditions, logged per file. (2) QAPPD test_10 raw file is 181,991 rows (label file same); not a trimming error. (3) QAPPD cycle period varies by run (37.7-49.0 s, median 41.5 s); described as varying excitation, median used only as split margin. (4) QAPPD anomaly segments all lie inside training voltage range; outside-range rows total <=136 (<=1.36 s, test_3/8/9/10), all labelled normal; earlier "18.66 V in test_7/anomaly" claims corrected (test_3 Voltage1 max 18.66 V occurs in normal data). H3 out-of-coverage probes therefore use HAI and P-syn only, not QAPPD (amplitude only; rate/frequency coverage unchecked).
