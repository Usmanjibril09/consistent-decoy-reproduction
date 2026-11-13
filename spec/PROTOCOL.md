# PROTOCOL v1.10 (D33 Phase 3 spec, amended by D36, D43, D45; D45 made before any decoy output is scored; v1.8 = D73, editorial, made after all results were recorded: section 7 and 12 aligned with D34/D35, no rule, setting or result changed) | v1.9 = D74, editorial: section 4 annotated | v1.10 = D75, editorial: section 4 wording replaced by the implemented rule (D32), header version label corrected; no rule, setting or result changed

## 1 Question
Can a lightweight learned industrial decoy remain behaviourally consistent when an attacker actively changes its commands
to test whether it is real?

## 2 Threat model
Attacker has network access to the decoy's industrial endpoint: reads all process tags, writes command tags (valve/pump demands,
motor voltages), observes responses. Goal: decide real vs decoy. Tools: (a) passive fingerprinting classifiers; (b) active probes
(step, pulse, ramp, out-of-range commands); (c) physics checks (e.g. position must lag demand, current must follow voltage).

## 3 Data and splits (no time overlap; held-out sets used once, in Phase 3)
3.1 HAI 23.05 (1 Hz, CC BY-SA 4.0), boiler process P1. Command tags P1_PCV01D, P1_FCV01D, P1_FCV03D, P1_LCV01D, P1_PP04;
    5 command, 3 exogenous, 18 response, 10 constant tags (constant not modelled; P1_PP04D excluded, D25).
    Train: hai-train1-3. Validation: hai-train4. Held-out: normal parts of hai-test1/2 (fidelity) and the 10 command-only
    attacks passing 3.3 (spec/hai2305_command_attacks.csv; real-plant probe ground truth). Cross-time: HAI 22.04 P1, same tag set.
    Files are time-adjacent (1 s gaps): the first and last 600 s of hai-train4, hai-test1 and hai-test2 are excluded from model
    selection and evaluation, on top of the Phase 1 normal mask (attacks + 600 s after).
    Setpoint attacks (e.g. A213, A234) are not used: they act upstream of the controller; the decoy is driven by command tags.
3.2 QAPPD (Zenodo 10.5281/zenodo.20287835, 100 Hz, CC BY-SA 4.0). First 100 rows (Enable=0) dropped. Commands Voltage0/1, 11 responses.
    Train: train_1-6. Validation: train_7-8. Held-out: train_9-10 (unseen normal runs) and test_1-10 (point labels).
    All 33 labelled anomaly segments lie within the training voltage range (D24): test_7 (194.3 s voltage anomaly, peaks 15.32/15.91 V)
    is a within-range command probe; position/delay anomalies are changed operating conditions. QAPPD has no out-of-coverage probe.
3.3 Attack inclusion rule: an attack is a probe if its command leaves the 10-min pre-attack envelope by >= 5% of the tag's
    training range (1st-99th percentile). Attacks failing the rule are reported but excluded from primary endpoints.

## 4 Method (components; each removed in an ablation)
C1 Command-conditioned response model: GRU predicting response tags from command, exogenous and response history
   and state (not unconditioned trajectory generation). Width/depth chosen within the 12-configuration budget.
C2 Consistency layer: structure from hardware physics, parameters fitted on training data: physical actuator range where known
   (valve opening 0-100 %), training range only for tags without a physical limit; p99.9 rate limits; demand->position tracking
   (HAI; first-order lag rejected, D22); motor-current band: predicted current kept unless outside (V - k*w - c)/R +- p99.9 of training |I - (V - k*w - c)/R| (QAPPD, D32).
C3 Coverage-aware fallback: a command is out of coverage when outside its training min-max (D27). Blend weight
   w = min(1, excess / (5% of training range)); output = (1-w)*C1 + w*B3 (identified grey-box loop model).
Runtime budget: per-step inference below the sampling period (10 ms QAPPD, 1 s HAI) on the gateway-class CPU.
Training specification: D28 (spec/decisions.md); free-run evaluation, windows, selection rules, checkpoints.

## 5 Baselines (same splits; same budget per learned method, D28: 12 configurations with seed 0 for selection, best configuration retrained with seeds 1-4; results over seeds 0-4)
B1 Record-and-replay (low-interaction honeypot practice)    B2 Linear ARX / state-space identification
B3 First-order-plus-dead-time grey-box per loop (physics only)
B4 Unconditioned LSTM generator (Sassnick et al., EUROCAST 2024)    B5 Recurrent conditional GAN, RCGAN (Esteban, Hyland, Ratsch 2017)
B6 GAN + autoencoder adaptation (inspired by NeuralPot, ISCC 2020; not a reproduction, D30)    B7 TimeGAN    B8 Command-conditioned Transformer
B9 Seq2seq forecaster with Huber loss (NeuPot, IEEE TII 2023; re-implemented from published description, full text unavailable)

## 6 Ablations
A1 without C2   A2 without C3   A3 without command conditioning   A4 C2 with limits only (no dynamic relations)   A5 C1 only (no C2, no C3; former baseline B9)

## 7 Fingerprinting attacks (evaluation attacker; the decoy is the target, not the detector)
Probe sets: P-real (10 HAI command attacks, real plant response recorded); P-anom (QAPPD test_7, within-range voltage anomaly,
real response recorded); P-syn (synthetic steps, pulses, ramps and out-of-range commands, both plants; no real response).
Out-of-coverage probe = any command sample outside the tag's training min-max. P-syn out-of-range set by construction;
P-real members fixed in D27 before training; P-anom none.
Fingerprinting classifiers trained to separate real from decoy windows: K-dev (usable during development) and K-test
(never used before Phase 3): TimeGAN post-hoc 2-layer RNN classifier (Yoon et al. 2019); SimProcess pipeline, tsfresh features + random forest (Donadel et al., CPSS 2025); MiniRocket. K-dev: 1D-CNN.
Physics-rule detector (attacker-side, fixed before training, not enforced by C2; rules as kept in D35): QAPPD PitchDot and YawDot vs
centred rate of change of Pitch and Yaw (k = 10 samples); HAI measured flow P1_FT01Z, P1_FT02Z, P1_FT03 vs the six valve positions at a
common lag. Pitch-yaw coupling and pump command vs TIT03 were candidates dropped by the D35 criterion (validation R2 < 0.25; reported
as limitations). Only detector applicable to P-syn.

## 8 Metrics (established only)
Primary:
M1 FIT% = 100(1 - ||y - y_hat|| / ||y - mean(y)||) on response tags (system identification; Ljung). P-real, P-anom.
M2 Discriminative score = |accuracy - 0.5| of each K-test classifier, real vs decoy windows (Yoon et al., NeurIPS 2019). 0 = indistinguishable. Held-out normal windows only (needs real windows, D33). SimProcess component scored on a fixed, seeded window subsample (D45); other classifiers on all windows.
M2b SimProcess fidelity score = mean P(real) over decoy windows from the SimProcess classifier (Donadel et al. 2025). Higher = better. HAI normal windows only (SimProcess discards windows with large variations, D33); not applicable on QAPPD, where its window filter passes no training window (D43). Scored on the fixed D45 subsample (160 base windows per test half); filter pass rate reported beside it.
Secondary: physics-rule detector detection rate on P-syn; predictive score TSTR (Yoon et al. 2019); Context-FID (Jeha et al., ICLR 2022);
DTW; held-out normal FIT%; per-step latency; model size.

## 9 Hypotheses (each tested and reported separately; no combined pass condition)
H1 Full method has higher M1 than every baseline on P-real. M1 on P-real = FIT of the attack's response tag listed in
   spec/hai2305_command_attacks.csv (fixed D14, before training); all-tag FIT reported as secondary (D36).
H2 Full method has lower M2 than the best baseline, on each plant. Best baseline = lowest mean held-out M2 among B1-B9
   (choosing the strongest opponent after the fact can only work against H2).
H3 On out-of-range P-syn probes, removing C3 (A2 vs Full) raises the physics-rule detection rate, on each plant. M1 of A2 vs Full
   on the four P-real out-of-coverage attacks (A103, A105, A110, A205) is reported per attack without a test (n = 4: smallest
   one-sided signed-rank p = 1/16). Changed from v1.3 (D33): M2 needs real windows, which P-syn does not have.
H4 (added in D33 after the Phase 2 validation review, before any held-out data was opened) On out-of-range P-syn probes, adding
   C2/C3 lowers the physics-rule detection rate for each backbone: Full vs A5 (GRU) and B8-L vs B8 (Transformer), on each plant.

## 10 Statistics (D33)
Primary family, Holm at 0.05, m = 17:
  H1: 9 comparisons (Full vs B1-B9). Unit = attack (n = 10); per-attack M1 averaged over the 5 training seeds (deterministic
      methods: one value). One-sided Wilcoxon signed-rank. Smallest possible p = 1/1024, so H1 passes only with near-uniform wins.
  H2: 2 comparisons (one per plant). Statistic = paired difference in M2, averaged over the three K-test classifiers and all seeds;
      one-sided p from a moving-block bootstrap over the test half (blocks 1 h HAI, 60 s QAPPD; 10,000 resamples).
  H3: 2 comparisons; H4: 4 comparisons. Unit = P-syn base window (n = 20 per plant); detection rate per base window averaged over
      its out-of-range probes and the 5 training seeds. One-sided Wilcoxon signed-rank (zero differences dropped; p = 1 if all zero).
Every comparison also reports the median paired difference, matched-pairs rank-biserial correlation and a 95% bootstrap CI
(10,000 resamples, seed 2026). Per-attack and per-tag tables for all methods. Secondary metrics are descriptive, without tests.
Exact reproducibility of the analysis is not evidence of error control; only the tests above carry inferential claims.

## 11 Phases (one script each)
01_preprocess.py  02_train.py (method + baselines)  03_evaluate.py  04_ablation.py  05_statistics.py  06_figures.py  manuscript/
Shared code: src/decoy/{data,models,baselines,constraints,attacker,metrics}.py. Every run logs git commit and config hash.

## 12 Open items before freezing
O1 CLOSED (D19): Russo et al. (ITASEC 2026) has no process-dynamics model; writes only update replayed state; evaluated by rule coherence and latency, no fingerprinting classifiers or active probes. Cited as gap source.
O2 CLOSED (D21, corrected D25): five-loop P1 tags present in HAI 22.04; P1_PP04D absent from all 22.04 files, excluded from the shared tag set.
O4 CLOSED (D22): Qin et al. CGAN generates network-topology images (condition = compromised-node count); related work; B5 -> RCGAN.
O5 CLOSED (D22): no access; B9 built from published description and stated as such in the paper.
O6 CLOSED (D20): LLMPot fine-tunes an LLM on request/response packets collected by probing a live PLC; byte-level, own metrics (BCA, RVA), no fingerprinting classifiers. Not applicable to recorded datasets -> Related Work, not a baseline.
O3 CLOSED (D22): QAPPD V~I,w (R2 0.97/0.98), speed lag tau~240 ms (judge multi-step, not 1-step R2); HAI tracking + rate limit; PCV02 excluded.
O8 CLOSED (D34): Quanser Aero 2 motor voltage range +-24 V (Quanser QUARC documentation); QAPPD out-of-range probes clipped to +-24 V.

## 13 Phase 3 evaluation (D33; fixed before any held-out file is opened)
E1 Held-out files are opened only by scripts/03_evaluate.py (splits test, heldout, crosstime); every file is hashed in the run log.
   Normalisation = training mu/sd, checked against layers/parameters.json. Models frozen at phase2-done; nothing tuned on held-out data.
   Runs in .venv-eval (same torch/numpy as .venv-train, plus tsfresh/sktime) after reproducing Phase 2 validation FIT within 0.01.
E2 Decoy window = recorded command and exogenous traces plus the method's responses: W real samples, then H free-run steps
   (HAI 60 + 300 s, QAPPD 1 + 5 s, as in training). Events longer than H are tiled into consecutive windows, each re-warmed from real
   data; re-synchronisation every H is reported as a limitation. B4-B7 contribute their response channels only.
E3 P-real: the 10 attacks of 3.3; first window starts W before onset, tiled to cover the attack (A207: 3 windows); no window may
   contain another attack. M1 per attack = response-tag FIT over its windows scored as one
   set, mean over training seeds (D36). Held-out normal, cross-time and P-anom FIT: all response tags, pooled over windows.
E4 P-anom: QAPPD test_7 anomaly segment, tiled as in E3. Held-out normal FIT: normal windows of hai-test1/2 (600 s file edges
   excluded), QAPPD train_9/10, HAI 22.04 (cross-time); training stride.
E5 P-syn: 20 base windows per plant drawn with seed 33 from held-out normal windows; probes start at the first free-run step.
   Per command tag, both directions: step, pulse (HAI 30 s, QAPPD 0.5 s) and ramp over the horizon, amplitude 25% of the training
   range (in-range set; probes leaving training min-max are dropped from it); steps to 10% and 25% of the training range beyond the
   training max and min (out-of-range set), clipped to physical limits where known (valves 0-100 %; QAPPD voltage +-24 V, D34). Probes
   whose clipped target falls inside training min-max are dropped and counted.
E6 Physics-rule detector (section 7): threshold per rule = p99.9 of its residual on real training windows; a window is detected if
   any free-run step exceeds a threshold. The rate on real held-out normal windows is reported as the false-alarm rate.
E7 K-test: one classifier per method, training seed and classifier type, trained on held-out normal windows, real vs decoy (same
   commands). Each held-out file split in time: first half train, second half test, purge gap of one window. 5 classifier seeds per
   training seed (B2/B3 deterministic: one model; B1: replay seeds 0-4). Post-hoc 2-layer RNN (Yoon et al.); MiniRocket +
   RidgeClassifierCV; SimProcess as published on HAI (Kalman noise, window 20, filter eps 0.3, tsfresh, random forest); QAPPD window length = H without the window filter (it passes 0 training windows; D43). On P-syn the
   trained K-test classifiers score decoy probe windows (flag rate, secondary).
E8 Secondary: TSTR (post-hoc 2-layer next-step predictor trained on the decoy train half, MAE on the real test half); Context-FID
   with the official TS2Vec encoder trained on real training windows only, fixed before Phase 3; DTW; per-step latency on the Mac CPU
   (median of 1,000 steps, batch 1); parameter count.
E9 Second configuration B8-L = selected B8 weights + C2/C3 through constraints.apply, no retraining. Reported beside Full;
   not a baseline and not part of H1/H2.
