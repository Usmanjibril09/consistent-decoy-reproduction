# PROTOCOL v1.2 (frozen before any training; amends v1.1 per D28, no model trained yet)

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
   (HAI; first-order lag rejected, D22); voltage->current->speed chain (QAPPD); sign of steady-state gain.
C3 Coverage-aware fallback: a command is out of coverage when outside its training min-max (D27). Blend weight
   w = min(1, excess / (5% of training range)); output = (1-w)*C1 + w*B3 (identified grey-box loop model).
Runtime budget: per-step inference below the sampling period (10 ms QAPPD, 1 s HAI) on the gateway-class CPU.
Training specification: D28 (spec/decisions.md); free-run evaluation, windows, selection rules, checkpoints.

## 5 Baselines (same splits; same budget per learned method, D28: 12 configurations with seed 0 for selection, best configuration retrained with seeds 1-4; results over seeds 0-4)
B1 Record-and-replay (low-interaction honeypot practice)    B2 Linear ARX / state-space identification
B3 First-order-plus-dead-time grey-box per loop (physics only)
B4 Unconditioned LSTM generator (Sassnick et al., EUROCAST 2024)    B5 Recurrent conditional GAN, RCGAN (Esteban, Hyland, Ratsch 2017)
B6 GAN + autoencoder (NeuralPot, ISCC 2020)    B7 TimeGAN    B8 Command-conditioned Transformer
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
Physics-rule detector (attacker-side, fixed before training, not enforced by C2): QAPPD PitchDot vs 100*dPitch, pitch-yaw coupling;
HAI valve opening vs measured flow, pump command vs TIT03 direction. Only detector applicable to P-syn.

## 8 Metrics (established only)
Primary:
M1 FIT% = 100(1 - ||y - y_hat|| / ||y - mean(y)||) on response tags (system identification; Ljung). P-real, P-anom.
M2 Discriminative score = |accuracy - 0.5| of each K-test classifier, real vs decoy windows (Yoon et al., NeurIPS 2019). 0 = indistinguishable.
M2b SimProcess fidelity score = mean P(real) over decoy windows from the SimProcess classifier (Donadel et al. 2025). Higher = better.
Secondary: physics-rule detector detection rate on P-syn; predictive score TSTR (Yoon et al. 2019); Context-FID (Jeha et al., ICLR 2022);
DTW; held-out normal FIT%; per-step latency; model size.

## 9 Hypotheses (each tested and reported separately; no combined pass condition)
H1 Full method has higher M1 than every baseline on P-real.
H2 Full method has lower M2 than the best baseline, on each plant.
H3 On out-of-coverage probes, removing C3 (A2 vs full) raises M2 and the physics-rule detection rate (P-syn); M1 is tested only on P-real attacks in the D27 out-of-coverage list. If that list is empty, H3 is tested on P-syn only and stated so.

## 10 Statistics
Units: attack (HAI), run x window (QAPPD); 5 training seeds per learned method. Paired differences, cluster bootstrap by attack/run,
10,000 resamples, 95% CI; Wilcoxon signed-rank; Holm across all primary comparisons. Results reported per attack as well as pooled.

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
