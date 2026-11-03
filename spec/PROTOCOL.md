# PROTOCOL v0.1 (DRAFT - not frozen until approved and tagged protocol-v1)

## 1 Question
Can a lightweight learned industrial decoy remain behaviourally consistent when an attacker actively changes its commands
to test whether it is real?

## 2 Threat model
Attacker has network access to the decoy's industrial endpoint: reads all process tags, writes command tags (valve/pump demands,
motor voltages), observes responses. Goal: decide real vs decoy. Tools: (a) passive fingerprinting classifiers; (b) active probes
(step, pulse, ramp, out-of-range commands); (c) physics checks (e.g. position must lag demand, current must follow voltage).

## 3 Data and splits (no time overlap; held-out sets used once, in Phase 3)
3.1 HAI 23.05 (1 Hz, CC BY-SA 4.0), boiler P1 loops PC, FC, LC, TC, CC.
    Train: hai-train1-3. Validation: hai-train4. Held-out: normal parts of hai-test1/2 (fidelity) and the command-only attacks in
    spec/hai2305_command_attacks.csv (real-plant probe ground truth). Secondary: A213, A234 via haiend PP04-SP-OUT. Cross-time: HAI 22.04 P1.
3.2 QAPPD (Zenodo 10.5281/zenodo.20287835, 100 Hz, CC BY-SA 4.0). First 100 rows (Enable=0) dropped.
    Train: train_1-6. Validation: train_7-8. Held-out: runs 9-10 (unseen runs) + test_1-10 (voltage anomaly in test_7 = out-of-range probe;
    position/delay anomalies = changed operating conditions).
3.3 Attack inclusion rule: an attack is a probe if its command leaves the 10-min pre-attack envelope by >= 5% of the tag's
    training range (1st-99th percentile). Attacks failing the rule are reported but excluded from primary endpoints.

## 4 Method (components; each removed in an ablation)
C1 Command-conditioned response model: lightweight recurrent/temporal-convolutional network predicting responses from command
   history and state (not unconditioned trajectory generation).
C2 Consistency layer: structure from hardware physics, parameters fitted on training data: actuator limits and rate limits,
   demand->position lag (HAI), voltage->current->speed chain (QAPPD), sign of steady-state gain.
C3 Coverage-aware fallback: when a command lies outside training coverage, blend toward an identified grey-box loop model instead of
   extrapolating the network.
Runtime budget: per-step inference below the sampling period (10 ms QAPPD, 1 s HAI) on the gateway-class CPU.

## 5 Baselines (same splits, same tuning budget: 12 configurations x 5 seeds per learned method)
B1 Record-and-replay (low-interaction honeypot practice)    B2 Linear ARX / state-space identification
B3 First-order-plus-dead-time grey-box per loop (physics only)
B4 Unconditioned LSTM generator (Sassnick et al., EUROCAST 2024)    B5 Conditional GAN (Qin et al., Computer Networks 2024)
B6 GAN + autoencoder (NeuralPot, ISCC 2020)    B7 TimeGAN    B8 Command-conditioned Transformer    B9 Command-conditioned GRU, no C2/C3

## 6 Ablations
A1 without C2   A2 without C3   A3 without command conditioning   A4 C2 with limits only (no dynamic relations)

## 7 Evaluation suite
Probe sets: P-real (HAI command attacks, scored against recorded plant response); P-anom (QAPPD test_7 voltage anomaly);
P-syn (generated steps/pulses/ramps/out-of-range, both plants; no ground truth, scored only by classifiers and independent checks).
Classifiers: K-dev (allowed during development) and K-test (never used before Phase 3): MiniRocket, feature-based random forest,
1D-CNN, physics-rule detector.
Independent checks (not enforced by C2, avoids circularity): QAPPD PitchDot vs 100*dPitch; pitch-yaw coupling;
HAI valve opening vs measured flow; pump command vs TIT03 direction.

## 8 Metrics
Primary: (M1) probe-response NRMSE on P-real; (M2) K-test attacker AUC on probe windows (0.5 = indistinguishable);
(M3) independent-check violation rate on P-syn.
Secondary: held-out normal NRMSE, lag/gain error, DTW, per-step latency, model size.

## 9 Hypotheses (each tested and reported separately; no combined pass condition)
H1 Full method has lower M1 than every baseline on P-real.
H2 Full method has lower M2 than the best baseline, on each plant.
H3 Removing C3 increases M1/M3 on out-of-coverage probes (A2 vs full).

## 10 Statistics
Units: attack (HAI), run x window (QAPPD); 5 training seeds per learned method. Paired differences, cluster bootstrap by attack/run,
10,000 resamples, 95% CI; Wilcoxon signed-rank; Holm across all primary comparisons. Results reported per attack as well as pooled.

## 11 Phases (one script each)
01_preprocess.py  02_train.py (method + baselines)  03_evaluate.py  04_ablation.py  05_statistics.py  06_figures.py  manuscript/
Shared code: src/decoy/{data,models,baselines,constraints,attacker,metrics}.py. Every run logs git commit and config hash.

## 12 Open items before freezing
O1 Novelty boundary vs Russo, Marasco and Colajanni (ITASEC 2026, LLM honeypot with physical-constraint validation): read full paper.
O2 Confirm P1 tag names match in HAI 22.04.
O3 Confirm C2 relations per loop from the HAI technical PDF and the Quanser Aero 2 manual.
