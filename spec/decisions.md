# Decisions log (append only)
D1 2026-11-03  HAI = primary dataset (training, held-out test runs, probe tests). HAI 23.05 main; 22.04 cross-time check.
D2 2026-11-03  QAPPD (folder data/raw/EUROCAST, Quanser Aero 2) = second plant for generalisation. Commands: Voltage0/1, Enable. Responses: currents, motor speeds, pitch/yaw.
D3 2026-11-03  DeceptIIoT excluded (network flows only, no process values).
D4 2026-11-03  SWaT/EPIC not needed; QAPPD fills the second-plant role.
D5 2026-11-03  Structure approved: one script per phase (00 protocol ... 07 manuscript).
