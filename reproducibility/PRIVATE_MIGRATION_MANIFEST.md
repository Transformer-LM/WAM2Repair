# Private migration manifest

Git provides the portable source tree. The following artifacts are deliberately excluded from Git and must be copied privately from the previous machine, or acquired again from their licensed upstream sources. Do not publish them in this repository.

## Required to run the frozen WAM and pi0.5 interfaces

```text
${OPENPI_ROOT}/                         # compatible OpenPI source/runtime
${FASTWAM_ROOT}/                        # compatible FastWAM source/runtime
${PI05_CHECKPOINT}/                     # complete pi0.5 checkpoint directory, including norm stats
${FASTWAM_ACTION_DIT_CHECKPOINT}        # ActionDiT checkpoint
${FASTWAM_ROOT}/runs/policy_relevant_imagined_state_repair_5000/checkpoints/weights/step_005000_trainable_only.pt
${WAM2REPAIR_ROOT}/results/israc/P027C_LIBERO_COMBINED_NORM_STATS.json
```

The pi0.5 checkpoint is a sharded directory, not a single file. Preserve its complete manifest, metadata, parameter shards, and LIBERO normalization statistics.

## Required to reproduce the clean V8 held-out diagnostic

Copy these complete directories together; manifests and SHA256 fields inside them bind the provenance.

```text
${WAM2REPAIR_ROOT}/results/wam2repair/d0v8_e30_cleantrain_20260918T131200Z_2400001/
${WAM2REPAIR_ROOT}/results/wam2repair/d0v8_e34_cleantrain_v2_20260918T131600Z_2401000/
${WAM2REPAIR_ROOT}/results/wam2repair/d0v8_e36_cleantrain_20260918T132000Z_2402000/
${WAM2REPAIR_ROOT}/results/wam2repair/d0v8_e11_cleanval_20260918T132400Z_2403000/
${WAM2REPAIR_ROOT}/results/wam2repair/d0v8_e10_cleanheldout_20260918T134700Z_2411000/

${WAM2REPAIR_ROOT}/results/wam2repair/pair_d0v8_e30_cleantrain_20260918T132700Z_2404000/
${WAM2REPAIR_ROOT}/results/wam2repair/pair_d0v8_e34_cleantrain_20260918T132700Z_2404001/
${WAM2REPAIR_ROOT}/results/wam2repair/pair_d0v8_e36_cleantrain_20260918T132700Z_2404002/
${WAM2REPAIR_ROOT}/results/wam2repair/pair_d0v8_e11_cleanval_20260918T132700Z_2404003/
${WAM2REPAIR_ROOT}/results/wam2repair/pair_d0v8_e10_cleanheldout_20260918T135100Z_2412000/

${WAM2REPAIR_ROOT}/results/wam2repair/wam_d0v8_e30_cleantrain_20260918T132900Z_2405000/
${WAM2REPAIR_ROOT}/results/wam2repair/wam_d0v8_e34_cleantrain_20260918T133100Z_2406000/
${WAM2REPAIR_ROOT}/results/wam2repair/wam_d0v8_e36_cleantrain_20260918T133500Z_2407000/
${WAM2REPAIR_ROOT}/results/wam2repair/wam_d0v8_e11_cleanval_20260918T133900Z_2408000/
${WAM2REPAIR_ROOT}/results/wam2repair/wam_d0v8_e10_cleanheldout_20260918T135300Z_2413000/

${WAM2REPAIR_ROOT}/results/wam2repair/geometry_d0v8_e30_cleantrain_20260918T134200Z_2409000/
${WAM2REPAIR_ROOT}/results/wam2repair/geometry_d0v8_e34_cleantrain_20260918T134200Z_2409001/
${WAM2REPAIR_ROOT}/results/wam2repair/geometry_d0v8_e36_cleantrain_20260918T134200Z_2409002/
${WAM2REPAIR_ROOT}/results/wam2repair/geometry_d0v8_e11_cleanval_20260918T134200Z_2409003/
${WAM2REPAIR_ROOT}/results/wam2repair/geometry_d0v8_e10_cleanheldout_20260918T135700Z_2414000/

${WAM2REPAIR_ROOT}/results/wam2repair/unified4d_v8distance_clean_selection_train30_34_36_val11_20260918T134500Z_2410000/
${WAM2REPAIR_ROOT}/results/wam2repair/unified4d_v8distance_clean_eval_train30_34_36_val11_test10_20260918T135800Z_2415000/
```

## Optional historical archive

The remaining WAM2Repair result tree contains exploratory, failed, superseded, synthetic, and diagnostic runs. Preserve it as an archive if storage permits, but do not use it as a fresh training pool or report it as final evidence without re-auditing the split, target access, and selection provenance.
