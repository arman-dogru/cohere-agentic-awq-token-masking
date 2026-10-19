Held-out tools: 150 teacher-forced episodes over 10 tools never seen in calibration; free generation on 80 held-out and 80 in-distribution prompts (greedy, first call after a user request). 95% cluster bootstrap over episodes/prompts, mean over calibration seeds.

| condition | seeds | held-out call exact (TF) | held-out call NLL/tok | held-out answer NLL/tok | gen correct, held-out | gen correct, in-dist |
|---|---|---|---|---|---|---|
| fp | 1 | 0.656 [0.602, 0.709] | 0.062 [0.051, 0.073] | 1.806 [1.704, 1.920] | 0.637 [0.537, 0.738] | 0.487 [0.375, 0.600] |
| rtn | 1 | 0.459 [0.403, 0.515] | 0.088 [0.077, 0.099] | 2.020 [1.912, 2.139] | 0.438 [0.325, 0.550] | 0.463 [0.350, 0.575] |
| awq_short | 3 | 0.557 [0.503, 0.609] | 0.070 [0.060, 0.080] | 2.110 [1.999, 2.234] | 0.575 [0.475, 0.675] | 0.446 [0.354, 0.546] |
| awq_long_nomask | 3 | 0.602 [0.551, 0.655] | 0.063 [0.054, 0.073] | 1.823 [1.717, 1.936] | 0.621 [0.521, 0.717] | 0.458 [0.358, 0.567] |
| awq_long_masked | 3 | 0.551 [0.500, 0.603] | 0.085 [0.072, 0.099] | 1.913 [1.799, 2.040] | 0.546 [0.450, 0.642] | 0.475 [0.379, 0.579] |

Paired deltas (A − B, same items; only seeds present for both are used):

| A − B | call_exact | call_nll_tok | answer_nll_tok | gen_correct_heldout | gen_correct_indist |
|---|---|---|---|---|---|
| awq_long_masked − awq_long_nomask (3 seeds) | -0.051 [-0.089, -0.010] | +0.022 [+0.015, +0.029] | +0.089 [+0.071, +0.109] | -0.075 [-0.163, +0.012] | +0.017 [-0.071, +0.092] |
| awq_long_masked − awq_short (3 seeds) | -0.006 [-0.044, +0.034] | +0.015 [+0.008, +0.022] | -0.198 [-0.218, -0.177] | -0.029 [-0.104, +0.042] | +0.029 [-0.058, +0.117] |
| awq_long_nomask − awq_short (3 seeds) | +0.045 [+0.007, +0.083] | -0.007 [-0.011, -0.003] | -0.287 [-0.306, -0.267] | +0.046 [-0.042, +0.133] | +0.013 [-0.062, +0.092] |
| rtn − fp (1 seed) | -0.197 [-0.248, -0.146] | +0.026 [+0.018, +0.033] | +0.214 [+0.193, +0.237] | -0.200 [-0.325, -0.075] | -0.025 [-0.112, +0.062] |
| awq_short − rtn (3 seeds) | +0.098 [+0.043, +0.152] | -0.018 [-0.025, -0.011] | +0.090 [+0.065, +0.115] | +0.138 [+0.042, +0.233] | -0.017 [-0.104, +0.071] |
| awq_long_nomask − rtn (3 seeds) | +0.143 [+0.087, +0.199] | -0.025 [-0.032, -0.018] | -0.197 [-0.220, -0.176] | +0.183 [+0.058, +0.304] | -0.004 [-0.083, +0.075] |
| awq_long_masked − rtn (3 seeds) | +0.092 [+0.043, +0.139] | -0.003 [-0.010, +0.006] | -0.108 [-0.133, -0.084] | +0.108 [-0.004, +0.217] | +0.013 [-0.100, +0.117] |

AWQ alpha agreement with the main-grid run of the same condition/seed (1.0 = identical scales): awq_short s0: 1.0, awq_short s1: 1.0, awq_short s2: 1.0, awq_long_nomask s0: 1.0, awq_long_nomask s1: 1.0, awq_long_nomask s2: 1.0, awq_long_masked s0: 1.0, awq_long_masked s1: 1.0, awq_long_masked s2: 1.0
