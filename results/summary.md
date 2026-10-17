| condition | seeds | call_exact | call_nll_tok | answer_nll_tok | chat_nll_tok | mmlu_acc |
|---|---|---|---|---|---|---|
| fp | 1 | 0.564 [0.518, 0.611] | 0.079 [0.069, 0.089] | 2.401 [2.280, 2.534] | 1.686 [1.552, 1.809] | 0.445 [0.395, 0.493] |
| rtn | 1 | 0.467 [0.418, 0.514] | 0.117 [0.107, 0.128] | 2.572 [2.444, 2.710] | 1.830 [1.690, 1.959] | 0.340 [0.295, 0.385] |

Paired deltas (A − B; same eval items; 95% cluster-bootstrap CI; mean over calibration seeds):

| A − B | call_exact | call_nll_tok | answer_nll_tok | chat_nll_tok | mmlu_acc |
|---|---|---|---|---|---|
| rtn − fp | -0.098 [-0.141, -0.052] | +0.038 [+0.032, +0.045] | +0.170 [+0.148, +0.194] | +0.144 [+0.126, +0.162] | -0.105 [-0.157, -0.055] |
