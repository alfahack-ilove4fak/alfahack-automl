Network flows — attack or not?
==============================

Predict whether each network flow is malicious. Files are in `/data/input/flows_train.csv` and
`/data/input/flows_test.csv` (paths may differ on your machine).

Train has 19600 rows, test has 8400. The target column is the one that is missing in the test file.
Each row is identified by `flow_id`.

**Evaluation.** Accuracy is misleading here, so we do NOT use it. Submissions are scored with
binary cross-entropy on the predicted probability that the flow is an attack (target = 1). Lower is better.

Submission: a CSV with columns `flow_id` and `is_attack`, where `is_attack` is a probability in [0, 1].
