<!-- harness-report -->
### 📊 Харнесс: было `c79a23a` (dev) → стало `750f5c0-dirty` (PR + dev)

| Задача | Метрика | Было | Стало | Балл: было → стало | Δ балла | Итог | Время, с | Пик RAM, МБ |
|---|---|---|---|---|---|---|---|---|
| tasks/practice/batches_ordinal_qwk | qwk | 0.74758 | 0.73529 | 74.8 → 73.5 | -1.2 | ⚪ в пределах шума | 203 | 567 |
| tasks/practice/cancer_binary_auc | roc_auc | 0.99292 | 0.99292 | 98.6 → 98.6 | +0.0 | ⚪ в пределах шума | 20 | 394 |
| tasks/practice/claims_regression_mape_rowno | mape | 0.48206 | 0.48206 | 52.6 → 52.6 | +0.0 | ⚪ в пределах шума | 130 | 378 |
| tasks/practice/diabetes_regression_mae | mae | 48.98 | 48.98 | 21.1 → 21.1 | +0.0 | ⚪ в пределах шума | 7 | 281 |
| tasks/practice/digits_multiclass_accuracy | accuracy | 0.9873 | 0.9873 | 98.6 → 98.6 | +0.0 | ⚪ в пределах шума | 165 | 764 |
| tasks/practice/network_binary_logloss_crooked | logloss | 0.10544 | 0.10544 | 83.4 → 83.4 | +0.0 | ⚪ в пределах шума | 139 | 441 |
| tasks/practice/payments_binary_f1_rare | f1 | 0.33222 | 0.37948 | 33.2 → 37.9 | +4.7 | 🟢 лучше | 188 | 475 |
| tasks/practice/sales_regression_dates_rmsle | rmsle | 0.46989 | 0.46989 | 54.7 → 54.7 | +0.0 | ⚪ в пределах шума | 125 | 478 |
| tasks/practice/search_ranking_ndcg | ndcg | 0.97336 | 0.97336 | 86.0 → 86.0 | +0.0 | ⚪ в пределах шума | 122 | 353 |
| tasks/practice/utilities_multitarget_mae | mae | 0.80338 | 0.80338 | 60.2 → 60.2 | +0.0 | ⚪ в пределах шума | 45 | 374 |
| tasks/public/multiclass | f1_macro | 0.64736 | 0.6455 | 62.9 → 62.8 | -0.2 | ⚪ в пределах шума | 216 | 370 |
| tasks/public/regression | rmse | 50.406 | 50.406 | 11.3 → 11.3 | +0.0 | ⚪ в пределах шума | 99 | 452 |

**Средний балл: 61.4 → 61.7 (+0.3)**

<sub>без LLM, один прогон. Балл — по формуле организаторов относительно тривиального ответа (`evaluation/local_score.py`). Разница меньше 2 балла может быть шумом. Перезапуск — снять и снова поставить метку `harness`.</sub>
