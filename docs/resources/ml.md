# Материалы: ML-pipeline инженер (проверено 2026-10-20)

Общие материалы — [README.md](README.md). Ссылки на документацию — **наших версий** из `requirements.txt`.

## Документация моделей и инструментов

| Материал | Чем полезен |
|---|---|
| [LightGBM 4.6: Parameters Tuning](https://lightgbm.readthedocs.io/en/v4.6.0/Parameters-Tuning.html) | какие параметры крутить для точности / скорости / против переобучения |
| [LightGBM 4.6: Parameters](https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html) | полный список параметров и их алиасов |
| [CatBoost: CatBoostClassifier](https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier) | параметры, `cat_features`, ранняя остановка |
| [XGBoost 2.1: параметры](https://xgboost.readthedocs.io/en/release_2.1.0/parameter.html) | параметры бустинга, `enable_categorical` |
| [scikit-learn 1.5: Cross-validation](https://scikit-learn.org/1.5/modules/cross_validation.html) | KFold, StratifiedKFold, GroupKFold, TimeSeriesSplit — когда какой |
| [scikit-learn 1.5: GroupKFold](https://scikit-learn.org/1.5/modules/generated/sklearn.model_selection.GroupKFold.html) | фолды без «одного здания в train и в валидации» |
| [scikit-learn 1.5: Common pitfalls](https://scikit-learn.org/1.5/common_pitfalls.html) | утечки данных и непоследовательная предобработка — must read |
| [scikit-learn 1.5: Metrics](https://scikit-learn.org/1.5/modules/model_evaluation.html) | все метрики и их тонкости (macro/weighted, proba vs label) |
| [Optuna 4.2.1](https://optuna.readthedocs.io/en/v4.2.1/) | подбор гиперпараметров, `timeout`, pruning |
| [AutoGluon Tabular](https://auto.gluon.ai/stable/tutorials/tabular/index.html) | AutoML «под ключ»: как гипотеза или источник идей (пресеты, стекинг) ⚠️ у нас 1.3.1, а ссылка на stable — сверяй API |
| [LightAutoML](https://lightautoml.readthedocs.io/en/latest/) | AutoML от Сбера, силён на табличных задачах ⚠️ у нас 0.4.2 |
| [FLAML: Task-Oriented AutoML](https://microsoft.github.io/FLAML/docs/Use-Cases/Task-Oriented-AutoML) | лёгкий AutoML с жёстким бюджетом времени — подходит под наши 3 часа |
| [pandas 2.2 User Guide](https://pandas.pydata.org/pandas-docs/version/2.2/user_guide/index.html) | работа с данными, типы, категории |

## Практика и исследования

| Материал | Чем полезен |
|---|---|
| [Kaggle Learn: Feature Engineering](https://www.kaggle.com/learn/feature-engineering) | mutual information, кластеры, target encoding — с упражнениями |
| [Why tree-based models still outperform deep learning on tabular data, arXiv:2207.08815](https://arxiv.org/abs/2207.08815) | почему бустинги — основа агента, а нейросети — нет |
| [AIDE, arXiv:2502.13138](https://arxiv.org/abs/2502.13138) | какие ML-шаги агенты делают успешно на Kaggle-задачах |
| [WiDS Datathon 2022 (Kaggle)](https://www.kaggle.com/competitions/widsdatathon2022) | ⚠️ похоже на источник нашей регрессии (site_eui, Year_Factor, State_Factor) — обсуждения и решения участников. **Только для понимания данных**: хаки под этот датасет в агента не берём |
| [UNSW-NB15 (UNSW)](https://research.unsw.edu.au/projects/unsw-nb15-dataset) | первоисточник многоклассовой задачи: описание признаков сетевых потоков |
