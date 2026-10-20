# Материалы: Исследователь / Data Scientist (проверено 2026-10-20)

Общие материалы — [README.md](README.md). Документация моделей — в [ml.md](ml.md).

## Чужие решения AutoML/DS-агентов (для R-L2-2 и playbook)

| Материал | Чем полезен |
|---|---|
| [karpathy/autoresearch](https://github.com/karpathy/autoresearch) | проект из кейса: цикл AutoResearch на практике — как формулируются направления исследования |
| [AIDE, arXiv:2502.13138](https://arxiv.org/abs/2502.13138) · [код](https://github.com/WecoAI/aideml) | поиск по дереву решений; какие улучшения агенты находят чаще всего |
| [DS-Agent, arXiv:2402.17453](https://arxiv.org/abs/2402.17453) · [код](https://github.com/guosyjlu/DS-Agent) | банк прецедентов решённых задач — прообраз нашего playbook |
| [AutoKaggle, arXiv:2410.20424](https://arxiv.org/abs/2410.20424) | мультиагентное решение соревнований по фазам (EDA → признаки → модель), с проверками на каждом шаге |
| [Data Interpreter, arXiv:2402.18679](https://arxiv.org/abs/2402.18679) | план DS-задачи как граф шагов |
| [AutoML-Agent, arXiv:2410.02958](https://arxiv.org/abs/2410.02958) | полный пайплайн AutoML мультиагентами — идеи для универсальных гипотез |
| [MLAgentBench, arXiv:2310.03302](https://arxiv.org/abs/2310.03302) · [код](https://github.com/snap-stanford/MLAgentBench) | как измерять успех ML-агента; типичные ошибки (галлюцинации, «бег по кругу») |
| [MLE-bench, arXiv:2410.07095](https://arxiv.org/abs/2410.07095) · [код](https://github.com/openai/mle-bench) | бенчмарк ML-агентов на 75 Kaggle-задачах — откуда брать идеи «что работает в среднем» |
| [R&D-Agent, arXiv:2505.14738](https://arxiv.org/abs/2505.14738) · [код](https://github.com/microsoft/RD-Agent) | цикл «гипотеза → реализация → обратная связь», лидер MLE-bench на момент публикации ⚠️ проверь актуальность |

## Валидация и честная оценка (для R-L2-1)

| Материал | Чем полезен |
|---|---|
| [scikit-learn 1.5: Cross-validation](https://scikit-learn.org/1.5/modules/cross_validation.html) | схемы фолдов и когда их применять |
| [scikit-learn 1.5: Common pitfalls](https://scikit-learn.org/1.5/common_pitfalls.html) | утечки — первое, что проверять в «слишком хорошем» результате |
| [Kaggle Learn: Intermediate ML](https://www.kaggle.com/learn/intermediate-machine-learning) | уроки про утечки и кросс-валидацию на простых примерах |

## Данные публичных задач

| Материал | Чем полезен |
|---|---|
| [WiDS Datathon 2022 (Kaggle)](https://www.kaggle.com/competitions/widsdatathon2022) | ⚠️ вероятный источник регрессии: обсуждения и EDA участников. Выводы — только универсальные |
| [UNSW-NB15 (UNSW)](https://research.unsw.edu.au/projects/unsw-nb15-dataset) | первоисточник классификации атак: что значат признаки |

## Графики и питч

| Материал | Чем полезен |
|---|---|
| [seaborn tutorial](https://seaborn.pydata.org/tutorial.html) | быстрые красивые графики для EDA и слайдов |
| [matplotlib 3.9](https://matplotlib.org/3.9.3/index.html) | всё, чего нет в seaborn |
| [Building effective agents (Anthropic)](https://www.anthropic.com/engineering/building-effective-agents) | язык и схемы, чтобы объяснить жюри устройство агента |
