# Материалы: LLM/agent-инженер (проверено 2026-10-20)

Общие материалы — [README.md](README.md).

## Документация

| Материал | Чем полезен |
|---|---|
| [openai-python (GitHub)](https://github.com/openai/openai-python) | наш клиент к Qwen (OpenAI-совместимый API): `base_url`, `timeout`, `max_retries`, ответы с `usage` для учёта токенов. Версия у нас 1.68.2 |
| [Qwen docs](https://qwen.readthedocs.io/en/latest/) | особенности моделей Qwen: формат чата, рекомендуемые параметры генерации, function calling ⚠️ точная модель на платформе — уточнить у организаторов |
| [Pydantic 2.10](https://docs.pydantic.dev/2.10/) | валидация ответов LLM (`model_validate`, `model_validate_json`) — так мусор от модели не ломает цикл |
| [LangGraph](https://langchain-ai.github.io/langgraph/) | если решим строить агента графом ролей (есть в requirements, v0.6.7) ⚠️ не обязателен |
| [Prompting Guide](https://www.promptingguide.ai/) | техники промптов: few-shot, chain-of-thought, ReAct — с примерами |
| [Building effective agents (Anthropic)](https://www.anthropic.com/engineering/building-effective-agents) | когда нужен «агент», а когда хватит простого workflow; паттерны evaluator-optimizer, orchestrator-workers — прямо про наш цикл |

## Решения и исследования других (AutoML/DS-агенты)

| Материал | Чем полезен |
|---|---|
| [karpathy/autoresearch](https://github.com/karpathy/autoresearch) | проект, на который ссылается кейс: агент сам улучшает обучение модели циклом «изменение → эксперимент → оценка» |
| [AIDE — WecoAI/aideml](https://github.com/WecoAI/aideml) + статья [arXiv:2502.13138](https://arxiv.org/abs/2502.13138) | дерево решений: агент черновик → исправление ошибок → улучшение лучшего узла. Хорошая модель для нашего выбора «какую гипотезу развивать» |
| [DS-Agent — guosyjlu/DS-Agent](https://github.com/guosyjlu/DS-Agent) + [arXiv:2402.17453](https://arxiv.org/abs/2402.17453) | «рассуждение по прецедентам»: агент берёт похожие решённые задачи как примеры — аналог нашего playbook |
| [Data Interpreter, arXiv:2402.18679](https://arxiv.org/abs/2402.18679) (код в [MetaGPT](https://github.com/geekan/MetaGPT)) | планирование DS-задачи графом шагов, проверка промежуточных результатов |
| [AutoML-Agent, arXiv:2410.02958](https://arxiv.org/abs/2410.02958) | мультиагентный AutoML: разбор требований → план → исполнение → проверка — идеи для ролей внутри агента |
| [R&D-Agent — microsoft/RD-Agent](https://github.com/microsoft/RD-Agent) + [arXiv:2505.14738](https://arxiv.org/abs/2505.14738) | разделение «исследователь предлагает / разработчик реализует» — как у нас agent и ml |
| [MLE-bench, arXiv:2410.07095](https://arxiv.org/abs/2410.07095) | как оценивают ML-агентов на Kaggle-задачах; типичные провалы агентов |

## Подсказки по нашему кейсу

- Модель небольшая — короткие промпты, один шаг за запрос, строгий JSON-формат ответа с примером.
- Агент обязан работать без LLM: любой вызов → `None` при ошибке → fallback на playbook.
- Кейс прямо предупреждает про «взлом метрики» агентами (пример Weco AI) — гипотезы принимаем только по честной CV.
