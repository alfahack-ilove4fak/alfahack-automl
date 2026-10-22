# Материалы: QA / MLOps (проверено 2026-10-20)

Общие материалы — [README.md](README.md).

| Материал | Чем полезен |
|---|---|
| [pytest](https://docs.pytest.org/en/stable/) | фикстуры, `parametrize`, `tmp_path`, `monkeypatch` — всё, что есть в наших тестах |
| [Ruff](https://docs.astral.sh/ruff/) | правила линтера и форматтер (наш конфиг — `pyproject.toml`) |
| [GitHub Actions](https://docs.github.com/en/actions) | workflow, артефакты, `workflow_dispatch`, кэш — наши `ci.yml`, `guard.yml`, `run-agent.yml` |
| [GitHub Codespaces](https://docs.github.com/en/codespaces/overview) | облачная машина 4 ядра/16 ГБ для 3-часовых репетиций |
| [Docker: resource constraints](https://docs.docker.com/engine/containers/resource_constraints/) | `--cpus`, `--memory` — эмуляция лимитов платформы |
| [Python 3.12: subprocess](https://docs.python.org/3.12/library/subprocess.html) | запуск агента отдельным процессом с таймаутом (харнесс, pre_submit) |
| [Python 3.12: resource](https://docs.python.org/3.12/library/resource.html) | лимит памяти процесса на Linux (`RLIMIT_AS`) |
| [psutil](https://psutil.readthedocs.io/en/latest/) | замер пиковой памяти и CPU в репетициях |
| [scikit-learn 1.5: Metrics](https://scikit-learn.org/1.5/modules/model_evaluation.html) | реестр метрик: точные определения и параметры (`average=`, `multi_class=`) |
| [uv](https://docs.astral.sh/uv/) | `uv sync --group`, кэш в CI, чистые окружения для pre_submit |
| [MLE-bench, arXiv:2410.07095](https://arxiv.org/abs/2410.07095) | как другие строят «харнесс» для оценки ML-агентов и защищаются от подгонки |
| PDF кейса, приложение 3 — `reference/` | формулы баллов для `evaluation/local_score.py` |
