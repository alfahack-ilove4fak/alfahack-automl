# Материалы: Капитан / интегратор (проверено 2026-10-20)

Общие материалы — [README.md](README.md).

| Материал | Чем полезен |
|---|---|
| `reference/baseline.py` + PDF кейса | первоисточник интерфейса `solution.py` и правил платформы |
| [Python 3.12: subprocess](https://docs.python.org/3.12/library/subprocess.html) | песочница для кода LLM: отдельный процесс, таймаут, чистое окружение |
| [Python 3.12: resource](https://docs.python.org/3.12/library/resource.html) | лимит памяти для песочницы (Linux — как на платформе) |
| [psutil](https://psutil.readthedocs.io/en/latest/) | сколько памяти осталось, пиковое потребление |
| [Docker: resource constraints](https://docs.docker.com/engine/containers/resource_constraints/) | как платформа ограничивает контейнер — и как читать лимит изнутри (cgroup) |
| [GitHub Docs: Pull requests](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests) | merge-стратегии, ревью — для релизов `dev → main` |
| [GitHub Docs: Projects](https://docs.github.com/en/issues/planning-and-tracking-with-projects) | доска, поля, представления — для стендапов |
| [GitHub Actions](https://docs.github.com/en/actions) | `run-agent`, сторож веток |
| [Building effective agents (Anthropic)](https://www.anthropic.com/engineering/building-effective-agents) | общая архитектура агента — для решений и питча |
| [karpathy/autoresearch](https://github.com/karpathy/autoresearch) | проект из кейса — ориентир, как подать AutoResearch жюри |
| [Claude Code: best practices](https://www.anthropic.com/engineering/claude-code-best-practices) | как команде эффективно работать с Claude |
