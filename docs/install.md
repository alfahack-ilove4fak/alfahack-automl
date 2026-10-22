# Установка: общая для всех ролей (пункт 3.1)

Ставим 5 вещей: **Git, GitHub CLI (gh), uv (он же поставит Python 3.12), Claude Code, редактор**. Ролевые дополнения — в `docs/roles/<роль>.md`.
Каждый шаг: **команда → что должно появиться → что делать, если не появилось.**

> **Какой терминал открыть.** Windows: «PowerShell» (в начале строки `PS C:\...>`). macOS: «Терминал». Linux: любой терминал.
> После каждой установки **закрой и открой терминал заново** — иначе новая программа может «не найтись».

> **Медленный интернет / не качается?** Внизу раздел [«Если не скачивается»](#если-не-скачивается).

> **Windows: где держать проект.** Клонируй в короткий путь **без русских букв и пробелов**, например `C:\dev\`. Если имя пользователя Windows на кириллице (`C:\Users\Данил\…`), часть ML-библиотек может спотыкаться о такой путь.
> ```
> mkdir C:\dev
> cd C:\dev
> ```

---

## 1. Git

| ОС | Команда |
|---|---|
| Windows | `winget install --id Git.Git -e` |
| macOS | `xcode-select --install` (или `brew install git`) |
| Linux (Ubuntu/Debian) | `sudo apt update && sudo apt install -y git` |

**Проверка:** `git --version` → `git version 2.x.x`
**Не появилось:** на Windows перезапусти PowerShell; если `winget` не найден — скачай установщик с https://git-scm.com/downloads/win (все галочки по умолчанию).

Представься Git'у (один раз на компьютере; почту можно взять анонимную из https://github.com/settings/emails — вида `12345+nick@users.noreply.github.com`):
```
git config --global user.name "Твой Ник"
git config --global user.email "почта@для.github"
```
**Проверка:** `git config user.name` → печатает имя.

## 2. GitHub CLI (`gh`)

| ОС | Команда |
|---|---|
| Windows | `winget install --id GitHub.cli -e` |
| macOS | `brew install gh` |
| Linux (Ubuntu/Debian) | `sudo apt install -y gh` |

**Проверка:** `gh --version` → `gh version 2.x.x`
Войди в аккаунт:
```
gh auth login
```
Отвечай: `GitHub.com` → `HTTPS` → `Yes` (использовать для git) → `Login with a web browser`. Скопируй код, вставь в браузере.
**Проверка:** `gh auth status` → `✓ Logged in to github.com account <твой ник>`
**Не появилось:** `gh auth logout` и повтори; на Linux, если `apt` не находит `gh`, — инструкция https://github.com/cli/cli/blob/trunk/docs/install_linux.md

## 3. uv и Python 3.12

uv — менеджер Python и зависимостей. Он сам скачает нужный Python 3.12 и **не тронет** системный Python.

| ОС | Команда |
|---|---|
| Windows | `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 \| iex"` (или `winget install --id=astral-sh.uv -e`) |
| macOS | `curl -LsSf https://astral.sh/uv/install.sh \| sh` (или `brew install uv`) |
| Linux | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |

**Проверка:** `uv --version` → `uv 0.x.x`
Теперь Python:
```
uv python install 3.12
```
**Проверка:** `uv python find 3.12` → печатает путь к python 3.12.
**Не появилось:** перезапусти терминал (uv добавляет себя в PATH). Если скачивание висит — см. раздел ниже.

> **Важно:** команды `python` / `python3` после этого могут **не работать** — и это нормально. uv ставит Python только для проектов; в нашем репозитории всё запускается через `uv run python …`. На Windows `python` может открыть Microsoft Store — не обращай внимания, используй `uv run`.

### 3а. Системная библиотека для LightGBM (одна команда)

LightGBM (основная модель агента) требует системную библиотеку. Без неё `doctor.py` покажет ❌ «библиотеки ядра» с ошибкой загрузки `lightgbm`.

| ОС | Команда | Нужна, если |
|---|---|---|
| Windows | `winget install --id Microsoft.VCRedist.2015+.x64 -e` | не установлена Visual Studio 2015+ (обычно — нужна) |
| macOS | `brew install libomp` | всегда (Homebrew: https://brew.sh) |
| Linux | — | обычно уже есть; если ошибка `libgomp.so.1` — `sudo apt install -y libgomp1` |

Источник: README пакета LightGBM 4.6 (раздел установки). Если `winget` не нашёл пакет — скачай «Visual C++ Redistributable x64» с https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist.

## 4. Claude Code

Нужна подписка Pro/Max (у всех нас Pro).

| ОС | Команда |
|---|---|
| Windows (PowerShell) | `irm https://claude.ai/install.ps1 \| iex` (или `winget install Anthropic.ClaudeCode`) |
| macOS / Linux | `curl -fsSL https://claude.ai/install.sh \| bash` (или на macOS `brew install --cask claude-code`) |

**Проверка:** открой **новый** терминал → `claude --version` → `2.x.x (Claude Code)`. Глубокая проверка: `claude doctor`.
**Не появилось:** `claude` не найден → папка установки не в PATH: https://code.claude.com/docs/en/troubleshoot-install
На Windows обязательно стоит Git из шага 1 — Claude Code использует из него Git Bash.

Вход и настройка под наш проект — в [SETUP_CLAUDE.md](../SETUP_CLAUDE.md) (после клонирования репозитория).

## 5. Редактор

Рекомендуем **VS Code** (https://code.visualstudio.com) + расширения: Python, Jupyter (для Research), Claude Code.
Windows: `winget install --id Microsoft.VisualStudioCode -e` · macOS: `brew install --cask visual-studio-code` · Linux: `sudo snap install code --classic`.

---

## Если не скачивается

У части сетей (проверено у капитана) сайты `astral.sh`, `github.com`, `pypi.org` отдают файлы очень медленно.

1. **Сначала спроси в чате** — возможно, у кого-то уже скачано: `uv` — один файл, его можно передать на флешке/в Telegram.
2. **uv вручную из GitHub Releases:** https://github.com/astral-sh/uv/releases → файл для своей ОС
   (`uv-x86_64-pc-windows-msvc.zip`, `uv-aarch64-apple-darwin.tar.gz` для Mac на M-чипах, `uv-x86_64-apple-darwin.tar.gz` для Intel Mac, `uv-x86_64-unknown-linux-gnu.tar.gz` для Linux) → распаковать → положить `uv` (`uv.exe`) в папку из PATH (`~/.local/bin` на Mac/Linux, `%USERPROFILE%\.local\bin` на Windows).
3. **uv через pip:** `pip install uv` (если есть любой Python).
4. **Зависимости проекта:** ставь сначала только ядро (`uv sync`, ~150 МБ). Тяжёлую группу `full` (torch, autogluon и др., ~1.5 ГБ) — только если она нужна твоей роли и интернет позволяет.
5. Если ничего не помогает — работай в **GitHub Codespaces** (браузер, быстрый интернет на стороне GitHub): на странице репозитория `Code → Codespaces → Create codespace on dev`. Бесплатно ~30 часов в месяц на 4-ядерной машине.

Следующий шаг — [ONBOARDING.md](../ONBOARDING.md), шаг 2.
