"""Адреса проекта в одном месте. Переехали — правим только этот файл.

Используют: doctor.py, add_member.py, tasks.py, bootstrap_github.py.
"""

ORG = "alfahack-ilove4fak"  # организация GitHub, владелец репозитория и доски
REPO = f"{ORG}/alfahack-automl"
CAPTAIN = "thenlaooo"  # ник капитана: сливает в main, запасной ревьюер
PROJECT_NUMBER = 1  # номер доски задач в организации
PROJECT_TITLE = "Альфа Хак — AutoML-агент"

REPO_URL = f"https://github.com/{REPO}"
BOARD_URL = f"https://github.com/orgs/{ORG}/projects/{PROJECT_NUMBER}"
INVITATION_URL = f"https://github.com/orgs/{ORG}/invitation"
# GraphQL-запрос id доски (для добавления задач и выдачи доступа)
PROJECT_ID_QUERY = f'query={{organization(login:"{ORG}"){{projectV2(number:{PROJECT_NUMBER}){{id}}}}}}'
PROJECT_ID_JQ = ".data.organization.projectV2.id"
