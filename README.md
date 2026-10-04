# Smart Focus

Система адаптивной визуализации и управления фокусом для больших разреженных графов — порядка 100 000 вершин и сотен тысяч рёбер.

Автоматически анализирует топологию графа, классифицирует структуру, выбирает оптимальный алгоритм укладки и реализует управление фокусом с сохранением структурно важных элементов (мостов и хабов).

Ключевые компоненты:

- **Многоуровневый алгоритм укладки** (`backend/app/fastlayout.py`): огрубление паросочетанием по тяжёлым рёбрам со структурно-информированным запретом склейки разных сообществ, модель сил (1,−1), гибридная аппроксимация отталкивания (точное ближнее поле по сетке + случайная выборка дальнего), адаптивный шаг Ху.
- **WebGL-рендерер** (`frontend/src/components/GraphGL.tsx`): вершины — gl.POINTS с процедурными окружностями в шейдере, рёбра — индексная отрисовка по общему буферу координат, прозрачность ребра выводится из его концов.

## Структура репозитория

| Каталог | Что внутри |
|---|---|
| `backend/` | сервер FastAPI: пакет `app`, `requirements.txt` |
| `frontend/` | клиент на React + Vite |

## Стек технологий

- **Backend:** Python, FastAPI, NetworkX, igraph + leidenalg (fallback: python-louvain), NumPy, SciPy
- **Frontend:** TypeScript, React, Vite, D3.js, HTML5 Canvas, WebGL

## Запуск

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Фронтенд запускается на `http://localhost:5173` и проксирует API-запросы на `http://localhost:8000`.

## Возможности

- Загрузка пользовательских графов по списку рёбер
- Автоматическая классификация топологии (дерево, разреженный, кластеризованный, плотный, малый мир)
- Адаптивный выбор алгоритма укладки (Spring, Circular, Spectral, Kamada-Kawai, Shell, Community, Multilevel)
- Управление фокусом с настраиваемой глубиной BFS и сохранением мостов/хабов
- Свёртка крупных сообществ в мета-узлы для обзора глобальной структуры

## API

| Endpoint | Описание |
|---|---|
| `POST /api/graph/generate.bin` | Генерация графа |
| `POST /api/graph/upload.bin` | Загрузка графа по списку рёбер |
| `POST /api/graph/layout.bin` | Смена алгоритма укладки |
| `POST /api/graph/collapse.bin` | Свёртка крупных сообществ |
| `POST /api/graph/subgraph.bin` | Вход внутрь сообщества |
| `POST /api/graph/focus.bin` | Вычисление контекста фокуса |
| `POST /api/graph/focus/reset` | Сброс фокуса |
| `GET /api/health` | Проверка состояния |

Документация API доступна по адресу `http://localhost:8000/docs`.
