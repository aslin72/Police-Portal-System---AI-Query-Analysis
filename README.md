# Police Complaint AI Assistant

A beginner-friendly FastAPI and Streamlit portal where an AI-guided chat collects a complete complaint, explainable Python rules assign police priority, and officers review the resulting queue.

## Features

- Guided citizen chat that asks one missing-detail question at a time
- Review-before-submit complaint flow with optional evidence upload
- OpenRouter extraction with Groq and deterministic local fallbacks
- Rule-based priority, risk flags, routing, and recommended action
- Complaint tracking by ID and a filterable officer dashboard
- SQLite storage and local evidence files under `uploads/evidence/`

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set `OPENROUTER_API_KEY` in `.env`; `GROQ_API_KEY` is an optional provider fallback. The portal remains usable without either key through simpler local extraction rules.

## Run

Start the backend from the project root:

```bash
uvicorn backend.main:app --reload
```

Start the frontend in another terminal:

```bash
streamlit run frontend/streamlit_app.py
```

Open `http://localhost:8501`. API documentation is at `http://localhost:8000/docs`.

## API

- `POST /intake/chat` - continue a guided complaint draft
- `POST /complaints` - finalize and triage a complaint
- `GET /complaints` and `GET /complaints/{id}` - list or track complaints
- `PATCH /complaints/{id}/triage` - update status and officer notes
- `POST /complaints/{id}/evidence` - upload JPG, PNG, PDF, or TXT files up to 10 MB
- `GET /complaints/{id}/evidence` and `GET /evidence/{id}` - list and download evidence

## Verify

```bash
python -m py_compile backend/*.py frontend/streamlit_app.py
```
