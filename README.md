# Police Complaint AI Assistant

An AI-assisted police complaint intake and triage portal. Citizens can file and track complaints, upload evidence, and review follow-up questions. Officers can review complaints, filter the triage queue, update case status and notes, and download evidence.

## Tech stack

- Frontend: Next.js, React, TypeScript, Tailwind CSS, shadcn/ui
- Backend: FastAPI and Pydantic
- AI: LangChain with OpenRouter, with optional Groq fallback
- Triage: explainable Python rules in `backend/triage.py`
- Storage: SQLite plus local evidence files under `uploads/evidence/`

## Project structure

```text
backend/
  main.py          FastAPI application entry point
  routes.py        Complaint, triage, and evidence endpoints
  schemas.py       API request and response models
  ai_service.py    Complaint classification and extraction
  triage.py        Priority, risk, routing, and action rules
  questions.py     Follow-up question generation
  database.py      SQLite persistence and migrations
frontend/
  src/app/         Next.js application routes
  src/components/  Shared UI components
  src/lib/         API client, types, and UI helpers
uploads/evidence/  Runtime evidence storage; not committed
```

## Setup

Create and activate a Python environment, then install backend dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create the root environment file and add an OpenRouter key:

```bash
cp .env.example .env
```

```dotenv
OPENROUTER_API_KEY=your_openrouter_api_key
GROQ_API_KEY=your_optional_groq_api_key
```

Install frontend dependencies:

```bash
cd frontend
npm install
```

The frontend reads `NEXT_PUBLIC_API_URL` and defaults to `http://localhost:8000` when it is not set.

## Run locally

Start the backend from the repository root:

```bash
uvicorn backend.main:app --reload
```

The API runs at `http://localhost:8000`; interactive API documentation is available at `http://localhost:8000/docs`.

In a second terminal, start the frontend:

```bash
cd frontend
npm run dev
```

The web application runs at `http://localhost:3000`.

## Application routes

- `/citizen/file-complaint` - submit a complaint and upload evidence
- `/citizen/track-complaint` - retrieve a complaint by its ID
- `/officer/dashboard` - search and filter the triage queue
- `/officer/complaints/[id]` - review and update a complaint
- `/officer/evidence-review` - review uploaded evidence

## API endpoints

- `POST /complaints` - analyze and create a complaint
- `GET /complaints` - list complaints
- `GET /complaints/{id}` - retrieve one complaint
- `PATCH /complaints/{id}/triage` - update status and officer notes
- `POST /complaints/{id}/evidence` - upload evidence
- `GET /complaints/{id}/evidence` - list evidence metadata
- `GET /evidence/{id}` - download an evidence file

Evidence uploads accept JPG, JPEG, PNG, PDF, and TXT files up to 10 MB per file. SQLite tables and missing columns are initialized automatically when the backend starts.

## Verification

```bash
python -m py_compile backend/*.py
cd frontend
npm run lint
npm run build
```
