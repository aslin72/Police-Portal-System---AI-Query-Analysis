# Police Complaint AI Assistant

An AI-powered police complaint assistant that classifies complaints, extracts key details, assigns priority, and saves records.

## Features

- Natural language complaint input
- AI-powered classification and entity extraction
- Rule-based priority assignment
- Category-wise follow-up questions
- SQLite storage for complaint records
- Streamlit web interface

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit |
| Backend | FastAPI |
| AI | LangChain + OpenRouter (meta-llama/llama-3.3-70b-instruct:free) |
| Priority | Rule-based Python |
| Storage | SQLite |

## Project Structure

```
backend/
  main.py         FastAPI app entry point
  routes.py       API endpoints
  schemas.py      Request/response models
  ai_service.py   LangChain + OpenRouter AI service (Groq fallback)
  database.py     SQLite operations
  questions.py    Category-wise follow-up questions
  priority.py     Rule-based priority assignment
frontend/
  app.py          Streamlit UI
complaints.db     SQLite database (auto-created)
.env              Environment variables (do not commit)
.env.example      Template for .env
requirements.txt  Python dependencies
```

## Setup

1. Clone the repository
2. Create a virtual environment:
   ```
   python -m venv venv
   ```
3. Activate it:
   - Mac/Linux: `source venv/bin/activate`
   - Windows: `venv\Scripts\activate`
4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
5. Get an OpenRouter API key from https://openrouter.ai/keys (free tier available).
6. Copy `.env.example` to `.env` and add your key:
   ```
   cp .env.example .env
   ```
   Then edit `.env` and set `OPENROUTER_API_KEY`. A Groq key is optional — only needed if OpenRouter fails.

## Running

You need two terminals:

**Terminal 1 — Start the backend:**
```
uvicorn backend.main:app --reload
```

Backend runs at http://localhost:8000  
Swagger docs at http://localhost:8000/docs

**Terminal 2 — Start the frontend:**
```
streamlit run frontend/app.py
```

Frontend runs at http://localhost:8501

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/complaints` | Submit a complaint for AI analysis |
| GET | `/complaints` | View all saved complaints |

## Example Input

```json
{
  "complaint_text": "Yesterday night near Tambaram, one car hit a bike and the rider was bleeding badly."
}
```

## Expected Output

```text
Category: road accident
Location: Tambaram
Priority: High
Follow-up Questions: road accident related questions
```

## How It Works

1. User enters a complaint in natural language
2. Streamlit sends it to the FastAPI backend
3. AI classifies the complaint and extracts location, time, persons, and summary
4. Backend assigns priority using simple Python rules
5. Complaint is saved in SQLite
6. Streamlit displays the AI result and follow-up questions
7. Records page shows all saved complaints
