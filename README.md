# Police Complaint AI Assistant

An AI-powered police complaint assistant that classifies complaints, extracts key details, assigns priority, and saves records.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit |
| Backend | FastAPI |
| AI | LangChain + Groq API |
| Storage | SQLite |

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
5. Get a Groq API key from https://console.groq.com/
6. Add your key to `.env`:
   ```
   GROQ_API_KEY=your_key_here
   ```

## Running

You need two terminals:

**Terminal 1 — Start the backend:**
```
uvicorn backend.main:app --reload
```

**Terminal 2 — Start the frontend:**
```
streamlit run frontend/app.py
```

The app opens at http://localhost:8501

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/complaints` | Submit a complaint for AI analysis |
| GET | `/complaints` | View all saved complaints |

## Project Structure

```
frontend/app.py       Streamlit UI
backend/main.py       FastAPI app entry point
backend/routes.py     API endpoints
backend/schemas.py    Request/response models
backend/ai_service.py LangChain + Groq AI service
backend/database.py   SQLite operations
backend/questions.py  Category-wise follow-up questions
backend/priority.py   Rule-based priority assignment
complaints.db         SQLite database (auto-created)
.env                  Environment variables
requirements.txt      Python dependencies
```

## How It Works

1. User enters a complaint in natural language
2. Streamlit sends it to the FastAPI backend
3. AI classifies the complaint and extracts location, time, persons, and summary
4. Backend assigns priority using simple Python rules
5. Complaint is saved in SQLite
6. Streamlit displays the AI result and follow-up questions
7. Records page shows all saved complaints
