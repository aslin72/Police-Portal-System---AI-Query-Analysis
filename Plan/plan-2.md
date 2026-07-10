
# Critical Fix Plan — Police Complaint AI Assistant

## 1. Goal

Fix the current codebase so it runs cleanly before adding any new feature.

The priority is not feature development now. The priority is making the existing FastAPI + Streamlit + Groq + SQLite project stable, readable, and beginner-friendly.

## 2. Main Problems to Fix

The current codebase has these critical issues:

```text
1. Some Python files may be collapsed into one-line format.
2. Python indentation may be broken.
3. Backend should be tested independently before Streamlit.
4. AI JSON response handling should be safer.
5. Database initialization should be clear.
6. Frontend should handle backend errors properly.
7. README should include correct run commands.
```

## 3. Final Expected Project Structure

Keep the same simple structure:

```text
Police-Portal-System---AI-Query-Analysis/
│
├── backend/
│   ├── main.py
│   ├── routes.py
│   ├── schemas.py
│   ├── ai_service.py
│   ├── database.py
│   ├── questions.py
│   └── priority.py
│
├── frontend/
│   └── app.py
│
├── requirements.txt
├── .env.example
├── README.md
└── complaints.db
```

Do not add extra folders now.

## 4. Fix Order

## Step 1: Fix Python File Formatting

Open every Python file and check indentation manually.

Files to inspect:

```text
backend/main.py
backend/routes.py
backend/schemas.py
backend/ai_service.py
backend/database.py
backend/questions.py
backend/priority.py
frontend/app.py
```

Each file must have proper line breaks.

Bad format:

```python
from fastapi import FastAPI app = FastAPI() @app.get("/") def home(): return {"message": "ok"}
```

Correct format:

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "ok"}
```

This is the first fix because Python depends on indentation.

## Step 2: Run Syntax Check

After formatting, run this command from the project root:

```bash
python -m py_compile backend/main.py backend/routes.py backend/schemas.py backend/ai_service.py backend/database.py backend/questions.py backend/priority.py frontend/app.py
```

Expected result:

```text
No output
```

If there is no output, syntax is fine.

If there is an error, fix that file first before moving forward.

## Step 3: Clean `main.py`

`main.py` should only start FastAPI and include routes.

Keep it small.

Expected responsibility:

```text
Create FastAPI app
Enable CORS
Include complaint routes
Create database table on startup
```

Do not put AI logic or database logic inside `main.py`.

Recommended flow:

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes import router
from backend.database import create_table

app = FastAPI(title="Police Complaint AI Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

create_table()

app.include_router(router)
```

## Step 4: Clean `schemas.py`

`schemas.py` should only contain Pydantic models.

Keep only the required models.

Recommended models:

```python
from pydantic import BaseModel
from typing import List


class ComplaintRequest(BaseModel):
    complaint_text: str


class ComplaintResponse(BaseModel):
    id: int
    complaint_text: str
    category: str
    location: str
    incident_time: str
    persons_involved: List[str]
    summary: str
    priority: str
    followup_questions: List[str]
```

Do not add advanced schemas now.

## Step 5: Clean `ai_service.py`

`ai_service.py` should only handle AI classification and extraction.

It should return structured data like this:

```json
{
  "category": "road accident",
  "location": "Tambaram",
  "incident_time": "yesterday night",
  "persons_involved": ["car driver", "bike rider"],
  "summary": "Road accident involving a car and bike."
}
```

Critical rules:

```text
The model must return JSON only.
Do not parse random text manually.
Validate category after AI response.
If JSON parsing fails, return safe default values.
```

Safe default:

```python
{
    "category": "general issue recorded",
    "location": "Not specified",
    "incident_time": "Not specified",
    "persons_involved": [],
    "summary": complaint_text
}
```

## Step 6: Clean `priority.py`

`priority.py` should use simple rule-based priority.

Do not let AI fully decide priority.

Recommended logic:

```text
High:
child safety
murder / serious crime incident
fire accident
complaint contains injury, bleeding, missing, death, fatal, weapon

Medium:
cyber crime incident
public healthcare
complaint contains fraud, money, account, bank

Low:
general issue recorded
unclear or incomplete complaint
```

Keep this file short.

## Step 7: Clean `questions.py`

`questions.py` should only contain follow-up questions.

Use a simple dictionary.

Example:

```python
FOLLOWUP_QUESTIONS = {
    "road accident": [
        "Were there any injuries?",
        "Are there any witnesses?",
        "What vehicles were involved?"
    ],
    "cyber crime incident": [
        "What type of cybercrime occurred?",
        "Was there any financial loss?",
        "Do you have suspect details?"
    ]
}
```

Add a function:

```python
def get_questions(category: str):
    return FOLLOWUP_QUESTIONS.get(category, FOLLOWUP_QUESTIONS["general issue recorded"])
```

## Step 8: Clean `database.py`

`database.py` should only handle SQLite.

Required functions:

```text
create_table()
save_complaint()
get_complaints()
```

Do not mix API logic inside this file.

Database fields:

```text
id
complaint_text
category
location
incident_time
persons_involved
summary
priority
followup_questions
created_at
```

Store lists as JSON strings:

```text
persons_involved
followup_questions
```

This keeps SQLite simple.

## Step 9: Clean `routes.py`

`routes.py` should connect the backend flow.

Expected backend flow:

```text
Receive complaint text
Call AI service
Assign priority
Get follow-up questions
Save complaint
Return response
```

Only keep two endpoints:

```text
POST /complaints
GET /complaints
```

Do not add extra endpoints now.

## Step 10: Test Backend First

Run FastAPI:

```bash
uvicorn backend.main:app --reload
```

Open Swagger:

```text
http://localhost:8000/docs
```

Test this input:

```json
{
  "complaint_text": "Yesterday night near Tambaram, one car hit a bike and the rider was bleeding badly."
}
```

Expected output should include:

```text
category: road accident
location: Tambaram
priority: High
followup_questions: road accident related questions
```

Do not touch Streamlit until FastAPI works.

## Step 11: Clean `frontend/app.py`

Streamlit should only handle UI.

Frontend responsibilities:

```text
Take user complaint
Send POST request to FastAPI
Display AI classification
Display extracted details
Display priority
Display follow-up questions
Show saved complaints
```

Use `requests` to call backend.

Backend URL:

```python
API_URL = "http://localhost:8000"
```

Basic call:

```python
response = requests.post(
    f"{API_URL}/complaints",
    json={"complaint_text": complaint_text}
)
```

Handle backend failure:

```python
if response.status_code != 200:
    st.error("Backend error. Please check if FastAPI is running.")
```

## Step 12: Test Full App

Start backend:

```bash
uvicorn backend.main:app --reload
```

Start frontend in another terminal:

```bash
streamlit run frontend/app.py
```

Test these complaints:

```text
A child is missing near school since evening.
Someone hacked my bank account and took money.
A fire accident happened near the market.
A woman is being harassed near the bus stop.
Yesterday night near Tambaram, one car hit a bike.
```

Check:

```text
Backend does not crash
Frontend displays result
Complaint is saved
Records are visible
Priority looks reasonable
Follow-up questions match category
```

## Step 13: Update `requirements.txt`

Keep only required packages.

Recommended:

```text
fastapi
uvicorn
streamlit
requests
python-dotenv
langchain
langchain-groq
pydantic
```

Do not add unnecessary libraries.

## Step 14: Add `.env.example`

Do not commit real API keys.

Create:

```text
GROQ_API_KEY=your_groq_api_key_here
```

Actual `.env` should stay local.

## Step 15: Update README

README must include:

```text
Project overview
Features
Tech stack
Folder structure
Setup steps
How to run backend
How to run frontend
Example complaint input
Expected output
```

Run commands:

```bash
pip install -r requirements.txt
```

```bash
uvicorn backend.main:app --reload
```

```bash
streamlit run frontend/app.py
```

## 16. Final Quality Checklist

Before pushing to GitHub, confirm:

```text
All Python files have correct indentation
No file is collapsed into one long line
Backend runs successfully
Swagger works
POST /complaints works
GET /complaints works
Streamlit app works
SQLite saves complaints
No real API key is committed
README has correct instructions
Each file stays below 100 lines
```

## 17. What Not to Add Now

Do not add:

```text
RAG
Vector database
PDF reports
Analytics charts
Admin login
Complaint status tracking
Email alerts
SMS alerts
Local LLM
LangGraph
Image upload
Complex UI styling
```

These are distractions right now.

## 18. Final Implementation Priority

Follow this exact order:

```text
1. Fix file formatting
2. Run syntax check
3. Fix backend files
4. Test FastAPI
5. Fix AI JSON output
6. Fix SQLite save/fetch
7. Connect Streamlit
8. Test complete flow
9. Update README
10. Push clean code
```

## 19. Final Verdict

The project idea is good.

The current priority is code stability, not new features.

Once the basic flow works cleanly, the project is already strong enough for beginner students:

```text
Complaint text
→ AI classification
→ Entity extraction
→ Priority assignment
→ Follow-up questions
→ SQLite save
→ Streamlit records view
```
