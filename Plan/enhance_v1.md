
# Police Portal Real-World Feature Implementation Plan

## Summary

  Upgrade the current beginner-friendly FastAPI + Streamlit + SQLite project into a stronger real-world police complaint intake MVP.

  Implement four features:

- Citizen Complaint Filing with structured reporter and incident fields.
- Smart Triage for Police using explainable rule-based logic, not AI-only decisions.
- Evidence Upload using local file storage plus SQLite metadata.
- Dynamic Follow-Up Question Engine based on category and missing complaint details.

  Keep the existing stack: Streamlit frontend, FastAPI backend, SQLite database, LangChain/OpenRouter AI service. Do not add
  authentication, cloud storage, vector databases, or deployment infrastructure in this phase.

  Target plan file path: Plan/police-portal-realworld-features-plan.md.

## Key Changes

### 1. Citizen Complaint Filing

  Enhance the complaint submission flow while preserving the current POST /complaints behavior.

  Frontend changes:

- Replace the single complaint text box with a “Citizen Filing” tab.
- Collect:

  - complaint_text required
  - reporter_name optional
  - reporter_phone optional
  - reporter_email optional
  - incident_location optional
  - incident_time optional
  - evidence files optional
- Submit the complaint first, then upload evidence files using the returned complaint ID.
- Show the citizen:

  - complaint ID
  - AI category
  - priority
  - extracted summary
  - follow-up questions
  - evidence upload success/failure status

  Backend changes:

- Extend request/response schemas to support reporter and incident metadata.
- Store original citizen-provided fields separately from AI-extracted fields.
- Preserve the current AI extraction flow in backend/ai_service.py.

  Database changes:

- Add complaint columns:
  - reporter_name
  - reporter_phone
  - reporter_email
  - citizen_incident_location
  - citizen_incident_time
  - status
  - assigned_unit
  - triage_reason
  - risk_flags
  - recommended_action
  - officer_notes
  - updated_at

  Migration approach:

- Keep SQLite.
- Add a lightweight migration function that checks existing columns with PRAGMA table_info.
- Use ALTER TABLE only for missing columns.
- Preserve existing complaints.db records.

### 2. Smart Triage for Police

  Create a new backend triage layer that produces explainable police-facing recommendations.

  Add backend/triage.py.

  Triage output:

  {
    "priority": "High",
    "assigned_unit": "Traffic Police",
    "risk_flags": ["injury_reported", "urgent_medical_attention"],
    "recommended_action": "Review immediately and contact emergency response if not already handled.",
    "triage_reason": "Road accident category with injury keyword detected."
  }

  Priority values:

- Emergency
- High
- Medium
- Low

  Rules:

- Emergency:

  - child missing
  - active fire
  - murder/death/fatal keywords
  - weapon/threat to life
- High:

  - child safety
  - fire accident
  - serious crime
  - road accident with injury/bleeding
  - women safety with immediate danger keywords
- Medium:

  - cyber crime incident
  - public healthcare
  - road accident without injury keywords
  - women help desk without immediate danger
- Low:

  - general issue recorded
  - incomplete or unclear complaint without risk keywords

  Assigned unit mapping:

- child safety → Child Protection Desk
- cyber crime incident → Cyber Crime Cell
- women help desk → Women Help Desk
- public healthcare → Public Health Coordination
- road accident → Traffic Police
- murder / serious crime incident → Serious Crime Unit
- fire accident → Fire and Emergency Coordination
- general issue recorded → General Desk

  Officer UI:

- Add an “Officer Triage” tab in Streamlit.
- Show complaints sorted by priority and newest first.
- Add filters for:

  - status
  - priority
  - category
  - assigned unit
- For each complaint, show:

  - complaint text
  - AI summary
  - extracted fields
  - citizen-provided fields
  - evidence list
  - triage reason
  - risk flags
  - recommended action
  - follow-up questions
- Allow officer to update:

  - status
  - officer_notes

  API additions:

- PATCH /complaints/{complaint_id}/triage
  - Request:

    {
    "status": "Under Review",
    "officer_notes": "Reviewed by traffic desk."
    }
  - Allowed statuses:

    - New
    - Under Review
    - Assigned
    - Resolved
    - Closed

### 3. Evidence Upload

  Use local filesystem storage for uploaded evidence and SQLite for metadata.

  Storage layout:

  uploads/
    evidence/
      complaint_1/
        generated_safe_filename.jpg
        generated_safe_filename.pdf

  Add uploads/ to .gitignore.

  Create a new SQLite table:

  CREATE TABLE IF NOT EXISTS evidence (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      complaint_id INTEGER NOT NULL,
      original_filename TEXT NOT NULL,
      stored_filename TEXT NOT NULL,
      file_path TEXT NOT NULL,
      content_type TEXT,
      file_size INTEGER,
      uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      FOREIGN KEY (complaint_id) REFERENCES complaints(id)
  )

  Backend behavior:

- Add POST /complaints/{complaint_id}/evidence.
- Accept multiple files.
- Create the complaint evidence folder if missing.
- Generate safe unique filenames.
- Save file metadata in SQLite.
- Reject files larger than 10 MB.
- Allow file types:

  - images: .jpg, .jpeg, .png
  - documents: .pdf
  - text: .txt
- Return uploaded evidence metadata.

  Add:

- GET /complaints/{complaint_id}/evidence
- GET /evidence/{evidence_id} for downloading/viewing evidence files.

  Frontend behavior:

- Citizen tab allows optional file upload during filing.
- Officer tab shows evidence metadata and download links.
- If complaint creation succeeds but evidence upload fails, keep the complaint and show an evidence-specific error.

### 4. Dynamic Follow-Up Question Engine

  Upgrade backend/questions.py from static category questions to a dynamic engine.

  Function signature:

  def get_followup_questions(category, ai_result, complaint_text, evidence_count=0):
      ...

  Question sources:

- Missing required details:

  - missing location
  - missing incident time
  - missing persons involved
  - missing reporter contact
- Category-specific templates.
- Evidence prompts for categories where proof is useful:

  - cyber crime
  - road accident
  - fire accident
  - serious crime
  - women help desk
- Urgency prompts when high-risk keywords are found.

  Rules:

- De-duplicate similar questions.
- Return maximum 7 questions.
- Put urgent/safety questions first.
- Keep existing category-specific questions as the base templates.

  Examples:

  For cyber crime:

  What platform, account, or phone number was involved?
  Was any money lost? If yes, how much?
  Can you upload screenshots, transaction receipts, or chat records?

  For missing child:

  What is the child's name and age?
  When and where was the child last seen?
  Is the child currently in immediate danger?
  Can you upload a recent photo?

## API And Data Flow

  Citizen complaint flow:

  Streamlit Citizen tab
  → POST /complaints
  → AI analysis
  → dynamic follow-up questions
  → smart triage
  → save complaint
  → optional POST /complaints/{id}/evidence
  → show complaint ID and results

  Officer triage flow:

  Streamlit Officer tab
  → GET /complaints
  → filter/sort records
  → GET /complaints/{id}/evidence
  → PATCH /complaints/{id}/triage
  → refresh dashboard

  Keep backward compatibility:

- Existing POST /complaints with only complaint_text must still work.
- Existing GET /complaints must still return records, now with extra fields.

## Test Plan

  Manual backend tests:

- Submit plain complaint without evidence.
- Submit complaint with reporter metadata.
- Submit road accident with injury and confirm priority becomes High.
- Submit missing child complaint and confirm priority becomes Emergency or High.
- Submit cyber fraud complaint and confirm assigned unit is Cyber Crime Cell.
- Upload valid .jpg, .png, .pdf, and .txt evidence files.
- Reject unsupported file type such as .exe.
- Reject file larger than 10 MB.
- Fetch evidence metadata by complaint ID.
- Download evidence by evidence ID.
- Update complaint status and officer notes.

  Manual frontend tests:

- Citizen tab submits complaint successfully.
- Citizen tab uploads multiple evidence files.
- Citizen tab handles backend unavailable state.
- Officer tab lists complaints newest first.
- Officer filters work for priority, status, category, and unit.
- Officer can update status and notes.
- Evidence links appear for complaints with uploads.

  Regression tests:

- Existing startup commands still work:

  uvicorn backend.main:app --reload
  streamlit run frontend/app.py
- Existing .env OpenRouter loading still works.
- Existing database records do not break after migration.
- persons_involved, followup_questions, and risk_flags remain JSON-encoded safely in SQLite.

## Assumptions

- Implementation stays beginner-friendly and uses the existing FastAPI + Streamlit + SQLite architecture.
- Evidence files are stored locally under uploads/, not in SQLite blobs or cloud storage.
- The first officer workflow is a Streamlit “Officer Triage” tab, not a separate authenticated police portal.
- Authentication, citizen login, SMS/email notifications, cloud deployment, and full case tracking are out of scope for this phase.
- AI helps classify and extract complaint details, but priority and triage remain explainable rule-based backend logic.

─ Worked for 3m 37s ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

› write the plan.md file in this folder" Plan"

• I can’t write files while Plan Mode is active, but this is the exact complete content to place in Plan/plan.md.

• Proposed Plan

# Police Portal Real-World Feature Implementation Plan

## Summary

  Upgrade the current FastAPI + Streamlit + SQLite police complaint assistant into a real-world MVP for complaint intake and police
  triage.

  Features to implement:

- Citizen Complaint Filing
- Smart Triage for Police
- Evidence Upload
- Dynamic Follow-Up Question Engine

  Keep the current stack simple:

- Frontend: Streamlit
- Backend: FastAPI
- Database: SQLite
- AI: Existing OpenRouter/LangChain flow
- Evidence storage: local uploads/ folder with metadata in SQLite

  Authentication, cloud storage, SMS/email alerts, and production deployment are out of scope for this phase.

## Key Implementation Changes

### Citizen Complaint Filing

  Enhance the current complaint form into a structured citizen filing flow.

  Frontend:

- Add a Citizen Complaint Filing section or tab in frontend/app.py.
- Collect:

  - complaint text, required
  - reporter name, optional
  - reporter phone, optional
  - reporter email, optional
  - incident location, optional
  - incident time, optional
  - evidence files, optional
- Submit complaint data to POST /complaints.
- After complaint creation, upload files to POST /complaints/{complaint_id}/evidence.
- Display complaint ID, category, priority, summary, follow-up questions, and evidence upload result.

  Backend:

- Extend ComplaintRequest and ComplaintResponse in backend/schemas.py.
- Store citizen-provided location/time separately from AI-extracted location/time.
- Keep AI analysis inside backend/ai_service.py.

  Database:

- Add missing complaint fields through a lightweight SQLite migration:
  - reporter_name
  - reporter_phone
  - reporter_email
  - citizen_incident_location
  - citizen_incident_time
  - status
  - assigned_unit
  - triage_reason
  - risk_flags
  - recommended_action
  - officer_notes
  - updated_at

  Default status:

  New

### Smart Triage for Police

  Add explainable rule-based triage. AI should classify and extract details, but police priority must remain deterministic and
  explainable.

  Create backend/triage.py.

  Triage should return:

- priority
- assigned unit
- risk flags
- recommended action
- triage reason

  Priority levels:

  Emergency
  High
  Medium
  Low

  Assigned unit mapping:

  child safety → Child Protection Desk
  cyber crime incident → Cyber Crime Cell
  women help desk → Women Help Desk
  public healthcare → Public Health Coordination
  road accident → Traffic Police
  murder / serious crime incident → Serious Crime Unit
  fire accident → Fire and Emergency Coordination
  general issue recorded → General Desk

  Example triage result:

  {
    "priority": "High",
    "assigned_unit": "Traffic Police",
    "risk_flags": ["injury_reported"],
    "recommended_action": "Review immediately and contact emergency response if needed.",
    "triage_reason": "Road accident category with injury keyword detected."
  }

  Frontend:

- Add an Officer Triage tab in Streamlit.
- Show complaints sorted by priority and newest first.
- Add filters for status, priority, category, and assigned unit.
- Show triage reason, risk flags, recommended action, evidence list, and follow-up questions.
- Allow officer to update status and notes.

  Backend:

- Add PATCH /complaints/{complaint_id}/triage.
- Allow updating:
  - status
  - officer_notes

  Allowed statuses:

  New
  Under Review
  Assigned
  Resolved
  Closed

### Evidence Upload

  Use local filesystem storage for evidence files.

  Add folder:

  uploads/evidence/

  Add to .gitignore:

  uploads/

  Storage pattern:

  uploads/evidence/complaint_1/file_name.pdf
  uploads/evidence/complaint_1/file_name.jpg

  Create SQLite table:

  CREATE TABLE IF NOT EXISTS evidence (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      complaint_id INTEGER NOT NULL,
      original_filename TEXT NOT NULL,
      stored_filename TEXT NOT NULL,
      file_path TEXT NOT NULL,
      content_type TEXT,
      file_size INTEGER,
      uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      FOREIGN KEY (complaint_id) REFERENCES complaints(id)
  );

  Backend endpoints:

  POST /complaints/{complaint_id}/evidence
  GET /complaints/{complaint_id}/evidence
  GET /evidence/{evidence_id}

  Rules:

- Max file size: 10 MB
- Allowed extensions:

  - .jpg
  - .jpeg
  - .png
  - .pdf
  - .txt
- Reject unsupported file types.
- Generate safe unique stored filenames.
- Keep original filename in metadata.

### Dynamic Follow-Up Question Engine

  Upgrade backend/questions.py from static questions to dynamic questions.

  New function:

  def get_followup_questions(category, ai_result, complaint_text, evidence_count=0):
      ...

  The engine should generate questions from:

- category-specific templates
- missing location
- missing incident time
- missing persons involved
- missing reporter contact
- evidence needs
- urgent safety indicators

  Rules:

- Return maximum 7 questions.
- Put urgent/safety questions first.
- Remove duplicate questions.
- Keep questions simple and citizen-friendly.

  Examples:

  Cyber crime:

  What platform, account, or phone number was involved?
  Was any money lost? If yes, how much?
  Can you upload screenshots, transaction receipts, or chat records?

  Missing child:

  What is the child's name and age?
  When and where was the child last seen?
  Can you upload a recent photo?
  Is the child currently in immediate danger?

  Road accident:

  Were there any injuries or fatalities?
  What vehicles were involved?
  Can you upload photos of the accident location or vehicle damage?

## API Flow

  Citizen filing flow:

  Streamlit Citizen tab
  → POST /complaints
  → AI classification and extraction
  → Dynamic follow-up questions
  → Smart triage
  → Save complaint
  → Optional evidence upload
  → Show complaint result

  Officer triage flow:

  Streamlit Officer tab
  → GET /complaints
  → Filter and review complaints
  → GET /complaints/{id}/evidence
  → PATCH /complaints/{id}/triage
  → Refresh dashboard

  Keep backward compatibility:

- Existing POST /complaints with only complaint_text must still work.
- Existing GET /complaints must continue returning complaints, now with extra fields.

## Test Plan

  Backend tests/manual checks:

- Submit complaint with only complaint_text.
- Submit complaint with reporter metadata.
- Submit road accident complaint with injury and confirm priority is High.
- Submit missing child complaint and confirm priority is Emergency or High.
- Submit cyber fraud complaint and confirm assigned unit is Cyber Crime Cell.
- Upload valid .jpg, .png, .pdf, and .txt evidence.
- Reject .exe or unsupported file type.
- Reject file larger than 10 MB.
- Fetch evidence list by complaint ID.
- Download evidence by evidence ID.
- Update complaint status and officer notes.

  Frontend tests/manual checks:

- Citizen form submits successfully.
- Citizen can upload multiple evidence files.
- Citizen sees complaint ID and follow-up questions.
- Officer tab lists complaints.
- Officer filters work.
- Officer can update status and notes.
- Evidence links appear in officer view.

  Regression checks:

  python -m py_compile backend/main.py backend/routes.py backend/schemas.py backend/ai_service.py backend/database.py backend/questions.py
  backend/priority.py frontend/app.py
  uvicorn backend.main:app --reload
  streamlit run frontend/app.py

## Assumptions

- This phase remains beginner-friendly and does not introduce authentication.
- Evidence files are stored locally, not in database blobs or cloud storage.
- Officer triage is implemented as a Streamlit tab, not a separate app.
- AI is used only for classification and extraction.
- Priority and police routing are rule-based for explainability.
- Existing database records must remain usable after migration.
