# Frontend Migration Plan: Streamlit to Enterprise Next.js Police Portal

## Summary

  Replace the current Streamlit frontend with a production-shaped Next.js frontend while keeping the existing FastAPI backend.

  The new frontend should feel like an enterprise government operations portal: clean, restrained, data-heavy, professional, and not
  decorative. Use the current backend APIs for complaint filing, triage, evidence upload, and evidence download. Remove Streamlit from
  the deployed UI path.

  Use stock/public imagery carefully: use Indian police/civic visuals from Pexels, Unsplash, or Wikimedia Commons, but do not use
  official Indian national/state police emblems unless permission is confirmed. The safer default is a custom India-facing portal logo
  using a shield/checkmark/tricolor accent, not the official Ashoka emblem.

## Key Changes

### Frontend Stack

  Build the new frontend inside frontend/ as a Next.js app and remove frontend/app.py.

  Use:

- Next.js App Router + TypeScript
- Tailwind CSS
- shadcn/ui for enterprise components
- lucide-react for icons
- TanStack Table for officer complaint tables
- React Hook Form + Zod for complaint form validation
- react-dropzone for evidence upload
- CopilotKit only as an optional assistant panel, not as the core workflow engine

  Remove:

- Streamlit dependency from requirements.txt
- Streamlit run instructions from docs
- Streamlit-specific UI state and request code

  Keep:

- FastAPI backend on port 8000
- OpenRouter AI usage in backend only
- Existing complaint, triage, and evidence logic

### Pages And UX

  Create these frontend routes:

  /
   /citizen/file-complaint
   /citizen/track-complaint
   /officer/dashboard
   /officer/complaints/[id]
   /officer/evidence-review

  Home page:

- Enterprise landing/dashboard entry page, not a marketing hero.
- Show the product name, Indian civic/police visual, and two primary actions:

  - File Complaint
  - Officer Dashboard
- Use a real stock/public image as a restrained banner or side visual.

  Citizen filing page:

- Structured complaint form with:

  - complaint text
  - reporter name
  - phone
  - email
  - incident location
  - incident time
  - evidence upload
- Submit complaint to POST /complaints.
- Upload evidence after complaint creation to POST /complaints/{id}/evidence.
- Show complaint ID, AI category, priority, assigned unit, summary, and follow-up questions.

  Citizen tracking page:

- MVP version accepts complaint ID and fetches matching complaint.
- Add backend endpoint GET /complaints/{complaint_id} so the UI does not rely on filtering all complaints client-side.
- Show status, assigned unit, priority, summary, created date, and updated date.

  Officer dashboard:

- Data table with sorting, filtering, and priority badges.
- Filters:

  - priority
  - status
  - category
  - assigned unit
- Metrics row:

  - New complaints
  - Emergency/High priority
  - Assigned
  - Closed
- Table row click opens /officer/complaints/[id].

  Complaint detail page:

- Full case view with:

  - original complaint
  - AI summary
  - reporter metadata
  - AI extracted location/time/persons
  - citizen-provided location/time
  - priority
  - assigned unit
  - risk flags
  - triage reason
  - recommended action
  - follow-up questions
  - evidence list/download links
  - officer notes
  - status update control
- Use PATCH /complaints/{id}/triage for officer status and notes.

  Evidence review page:

- List uploaded evidence grouped by complaint.
- Show file name, size, uploaded date, complaint ID, and download action.
- Use existing GET /complaints/{id}/evidence and GET /evidence/{evidence_id}.

### Visual Design And Assets

  Design direction:

- Enterprise government operations UI.
- No flashy gradients, no decorative blobs, no oversized startup-style hero.
- Use dense but readable layouts, calm colors, strong table views, and clear hierarchy.

  Color palette:

  --primary: #0B1F3A;
  --primary-2: #123C69;
  --accent: #2F855A;
  --warning: #D97706;
  --danger: #B42318;
  --background: #F6F8FA;
  --surface: #FFFFFF;
  --border: #D0D7DE;
  --text: #111827;
  --muted: #6B7280;

  Logo rule:

- Create a custom portal logo: shield + checkmark + subtle India tricolor accent.
- Do not use the State Emblem of India, Delhi Police logo, UP Police logo, or any state police mark unless permission is explicitly
  available.
- The official emblem has restricted usage; verify permissions before using it in a deployed product. Reference: MHA emblem rules PDF,
  https://www.mha.gov.in/sites/default/files/EmblemRules2007_12022019.pdf

  Photo sourcing plan:

- Use Pexels Indian Police search for free stock-style Indian police/community photos: https://www.pexels.com/search/indian%20police/
- Use Unsplash Indian Police search for broader civic/India visuals: https://unsplash.com/s/photos/indian-police
- Use Wikimedia Commons Police of India categories for public-domain/Creative Commons media, checking each file license before use:
  https://commons.wikimedia.org/wiki/Category:Police_of_India
- Preferred images:

  - police officer helping citizen
  - officer at desk/computer
  - traffic police or patrol visual
  - community policing scene
  - abstract evidence/cyber investigation visual
- Store selected assets under frontend/public/assets/.
- Add an ASSET_CREDITS.md file listing source URL, author, license, and usage notes for every external image.

### API And Integration

  Required backend additions:

- Add GET /complaints/{complaint_id} returning one complaint.
- Confirm CORS supports:

  - http://localhost:3000
  - deployed frontend URL
- Keep existing endpoints:

  - POST /complaints
  - GET /complaints
  - PATCH /complaints/{id}/triage
  - POST /complaints/{id}/evidence
  - GET /complaints/{id}/evidence
  - GET /evidence/{evidence_id}

  Frontend API client:

- Create one typed API layer for all FastAPI calls.
- Use NEXT_PUBLIC_API_URL, defaulting to http://localhost:8000.
- Keep OpenRouter keys only in backend .env; never expose them to Next.js.

  CopilotKit usage:

- Add only after the core pages work.
- Use CopilotSidebar or CopilotPopup as a filing/help assistant.
- Assistant responsibilities:

  - explain complaint filing steps
  - suggest what evidence to upload
  - summarize visible complaint details
  - help officers understand triage reason
- Do not let CopilotKit directly change complaint status or submit police actions in MVP.

## Test Plan

  Frontend checks:

- npm run lint
- npm run build
- Verify all routes load.
- Verify mobile and desktop layouts.
- Verify no text overflow in tables, cards, buttons, or forms.
- Verify evidence upload UI rejects unsupported files before API call.
- Verify API error states display professional messages.

  Backend integration checks:

- File complaint from Next.js citizen page.
- Upload multiple evidence files.
- View complaint in officer dashboard.
- Open complaint detail page.
- Download evidence.
- Update status and officer notes.
- Track complaint by ID.

  Regression checks:

- FastAPI still starts with:
- Next.js starts with:

  cd frontend
  npm run dev
- Production frontend build succeeds:

  cd frontend
  npm run build

## Assumptions

- The frontend will be fully migrated away from Streamlit.
- Backend remains FastAPI.
- Database can remain SQLite for now.
- Authentication is not part of this frontend migration phase.
- Official Indian/state police logos are not used without permission.
- The first release uses static selected assets from Pexels, Unsplash, or Wikimedia Commons with credits tracked in the repo.
