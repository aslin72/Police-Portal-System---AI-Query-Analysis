# Lovable Meta Prompt: Police Portal Frontend Redesign

You are building the frontend for an AI-assisted Police Complaint Portal. Create a modern, minimal, professional, attractive frontend that connects to the existing FastAPI backend without changing the backend contract.

## Product Context

This is not a marketing website. It is a serious public-safety operations portal with two user groups:

- Citizens who file complaints, upload evidence, and track complaint status.
- Police/officer users who triage complaints, review AI analysis, update status, add notes, and inspect evidence.

The UI should feel trustworthy, calm, official, and premium. Avoid playful, neon-heavy, crypto-dashboard, or generic SaaS styling. The design should look like a modern government operations system with polished motion and subtle depth.

## Existing Stack

Use the current project structure and technologies:

- Next.js App Router
- React + TypeScript
- Tailwind CSS
- shadcn/ui-style components
- lucide-react icons
- react-hook-form + zod
- react-dropzone
- TanStack Table
- Existing API helper pattern in `frontend/src/lib/types.ts`
- Backend base URL from `NEXT_PUBLIC_API_URL`, defaulting to `http://localhost:8000`

Do not replace the backend. Do not mock core flows if the real API is available. Keep the frontend compatible with the existing FastAPI endpoints.

## Backend API Contract

Use these endpoints exactly:

- `POST /complaints`
  - Purpose: submit complaint text and reporter metadata, then receive AI classification plus rule-based triage.
  - JSON body:
    ```json
    {
      "complaint_text": "string",
      "reporter_name": "string | null",
      "reporter_phone": "string | null",
      "reporter_email": "string | null",
      "incident_location": "string | null",
      "incident_time": "string | null"
    }
    ```

- `GET /complaints`
  - Purpose: officer dashboard list.

- `GET /complaints/{id}`
  - Purpose: citizen tracking and officer detail page.

- `PATCH /complaints/{id}/triage`
  - Purpose: officer updates complaint status and notes.
  - JSON body:
    ```json
    {
      "status": "New | Under Review | Assigned | Resolved | Closed",
      "officer_notes": "string | null"
    }
    ```

- `POST /complaints/{id}/evidence`
  - Purpose: upload one or more evidence files using `multipart/form-data`.
  - Field name must be `files`.
  - Allowed file types: `.jpg`, `.jpeg`, `.png`, `.pdf`, `.txt`.
  - Max size: 10 MB per file.

- `GET /complaints/{id}/evidence`
  - Purpose: list evidence metadata for one complaint.

- `GET /evidence/{id}`
  - Purpose: download evidence file.

## Complaint Data Shape

All complaint UI must handle this shape:

```ts
type Complaint = {
  id: number;
  complaint_text: string;
  category: string;
  location: string;
  incident_time: string;
  persons_involved: string[];
  summary: string;
  priority: "Emergency" | "High" | "Medium" | "Low" | string;
  followup_questions: string[];
  reporter_name: string | null;
  reporter_phone: string | null;
  reporter_email: string | null;
  citizen_incident_location: string | null;
  citizen_incident_time: string | null;
  status: "New" | "Under Review" | "Assigned" | "Resolved" | "Closed" | string;
  assigned_unit: string | null;
  triage_reason: string | null;
  risk_flags: string[];
  recommended_action: string | null;
  officer_notes: string | null;
  created_at: string | null;
  updated_at: string | null;
};
```

AI categories include:

- `child safety`
- `cyber crime incident`
- `women help desk`
- `public healthcare`
- `road accident`
- `murder / serious crime incident`
- `fire accident`
- `general issue recorded`

Assigned units include:

- Child Protection Desk
- Cyber Crime Cell
- Women Help Desk
- Public Health Coordination
- Traffic Police
- Serious Crime Unit
- Fire and Emergency Coordination
- General Desk

Risk flags may include:

- `injury_reported`
- `urgent_medical_attention`
- `weapon_involved`
- `child_involved`
- `fire_risk`
- `person_missing`
- `digital_fraud`
- `none_identified`

## Routes To Build Or Redesign

Preserve these routes:

- `/`
  - Professional portal entry screen, not a generic landing page.
  - Show clear citizen and officer entry actions.
  - Include a premium full-width civic/security visual background, subtle motion, and real product signals in the first viewport.

- `/citizen/file-complaint`
  - Main citizen complaint form.
  - Complaint text is required.
  - Reporter details are optional.
  - Incident location and time are optional.
  - Evidence upload is optional.
  - After submission, show AI classification, priority, assigned unit, summary, risk flags, follow-up questions, complaint ID, and evidence upload result.
  - Add a polished animated assistant/avatar sequence after complaint submission. It should communicate: analyzing report, classifying incident, routing to unit, complaint filed. The animation should be professional and not cartoonish.

- `/citizen/track-complaint`
  - Search by complaint ID.
  - Show current status, priority, category, assigned unit, timestamps, and AI summary.
  - Include a horizontal or vertical status timeline for New -> Under Review -> Assigned -> Resolved -> Closed.

- `/officer/dashboard`
  - Dense, scannable operations dashboard.
  - Include summary metrics: total complaints, emergency/high, assigned, resolved/closed.
  - Include filters for priority and status.
  - Include search.
  - Include sortable table with ID, priority, category, summary, status, assigned unit, created time.
  - Row click should open `/officer/complaints/{id}`.

- `/officer/complaints/[id]`
  - Complaint detail page.
  - Show original complaint, AI analysis, reporter details, citizen-submitted incident location/time, evidence list, follow-up questions, triage reason, risk flags, recommended action, timestamps.
  - Include officer controls to update status and officer notes.
  - Make the triage sidebar highly scannable.

- `/officer/evidence-review`
  - Show all uploaded evidence files grouped or listed with complaint ID, filename, file type, file size, uploaded date, category, and download action.
  - Empty state should look polished.

## Visual Design Direction

Use a modern minimal color palette that is professional and easy to change globally.

Hard requirement: all brand, semantic, status, glow, and surface colors must be defined as CSS variables in one global place, preferably `frontend/src/app/globals.css`. Components must consume semantic Tailwind/CSS variables, not hardcoded hex colors.

Create a token system like:

```css
:root {
  --background: ...;
  --foreground: ...;
  --surface: ...;
  --surface-elevated: ...;
  --primary: ...;
  --primary-foreground: ...;
  --accent: ...;
  --accent-foreground: ...;
  --muted: ...;
  --muted-foreground: ...;
  --border: ...;
  --ring: ...;
  --success: ...;
  --warning: ...;
  --danger: ...;
  --info: ...;
  --priority-emergency: ...;
  --priority-high: ...;
  --priority-medium: ...;
  --priority-low: ...;
  --glow-primary: ...;
  --glow-danger: ...;
  --nav-bg: ...;
}
```

Recommended palette direction:

- Base: cool off-white / very light gray background.
- Text: deep charcoal, not pure black.
- Primary: deep civic navy or graphite-blue.
- Accent: restrained teal or emerald for trust and success.
- Warning: amber.
- Danger/Emergency: controlled red.
- Surfaces: white and slightly elevated neutral panels.

Avoid a one-note palette. Do not make the UI only blue, only purple, or only gray. Use color sparingly and semantically.

## Layout And Components

Create a cohesive component system:

- App shell with responsive navigation.
- Clear citizen/officer navigation separation.
- Reusable page header component.
- Reusable metric cards.
- Reusable priority badges.
- Reusable status badges.
- Reusable empty states.
- Reusable loading skeletons.
- Reusable file upload/dropzone component.
- Reusable result summary panel.
- Reusable timeline/status stepper.
- Reusable evidence file row/card.
- Reusable triage sidebar.

Use icons from `lucide-react`. Do not use text-only buttons where a familiar icon improves scanability. Add accessible labels for icon-only controls.

Keep cards at a restrained radius, around 8px unless the current system requires otherwise. Do not nest cards inside cards. Use full-width sections and clean panels.

## Motion And Effects

Add polished motion, but keep it professional and performant.

Expected effects:

- Smooth page transition on route changes.
- Subtle fade/slide entrance for page sections.
- Hover lift on actionable cards.
- Gentle glow on primary submit buttons and critical action buttons.
- Focus rings that match the global token system.
- Animated status timeline when a complaint is loaded.
- Animated success/result panel after complaint submission.
- Animated upload progress or pending state.
- Skeleton loading states for dashboard, detail pages, and tracking.
- Toast notifications for success/error feedback.

Complaint submission animation:

- After clicking Submit Complaint, show a staged animated analysis experience.
- Stages:
  1. Securely receiving report.
  2. AI extracting category, location, time, and persons involved.
  3. Rule-based triage assigning priority and unit.
  4. Evidence upload, if files were attached.
  5. Complaint filed with generated complaint ID.
- Include a professional assistant/avatar or abstract AI operator animation.
- The avatar can be a minimal shield/operator orb, glassmorphic civic assistant, or Lottie-style character.
- Do not make it childish or cartoonish.
- Respect reduced-motion preferences. If `prefers-reduced-motion` is enabled, show non-animated progress states.

Background/asset requirements:

- Add a subtle product-relevant background asset or visual system: civic map grid, secure network lines, document flow, evidence chain, or police operations dashboard pattern.
- The background must be subtle and must not reduce readability.
- Avoid generic stock photos, decorative blobs, random gradient orbs, and distracting abstract art.
- Use SVG/CSS/canvas only if it supports the product story. Otherwise use generated/static bitmap assets placed in the appropriate public asset directory.

## UX Requirements

Citizen filing:

- Use a calm two-column layout on desktop: form on the left, guidance/secure submission summary on the right.
- On mobile, stack into one column.
- Keep the form fast and clear.
- Show validation inline.
- File dropzone should show accepted file types and max size.
- Show selected evidence files with remove actions before submit.
- Preserve complaint submission even if evidence upload fails, and clearly explain that the complaint was saved.

Officer dashboard:

- Prioritize information density and scanability.
- Emergency and High complaints must visually stand out without making the whole dashboard noisy.
- Include empty, loading, and error states.
- Table should remain usable on mobile via horizontal scroll or responsive row cards.

Complaint detail:

- Use a main content + sticky sidebar layout on desktop.
- Sidebar should contain priority, status, assigned unit, risk flags, recommended action, and update controls.
- Evidence download buttons must be real anchors styled like buttons, not invalid button/link nesting.

Tracking:

- Make the status timeline easy for citizens to understand.
- Do not expose sensitive internal-only fields too aggressively on citizen tracking; keep officer notes and sensitive triage details officer-focused unless already visible in existing behavior.

## Accessibility And Quality Rules

- Use semantic HTML.
- Buttons are for actions. Links are for navigation. Download actions should be anchors.
- Every icon-only button must have an accessible label.
- Maintain keyboard navigation.
- Maintain color contrast.
- Support responsive layouts from mobile to desktop.
- Do not let text overflow buttons, badges, cards, or tables.
- Use `aria-live` for submission progress/result messages where appropriate.
- Respect `prefers-reduced-motion`.
- Avoid hardcoded API URLs inside components; use the existing `NEXT_PUBLIC_API_URL` pattern.
- Avoid hardcoded colors inside components; use global tokens.

## Implementation Constraints

- Keep TypeScript strict enough to avoid `any` for API objects.
- Preserve existing API helper functions or improve them without breaking names used across the app.
- Keep backend field names exactly as returned by FastAPI.
- Do not rename backend routes.
- Do not require authentication unless explicitly asked; this project currently has no auth.
- Do not introduce a complex state manager. Local state and small reusable hooks are enough.
- If adding animation dependencies, prefer lightweight choices. Framer Motion is acceptable if used cleanly and not overused.
- Do not create fake workflows that are not supported by the backend.

## Final Deliverable Expected From Lovable

Produce a complete frontend redesign with:

- Global theme tokens that can be changed from one CSS location.
- Modern app shell/navigation.
- Redesigned home, citizen filing, tracking, officer dashboard, officer complaint detail, and evidence review routes.
- Polished loading, empty, error, and success states.
- Professional animations and page transitions.
- Complaint submission staged animation/avatar.
- File upload UI connected to the real evidence endpoint.
- Dashboard table and filters connected to the real complaints endpoint.
- Officer status/notes update connected to the real triage endpoint.
- Responsive design.
- Accessibility-safe links/buttons.

The final UI should look like a serious, modern, premium public-safety platform: minimal, trustworthy, motion-rich, and operationally useful.
