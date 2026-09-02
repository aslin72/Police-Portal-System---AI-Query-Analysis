# Chapter 4.6 — The Frontend's Own Shadow of the Backend's Brain

This chapter covers something genuinely worth pausing on: `frontend/src/lib/complaint-intelligence.ts`,
plus the three components built on top of it —
`EmergencyIntakeBanner`, `MissingDetailsDetector`, and `FIRDraftPreview`,
assembled together by `ComplaintFeaturePanel`. This file does something that
might, at first glance, look like duplicated effort: it re-implements a
simplified version of the category detection, priority estimation, and risk
flagging you already fully learned in Chapters 2.5 and 3.2 — entirely in
TypeScript, running entirely inside the citizen's browser, completely
independent of the real backend. Understanding *why* a project would
deliberately duplicate logic like this, and how it keeps the duplicate
honestly labeled as a duplicate, is the real lesson of this chapter.

## The same shape of logic, in a different language, for a different reason

```ts
const CATEGORY_KEYWORDS: Record<string, string[]> = {
  "child safety": ["missing child", "child missing", ...],
  "cyber crime incident": ["online fraud", "cyber crime", "phishing", ...],
  ...
};

function estimateCategory(text: string): { category: string; confidence: "high" | "low" } {
  const lower = text.toLowerCase();
  for (const [category, keywords] of Object.entries(CATEGORY_KEYWORDS)) {
    if (hasAnyKeyword(lower, keywords)) {
      return { category, confidence: "high" };
    }
  }
  return { category: "general issue recorded", confidence: "low" };
}
```

If this looks immediately familiar, it should — it's structurally identical
to `_deterministic_category` from Chapter 3.2's `ai_service.py`: a lookup
table of categories to keyword lists, checked in order, first match wins.
`estimatePriority` and `estimateRiskFlags`, right below it in the same file,
are the same story again, mirroring `_determine_priority` and
`_detect_risk_flags` from Chapter 2.5's `triage.py` closely enough that you
could, at this point, predict most of their behavior without reading them
line by line.

Here's the important question this raises, worth sitting with directly: if
the real backend already does this properly — using an actual AI model plus
a carefully built rules engine you've now studied in real depth — why does
the frontend bother re-implementing a simpler version of the exact same
logic, entirely on its own?

The answer is about **when** each one runs, not which one is "better."
The real backend classification only happens once a message is actually sent
to `chat/complaint` and processed — a real network round trip, taking real
time. This frontend version runs **instantly**, locally, on every single
render, with zero network cost at all, purely off whatever the citizen has
typed so far, even text they haven't sent yet. That speed is what makes
`EmergencyIntakeBanner` possible at all: it needs to notice a citizen
typing something like "he has a knife" and react within the same instant,
not after waiting for a full round trip to the AI. This is a genuinely
useful, real pattern worth remembering well beyond this one project: when a
backend computation is too slow for the instant feedback a good interface
needs, a deliberately simpler, client-side approximation of the same idea
can fill that gap — as long as it is never allowed to be mistaken for the
real, authoritative answer.

## Making sure "estimated" never gets confused with "official"

This is where the discipline really shows, and it's worth tracing exactly
how thoroughly this project protects against that confusion, because it
would be a genuinely serious mistake for an officer to ever be misled by a
number that only ever existed as a rough, local guess.

```ts
export interface DraftData {
  ...
  estimatedCategory: string;
  estimatedPriority: string;
  estimatedRiskFlags: string[];
  estimatedAssignedUnit: string;
  estimatedRecommendedAction: string;
  estimateConfidence: "high" | "low";
  ...
}
```

Every single field this file produces that came from its own guesswork is
named starting with `estimated` — not `category`, `priority`, the plain
names you'd see on a real, backend-confirmed `Complaint` object from Chapter
4.3. This is a naming discipline doing real, load-bearing work: anyone
reading this code later, or reading `FIRDraftPreview`'s JSX, can immediately
tell which fields are a client-side guess and which ones — reporter name,
the text the citizen actually typed — are simply facts being echoed back
unchanged.

```tsx
<Row
  label="Category"
  value={draftData.estimateConfidence === "high" ? draftData.estimatedCategory : "Not enough details to estimate yet"}
  estimated
/>
...
<span className={`... ${PRIORITY_COLORS[draftData.estimatedPriority] || PRIORITY_COLORS.Low}`}>
  {draftData.estimatedPriority} (estimated)
</span>
```

`FIRDraftPreview` — the drawer a citizen can open before actually filing,
introduced in Chapter 4.5 — carries the word "(estimated)" directly into the
visible text next to the priority badge, and its `Row` helper component
accepts an `estimated` prop specifically to render a small, muted
"(estimated)" label next to any value that came from this file rather than
the real backend. And right at the very top of that same drawer:

```tsx
<div className="rounded-lg border border-yellow-500/20 bg-yellow-500/5 p-3 text-xs text-yellow-300/80">
  This is a structured complaint preview generated from your details. It is not legal advice
  and does not constitute a filed FIR. Review the details before submitting.
</div>
```

This disclaimer is doing real, deliberate work, not just covering the
product legally — it's the single clearest statement in this entire
codebase of the exact same lesson Chapter 3.4 closed Part 3 with: **never
let an estimate be mistaken for an authoritative decision.** There, it was
about not trusting the AI blindly. Here, it's the exact same idea, aimed at
the interface itself: don't let a citizen — or, worse, an officer glancing
at a screen quickly — mistake a fast, local guess for the real, filed,
backend-confirmed truth. The moment `handleFileComplaint`, from the last
chapter, actually runs, none of this estimated data is what gets saved —
recall from Chapter 2.4 that filing runs the complaint's real text back
through the actual `analyze_complaint` and `triage_complaint` on the
backend, from scratch, independently of anything this file ever guessed.

## Reacting to danger the instant it's typed

```ts
const EMERGENCY_KEYWORDS = [
  "missing child", "child missing", ..., "weapon", "knife", "gun", ...,
  "suicide", "self-harm", "end my life", "kill myself",
  "immediate danger", "life threatening", "hostage",
  "murder", "death", "dead body", "found dead",
];

export function deriveComplaintInsight(
  collectedFields: Record<string, unknown>,
  messages: Array<{ role: string; content: string }>,
): ComplaintInsight {
  ...
  const emergencyMode = hasAnyKeyword(userText, EMERGENCY_KEYWORDS);
  ...
  const requiredEmergencyFields: MissingField[] = [];
  if (emergencyMode) {
    if (!location) { requiredEmergencyFields.push({ key: "incident_location", ... }); }
    if (!phone) { requiredEmergencyFields.push({ key: "reporter_phone", ... }); }
    if (!hasImmediateDangerStatus(userText)) { requiredEmergencyFields.push({ key: "immediate_danger_status", ... }); }
    ...
  }
  ...
}
```

`deriveComplaintInsight` is the single function every screen you saw in
Chapter 4.5 actually calls, and it's worth noticing one genuinely new
keyword list here that doesn't exist anywhere in the backend at all:
`"suicide", "self-harm", "end my life", "kill myself"` — self-harm risk
isn't a category the backend's `ai_service.py` or `triage.py` ever
mentions. This is a deliberate, frontend-only safety net, and it makes real
sense specifically at this layer: the goal here isn't classifying a
complaint into a police department's routing categories at all — it's
noticing, as fast as technically possible, that whoever is typing right now
might be in genuine, immediate danger, and immediately asking for the two
things that would actually let a real person be reached: a location, and a
phone number. That's a fundamentally different, narrower, and more urgent
job than the backend's classification, which is exactly why it makes sense
that its logic lives here, entirely separately, tuned for instant reaction
rather than accurate routing.

`hasImmediateDangerStatus` is worth a specific look, because it's checking
for the *opposite* signal:

```ts
function hasImmediateDangerStatus(text: string): boolean {
  return hasAnyKeyword(text, [
    "safe now", "not in danger", "no immediate danger", "danger is over",
    "danger has passed", "threat stopped", "attacker left", "attacker ran away",
    "still in danger", "danger is still active", "attacker is still here",
    "fire is still active", "still trapped",
  ]);
}
```

Despite its name, this function isn't asking "is there danger" — it's
asking "has the citizen already told us, one way or the other, whether
they're currently safe." If they've said *either* "I'm safe now" *or*
"he's still here," this function returns `true`, because either answer
means that specific question doesn't need to be asked again — only silence
on the question leaves it in `requiredEmergencyFields`. This is the exact
same discipline you saw the backend's own `questions.py` apply in Chapter
2.6: never ask a citizen something they've effectively already answered.

## Composing three components into one panel

```tsx
export function ComplaintFeaturePanel({ insight }: ComplaintFeaturePanelProps) {
  return (
    <div className="space-y-3">
      {insight.emergencyMode && insight.requiredEmergencyFields.length > 0 && (
        <EmergencyIntakeBanner emergencyReason={insight.emergencyReason} requiredEmergencyFields={insight.requiredEmergencyFields} />
      )}
      <MissingDetailsDetector missingDetails={insight.missingDetails} collectedDetails={insight.collectedDetails} criticalMissing={insight.criticalMissing} />
    </div>
  );
}
```

This last piece is worth calling out purely for how small and plain it is,
after everything leading up to it: `ComplaintFeaturePanel` computes
nothing at all — it's pure composition, taking the one `insight` object
`deriveComplaintInsight` already fully computed back in the chat page, and
handing the right slices of it down to two focused child components, each
rendering only what it needs, each responsible for exactly one job — the
emergency banner for immediate danger, the details detector for general
completeness. This is the same "each piece has one job" discipline you
first learned all the way back in Chapter 1.2's architecture map, now
showing up again, at the smallest possible scale, inside a single UI panel.

## Think about it

1. This file's keyword lists for category and priority estimation are
   similar to, but not identical to, the backend's own lists in
   `triage.py` and `ai_service.py`. What real risk does that create if a
   developer updates one list — say, adding a new emergency keyword to the
   backend — without remembering to update this file too?
2. The self-harm keywords in `EMERGENCY_KEYWORDS` have no equivalent
   anywhere in the backend. If you were asked to make sure a complaint
   mentioning self-harm was also flagged specially once it reaches the
   officer's dashboard, not just instantly in the citizen's own browser,
   what would you need to add, and where?
3. Every estimated value in `DraftData` is prefixed with `estimated` and
   displayed with an "(estimated)" label. Can you think of a way a citizen
   or officer could still, despite all of this, end up mistakenly treating
   one of these estimates as final? What would you change to close that
   gap further?
