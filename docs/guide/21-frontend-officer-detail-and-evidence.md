# Chapter 4.8 — Reviewing One Complaint, and All the Evidence

The last two officer screens: `frontend/src/app/officer/complaints/[id]/page.tsx`,
where an officer actually works a single complaint, and
`frontend/src/app/officer/evidence-review/page.tsx`, a cross-complaint view
of every uploaded file. Between them, they close out the citizen-to-officer
loop first sketched all the way back in Chapter 1.2, and introduce one last
genuinely new idea: how a dynamic route actually receives its URL parameter
in a modern Next.js app.

## Reading a dynamic route's parameter

```tsx
export default function ComplaintDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);
  const complaintId = parseInt(id);
  ...
```

Recall from Chapter 4.1 that the folder name `[id]` is what makes this a
dynamic route at all, matching a URL like `/officer/complaints/42`. What's
genuinely worth pausing on here is `params: Promise<{ id: string }>` — in
current versions of Next.js, route parameters arrive wrapped in a
`Promise`, not as a plain, already-available value, and `use(params)`
— a relatively new React function, distinct from a custom hook, callable
directly inside a component body — is what unwraps that promise and hands
back the real value, `{ id: "42" }`, pausing the component's rendering
until it's ready. `parseInt(id)` then converts that URL text into a real
number, the same conversion you've now seen in several places throughout
Part 4, always because a URL is fundamentally text, no matter what it
represents.

## Loading two things that both depend on the same ID

```tsx
useEffect(() => {
  async function load() {
    try {
      const [comp, ev] = await Promise.all([
        getComplaint(complaintId),
        getEvidenceByComplaint(complaintId).catch(() => ({ evidence: [] })),
      ]);
      setComplaint(comp);
      setEvidence(ev.evidence || []);
      if (initialLoad.current) {
        setStatus(comp.status);
        setNotes(comp.officer_notes || "");
        initialLoad.current = false;
      }
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      setError(msg);
    }
    setLoading(false);
  }
  load();
}, [complaintId]);
```

`Promise.all([...])` is worth understanding precisely, because it's a
genuinely useful pattern any time a page needs more than one independent
piece of data at once: rather than `await`ing `getComplaint` and then,
separately, `await`ing `getEvidenceByComplaint` — one after the other,
wasting time waiting for the first to fully finish before even starting the
second — `Promise.all` kicks off both network requests at the same moment,
and waits only until *both* have completed, whichever takes longer. This
directly speeds up how quickly the page can actually show something useful,
since these two lookups have nothing to do with each other and there's no
real reason to make one wait for the other.

Notice `getEvidenceByComplaint(complaintId).catch(() => ({ evidence: [] }))`
— its own inline `.catch(...)`, separate from the outer `try`/`catch`
wrapping the whole `load` function. This is a deliberate, specific choice:
if fetching evidence fails for any reason, this page shouldn't treat that as
a total failure of the whole page — an officer should still be able to see
and work the complaint itself, just with an empty evidence list, rather than
a full error screen over what might be a small, unrelated evidence-lookup
problem. Compare this to `getComplaint(complaintId)` right beside it, which
has no such fallback — if the complaint itself can't be loaded, there's
genuinely nothing useful left for this page to show, so that failure is
allowed to propagate up to the outer `catch` and produce the real error
screen you'll see rendered further down the file.

`initialLoad.current` — a `useRef`, the same tool from Chapter 4.5, but used
here for a different purpose than scrolling: as a plain flag that survives
across re-renders without itself triggering one. This `useEffect` actually
re-runs any time `complaintId` changes (an officer could, in principle,
navigate directly from one complaint's URL to another), but `setStatus` and
`setNotes` should only ever be seeded from the server's data the very first
time this specific complaint loads — otherwise, an officer who had already
started typing new notes could have their in-progress typing silently
overwritten every time this effect happened to re-run. `initialLoad.current`
is what prevents that: `true` on the very first successful load, flipped to
`false` immediately afterward, so every subsequent load skips re-seeding
those two fields.

## Updating a complaint, and being honest about the trade-off

```tsx
async function handleUpdate() {
  setUpdating(true);
  setUpdateMsg(null);
  try {
    const updated = await updateTriage(complaintId, { status: status || null, officer_notes: notes });
    setComplaint(updated);
    setUpdateMsg({ type: "success", text: "Complaint updated successfully." });
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e);
    setUpdateMsg({ type: "error", text: msg });
  }
  setUpdating(false);
}
```

`updateTriage`, from Chapter 4.3's API client, calls the exact `PATCH
/complaints/{id}/triage` endpoint you learned in complete detail back in
Chapter 2.4 — including its `422` validation against the fixed `STATUSES`
set. Notice this function always sends both `status` and `officer_notes`
together, every single time an officer clicks "Update Complaint," even
though the backend endpoint you already know is perfectly capable of
handling either one alone. This is a reasonable, deliberate simplification
for this particular screen: rather than tracking which specific field an
officer actually changed, this page just always sends the current values of
both fields it's already displaying — genuinely simpler frontend code, at
the small cost of occasionally sending a field's unchanged value back to the
backend, which, since the backend re-writes it to the identical value it
already had, causes no real harm at all.

## A cross-complaint view, built entirely from calls you already know

```tsx
useEffect(() => {
  async function load() {
    try {
      const complaints = await getComplaints();
      const results: EvidenceWithComplaint[] = [];
      for (const c of complaints) {
        try {
          const ev = await getEvidenceByComplaint(c.id);
          for (const record of ev.evidence || []) {
            results.push({ ...record, complaintId: c.id, complaintText: c.summary || c.complaint_text, category: c.category });
          }
        } catch {
          // complaint has no evidence
        }
      }
      setAllEvidence(results);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      setError(msg);
    }
    setLoading(false);
  }
  load();
}, []);
```

The evidence-review page has no endpoint of its own on the backend at
all — there's no `GET /evidence` that returns everything across every
complaint. Instead, this page builds that combined view itself, entirely on
the frontend: fetch every complaint with `getComplaints()`, then, for each
one, fetch its evidence with `getEvidenceByComplaint(c.id)` — the exact same
two functions you already know from Chapters 4.3 and 4.7 — and flatten
everything into one single array, `results`, tagging each evidence record
with which complaint it belongs to along the way. `{ ...record, complaintId:
c.id, ... }` uses the spread operator you first met back in Chapter 4.4 to
build a new object containing everything from the original `record`, plus
three extra fields layered on top.

This is genuinely worth pausing on as a real trade-off, not just incidental
implementation detail: this loop makes one network request per complaint,
sequentially, one after another — for a handful of complaints, that's
unnoticeable; for a police department with tens of thousands of filed
complaints, this exact same code would become slow and, eventually,
genuinely impractical. It's a completely reasonable choice for this
project's current scale, and a good, concrete example of something worth
learning to recognize in any codebase: a piece of code that is correct and
perfectly fine today, but carries a real, specific, and identifiable limit
on how far it can scale before it needs to be revisited — in this case,
almost certainly by adding a real backend endpoint that returns all evidence
directly, in one single request, instead.

## Think about it

1. `initialLoad.current` prevents an officer's in-progress notes from being
   overwritten by a re-fetch. Can you think of a different situation on this
   same page where *not* re-syncing with the server could cause its own
   problem — data going stale while the officer keeps looking at an old
   version of the complaint?
2. The evidence-review page's loop silently swallows a failure for any one
   complaint's evidence lookup (`catch { // complaint has no evidence }`)
   and moves on to the next. Is there a difference between "this complaint
   genuinely has no evidence" and "the evidence lookup for this complaint
   failed for some other reason"? Does this page's code currently let you
   tell those two cases apart?
3. You now understand exactly why the evidence-review page makes one
   network request per complaint. If you were asked to fix the scaling
   problem described above by adding a real backend endpoint, sketch, in a
   sentence or two, what that endpoint's URL, method, and response shape
   might reasonably look like, based on everything you learned about this
   project's existing endpoints in Part 2.
