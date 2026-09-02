# Chapter 4.4 — The Complaint Form: State, Validation, and File Uploads

`frontend/src/app/citizen/file-complaint/page.tsx` is a citizen's first real
option for filing a complaint, from Chapter 1.1 — a structured form, for
someone who already knows what they want to say. This chapter is where
`useState`, mentioned back in Chapter 4.1, finally gets a full, real
treatment, alongside two genuinely important frontend ideas: form validation
and drag-and-drop file uploads.

## Holding onto values that change

```tsx
export default function FileComplaintPage() {
  const [result, setResult] = useState<Complaint | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [evidenceFiles, setEvidenceFiles] = useState<File[]>([]);
  const [evidenceErrors, setEvidenceErrors] = useState<string[]>([]);
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState<string | null>(null);
```

Read `useState<Complaint | null>(null)` precisely: `useState` returns a pair
— the current value (`result`), and a function to change it (`setResult`) —
and the value you pass in, `null`, is that state's starting value. The
`<Complaint | null>` part is TypeScript telling React exactly what type this
particular piece of state is allowed to hold, so the compiler can catch a
mistake — trying to read `result.category` before checking `result` isn't
`null`, say — before the code ever runs.

This is worth understanding as more than boilerplate: every one of these
seven separate pieces of state tracks one specific fact this component needs
to remember between renders. `result` holds the complaint once it's
successfully filed. `loading` tracks whether a submission is currently in
flight, so the submit button can show "Submitting..." and disable itself.
`error` holds a message if something went wrong. `evidenceFiles` holds
whatever files the citizen has attached so far. Each one is independent,
and — this is the part that takes getting used to — updating any one of
them, by calling its setter function, causes React to re-render this
component with the new value, automatically, exactly as promised back in
Chapter 4.1.

## Validating a form before it ever reaches the backend

```tsx
const formSchema = z.object({
  complaint_text: z.string().min(10, "Please provide at least 10 characters"),
  reporter_name: z.string().optional(),
  reporter_phone: z.string().optional(),
  reporter_email: z.string().email("Invalid email").optional().or(z.literal("")),
  incident_location: z.string().optional(),
  incident_time: z.string().optional(),
});

type FormValues = z.infer<typeof formSchema>;
```

This should feel genuinely familiar — it's the frontend's own version of
Pydantic's schema validation from Chapter 2.2, using a library called
**Zod** instead. `z.object({...})` describes the exact shape a valid form
submission must have: `complaint_text` must be a string of at least 10
characters (echoing the same "don't accept an empty or trivial complaint"
instinct you saw enforced server-side by `min_length=1` in Chapter 2.2 and
independently by `_check_ready_to_file`'s ten-character floor in Chapter
3.3), `reporter_email` must either be a genuinely valid email address or an
empty string, and everything else is optional. `type FormValues =
z.infer<typeof formSchema>` is a small piece of real TypeScript power worth
noting: rather than writing out a separate `FormValues` interface by hand
and risking it drifting out of sync with the validation rules, this line
derives the TypeScript type directly from the Zod schema itself — one single
definition, two uses, guaranteed to never disagree with each other.

```tsx
const form = useForm<FormValues>({
  resolver: zodResolver(formSchema),
  defaultValues: { complaint_text: "", reporter_name: "", ... },
});
```

`useForm`, from a library called **react-hook-form**, manages this entire
form's state and validation for you — tracking every field's current value,
which fields have errors, and re-validating as the citizen types —
so this component doesn't need seven more individual `useState` calls, one
per form field. `resolver: zodResolver(formSchema)` is the bridge connecting
the two libraries: it tells react-hook-form to use the Zod schema above as
its validation logic.

Why validate on the frontend at all, when Chapter 2.2 already showed you
that the backend validates everything with Pydantic regardless? This is
worth answering directly, because it's a real, common question: frontend
validation exists purely for the citizen's immediate experience — catching
"you forgot to write anything" or "that's not a valid email" instantly, as
they type, without a round trip to the server and back. It is never a
substitute for backend validation, only a convenience layered on top of it —
the backend's validation is the one that actually protects the system,
because, as you learned in Chapter 2.2, nothing arriving from outside the
server can ever be fully trusted, no matter how careful the frontend was.

## Handling dropped files

```tsx
const onDrop = (accepted: File[], rejected: { file: File; errors: readonly { message: string }[] }[]) => {
  const newFiles: File[] = [];
  const errors: string[] = [];
  for (const f of accepted) {
    if (!isAllowedFileType(f)) {
      errors.push(`${f.name}: Unsupported file type`);
    } else if (!isAllowedFileSize(f)) {
      errors.push(`${f.name}: Exceeds 10 MB limit`);
    } else {
      newFiles.push(f);
    }
  }
  ...
  setEvidenceFiles((prev) => [...prev, ...newFiles]);
  setEvidenceErrors((prev) => [...prev, ...errors]);
};

const { getRootProps, getInputProps, isDragActive } = useDropzone({
  onDrop,
  accept: { "image/jpeg": [".jpg", ".jpeg"], "image/png": [".png"], "application/pdf": [".pdf"], "text/plain": [".txt"] },
  maxSize: 10 * 1024 * 1024,
});
```

`useDropzone`, from `react-dropzone`, handles all the low-level browser
machinery of a drag-and-drop file zone, and hands this component a callback,
`onDrop`, whenever files are dropped or selected. Look closely at
`isAllowedFileType` and `isAllowedFileSize`, imported from `lib/types.ts` —
this is a genuinely important detail to catch: **the exact same rules from
Chapter 2.4's evidence-upload endpoint — allowed extensions, a 10 MB cap —
are checked here too, on the frontend, before a file ever gets uploaded at
all.** This isn't duplication for its own sake; it's the same principle from
the section above, applied to files instead of form fields: reject an
obviously oversized or wrong-type file immediately, in the browser, so the
citizen doesn't have to wait through an upload only to have it rejected by
the server after the fact — while the backend's own check in Chapter 2.4
remains the one that actually enforces the rule, since nothing stops a
different, less careful piece of code from sending a request straight to the
backend without going through this form at all.

`setEvidenceFiles((prev) => [...prev, ...newFiles])` introduces a pattern
worth understanding precisely: passing a function to a state setter, instead
of a plain value. `prev` is the state's current value at the exact moment
this update actually runs; `[...prev, ...newFiles]` uses the spread
operator (`...`) to build a brand new array containing everything already in
`prev`, plus everything in `newFiles`, appended after it. Using the
function form here, rather than writing `setEvidenceFiles([...evidenceFiles,
...newFiles])` directly, protects against a subtle real bug: if two file
drops happened in quick succession, reading `evidenceFiles` directly could
use a stale, out-of-date snapshot of the array; asking React to hand you the
definitely-current value via `prev` avoids that entirely.

## Submitting, in the right order

```tsx
async function onSubmit(values: FormValues) {
  setLoading(true);
  setError(null);
  setResult(null);
  setUploadResult(null);

  try {
    const payload: Record<string, string | null> = { complaint_text: values.complaint_text };
    if (values.reporter_name) payload.reporter_name = values.reporter_name;
    ...

    const complaint = await submitComplaint(payload as unknown as ...);
    setResult(complaint);

    if (evidenceFiles.length > 0) {
      setUploading(true);
      try {
        const ev = await uploadEvidence(complaint.id, evidenceFiles);
        setUploadResult(`Successfully uploaded ${ev.uploaded.length} file(s).`);
      } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : String(e);
        setUploadResult(`Evidence upload failed: ${msg}. Complaint saved.`);
      }
      setUploading(false);
    }
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e);
    setError(msg || "Failed to submit complaint");
  }
  setLoading(false);
}
```

This function is called by react-hook-form only after the Zod validation
from earlier in this chapter has already passed — `values` here is
guaranteed to already match `FormValues`. Notice the deliberate, careful
building of `payload`: rather than sending every form field as-is, each
optional field is only added if it actually has a value — `if
(values.reporter_name) payload.reporter_name = values.reporter_name` — so an
empty string typed and then deleted doesn't get sent to the backend as a
real value, keeping the request as close as possible to what a citizen
actually intended to share.

The two network calls here — `submitComplaint` and, conditionally,
`uploadEvidence` — happen in a deliberate sequence, not at the same time,
and it's worth understanding exactly why. `uploadEvidence(complaint.id,
...)` genuinely needs `complaint.id`, which doesn't exist until
`submitComplaint` has already succeeded and the backend has assigned it — so
these two calls cannot happen simultaneously, only one after the other. And
notice the *inner* `try`/`catch`, wrapped specifically around the evidence
upload: if the complaint itself was saved successfully but the evidence
upload then fails for any reason, this code deliberately does not treat that
as one big failure — it sets a clear, honest message,
`"Evidence upload failed: ... Complaint saved."`, while leaving `result`
populated with the complaint that genuinely was saved. This mirrors,
on the frontend, exactly the same value from Chapter 2.4's backend evidence
endpoint: never let a failure in one part of a multi-step process silently
destroy or hide the part that actually did succeed.

## Reading the result back out

```tsx
{result.risk_flags && result.risk_flags.length > 0 && result.risk_flags[0] !== "none_identified" && (
  <div>
    ...
    {result.risk_flags.map((flag) => (
      <Badge key={flag} variant="outline" className="text-xs">
        {flag.replace(/_/g, " ")}
      </Badge>
    ))}
  </div>
)}
```

This is worth reading closely as a small, concrete piece of frontend/backend
literacy, tying directly back to Part 2. `risk_flags[0] !== "none_identified"`
only makes sense to write at all because you already know, from Chapter
2.5's close reading of `_detect_risk_flags`, that this exact placeholder
string is guaranteed to be the sole entry whenever nothing else was
detected. The frontend engineer who wrote this line was relying on a
guarantee made deep inside the backend's Python, three files away, and
having read both sides yourself now, you can verify that reliance is
actually justified. `flag.replace(/_/g, " ")` is a small regular expression
— you already learned this exact syntax's building blocks in Chapter
3.3 — replacing every underscore in a flag like `"weapon_involved"` with a
space, purely so it displays to a human as "weapon involved" instead of with
the raw, code-friendly underscore still showing.

## Think about it

1. This form validates with Zod on the frontend and, separately, Pydantic on
   the backend — two different schemas, in two different languages, that
   have to be kept in agreement by hand. What would you need to double-check
   if you were asked to raise the minimum complaint length from 10 characters
   to 25?
2. The `onSubmit` function's inner `try`/`catch` around the evidence upload
   means a citizen can end up with a filed complaint and a failed evidence
   upload in the same submission. What should the citizen be able to do in
   that situation, and does the current page give them any way to do it?
3. `useDropzone`'s `accept` option and this project's `isAllowedFileType`
   function both restrict which file types are allowed, using two
   independently written lists. What's the risk of these two lists quietly
   drifting apart over time, and how would you notice if they had?
