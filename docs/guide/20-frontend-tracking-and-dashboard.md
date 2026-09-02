# Chapter 4.7 — Tracking a Complaint, and the Officer's Real Dashboard

This chapter closes the citizen's side of the story and opens the officer's.
`frontend/src/app/citizen/track-complaint/page.tsx` is short and, by now,
should read almost entirely on its own — a good checkpoint for how far
you've come. `frontend/src/app/officer/dashboard/page.tsx` is the real
counterpart to Chapter 4.2's fake landing-page preview: the actual working
queue an officer uses every day, built on real data and a genuine data-table
library.

## Tracking: a small, complete round trip

```tsx
async function handleTrack() {
  const id = parseInt(idInput.trim());
  if (isNaN(id) || id < 1) {
    setError("Please enter a valid complaint ID");
    return;
  }
  setLoading(true);
  setError(null);
  setComplaint(null);
  try {
    const result = await getComplaint(id);
    setComplaint(result);
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e);
    setError(msg || "Complaint not found");
  }
  setLoading(false);
}
```

`parseInt(idInput.trim())` converts the citizen's typed text into a real
number; `isNaN(id)` catches the case where what they typed wasn't a valid
number at all — this is frontend input validation again, the same instinct
from Chapter 4.4, catching an obviously bad ID before ever bothering the
network. `getComplaint(id)`, from Chapter 4.3's API client, does the actual
work — and recall from Chapter 2.4 that a nonexistent ID comes back from the
backend as a `404`, which `getComplaint` turns into a thrown `Error`, caught
right here and shown to the citizen as a plain, honest "Complaint not
found."

```tsx
const STATUS_FLOW = ["New", "Under Review", "Assigned", "Resolved", "Closed"];
...
{STATUS_FLOW.map((s, i) => {
  const currentIdx = STATUS_FLOW.indexOf(complaint.status);
  const done = i <= currentIdx;
  return (
    <Badge variant={done ? "default" : "outline"} className={done ? "bg-primary text-primary-foreground" : ""}>
      {s}
    </Badge>
  );
})}
```

This little progress bar is worth a second look: `STATUS_FLOW` is the exact
same five-stage lifecycle you first learned as a plain Python `set` in
`routes.py`'s `STATUSES`, back in Chapter 2.4 — here it's an ordered array
instead, deliberately, because unlike the backend's use (which only needed
to check membership), this one needs a real, fixed sequence, so it can
compute `currentIdx` and mark every stage up to and including the
complaint's actual current status as "done." This is a small, neat
illustration of something worth remembering generally: the *same*
underlying fact — the five statuses a complaint can hold — gets represented
differently in different places, each shape chosen to fit what that
particular piece of code actually needs to do with it.

## The dashboard: real data, a real table library

```tsx
useEffect(() => {
  getComplaints()
    .then(setComplaints)
    .catch((e) => setError(e.message))
    .finally(() => setLoading(false));
}, []);
```

Contrast this immediately with Chapter 4.2's landing page: this is the real
thing. One `useEffect`, running once on load, calling the real
`getComplaints()` from Chapter 4.3, which hits the real
`GET /complaints` endpoint from Chapter 2.4, which reads real rows out of
the real SQLite database from Chapter 2.3. `.then(setComplaints)` is a
slightly different but equivalent way of writing `await` — chaining what to
do once the `Promise` resolves — and `.finally(() => setLoading(false))`
plays the exact same role as the `finally` block you learned in Chapter 4.5:
runs regardless of success or failure, so the loading skeleton always
eventually disappears.

```tsx
const columns: ColumnDef<Complaint>[] = [
  { accessorKey: "id", header: "ID", size: 60, cell: ({ getValue }) => <span className="font-mono text-xs">#{getValue<number>()}</span> },
  { accessorKey: "priority", header: "Priority", cell: ({ getValue }) => <Badge {...getPriorityBadgeProps(getValue<string>())}>{getValue<string>()}</Badge> },
  ...
];

const table = useReactTable({
  data: filteredData,
  columns,
  getCoreRowModel: getCoreRowModel(),
  getSortedRowModel: getSortedRowModel(),
  getFilteredRowModel: getFilteredRowModel(),
  onSortingChange: setSorting,
  onColumnFiltersChange: setColumnFilters,
  onGlobalFilterChange: setGlobalFilter,
  state: { sorting, columnFilters, globalFilter },
});
```

This is `@tanstack/react-table`, a library specifically for building
data-heavy tables — sortable columns, filtering, and more — without hand-
rolling all of that logic yourself. `columns` is a declarative description,
one entry per column: which field of a `Complaint` it reads
(`accessorKey`), what header text to show, and, through `cell`, exactly how
to render that specific column's value — notice `priority`'s cell reuses
`getPriorityBadgeProps` from `lib/badge-utils.ts`, the exact same small
helper used back on the citizen's own filing-result screen in Chapter 4.4,
so a priority badge looks and behaves identically everywhere it appears in
this app. `useReactTable(...)` is the hook that turns that plain
description, plus the raw `filteredData` array, into a fully working,
interactive table object — `table.getHeaderGroups()` and
`table.getRowModel().rows`, used further down in the actual JSX, are how
that object hands back exactly what needs to be rendered, already sorted
and filtered according to whatever state the officer has currently set.

```tsx
const filteredData = useMemo(() => {
  let data = complaints;
  if (priorityFilter !== "all") data = data.filter((c) => c.priority === priorityFilter);
  if (statusFilter !== "all") data = data.filter((c) => c.status === statusFilter);
  return data;
}, [complaints, priorityFilter, statusFilter]);
```

Notice this project layers its own simple dropdown filters — priority and
status — on top of react-table's own filtering machinery, computed
separately with a `useMemo` you already fully understand from Chapter 4.5,
rather than routing every filter through the library. `.filter(...)` is
JavaScript's version of a Python list comprehension's filtering half — keep
only the items where the given condition is true — building a brand new,
narrower array without ever modifying the original `complaints` array.

## Turning a flat list of complaints into real numbers

```tsx
const metrics = useMemo(() => {
  const total = complaints.length;
  const emergencyHigh = complaints.filter((c) => c.priority === "Emergency" || c.priority === "High").length;
  const underReview = complaints.filter((c) => c.status === "Under Review").length;
  const closed = complaints.filter((c) => c.status === "Resolved" || c.status === "Closed").length;
  return { total, emergencyHigh, underReview, closed };
}, [complaints]);
```

This is worth reading as a genuinely useful, general pattern: a flat array
of records in, a small object of derived counts out, recomputed only when
the underlying data actually changes. `buildDailyComplaintData`, a little
further up the same file, does something slightly more involved with the
exact same underlying idea:

```tsx
function buildDailyComplaintData(complaints: Complaint[]): ChartPoint[] {
  if (complaints.length === 0) return [{ label: "No data", value: 0 }];
  const today = new Date();
  const days = Array.from({ length: 7 }, (_, index) => {
    const date = new Date(today);
    date.setHours(0, 0, 0, 0);
    date.setDate(today.getDate() - (6 - index));
    return date;
  });
  return days.map((date) => {
    const count = complaints.filter((complaint) => {
      if (!complaint.created_at) return false;
      const created = new Date(complaint.created_at);
      return created.getFullYear() === date.getFullYear() && created.getMonth() === date.getMonth() && created.getDate() === date.getDate();
    }).length;
    return { label: formatChartLabel(date), value: count };
  });
}
```

`Array.from({ length: 7 }, (_, index) => ...)` builds an array of exactly
seven items by running the given function seven times, once per index —
here, producing the last seven calendar days, oldest first, each one
`(6 - index)` days back from today. For each of those seven days, the inner
`.filter(...)` counts how many complaints were `created_at` on that exact
date, comparing year, month, and day individually rather than the full
timestamp, since a complaint's exact creation time down to the second
obviously won't match a day boundary exactly. The result, an array of
`{ label, value }` points, is exactly the shape `AnimatedComplaintLineChart`
— the same chart component used for the landing page's fake preview data
back in Chapter 4.2 — expects, which is precisely why that one component can
be reused for both a fabricated marketing preview and this page's genuinely
real, live data: the chart itself has no idea, and no need to know, where
its numbers actually came from.

## Think about it

1. `buildDailyComplaintData` compares year, month, and day separately rather
   than comparing full `Date` objects directly. What real bug would you run
   into if it instead compared two `Date` objects with `===` to check
   whether a complaint happened "on" a given day?
2. The dashboard's `filteredData` narrows the table down using
   `priorityFilter` and `statusFilter`, computed with a plain `useMemo`,
   while sorting and the search box are handled entirely by react-table's
   own internal state. Why do you think this project didn't route the
   priority and status filters through react-table's filtering system too,
   using one single mechanism for everything?
3. `AnimatedComplaintLineChart` can't tell the difference between the
   landing page's fabricated data and the dashboard's real data — it just
   draws whatever points it's given. What's one advantage, and one risk, of
   building a component this generic and reusable?
