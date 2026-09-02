# Chapter 4.2 — The Landing Page and Navigation

This chapter covers the first thing anyone actually sees: `frontend/src/app/page.tsx`
(the landing page, at the site's root URL `/`), `frontend/src/components/navbar.tsx`,
and `frontend/src/components/brand/PortalLogo.tsx`. It's also where you'll
meet a genuinely important distinction: not everything that looks like real
data on a screen actually is real data.

## Rendering a list from an array

```tsx
const services = [
  { title: "File a Complaint", body: "Submit structured complaints with location, time, details, and supporting evidence.", href: "/citizen/chat", icon: FileText },
  { title: "AI-Powered Triage", body: "Classify, prioritize, and route complaints to the right unit using explainable rules.", href: "/citizen/chat", icon: Brain },
  ...
];
```

```tsx
{services.map((service, index) => {
  const Icon = service.icon;
  return (
    <Link key={service.title} href={service.href} className="...">
      <Icon className="h-7 w-7 text-accent" />
      <h2>{service.title}</h2>
      <p>{service.body}</p>
    </Link>
  );
})}
```

This is the single most common pattern you'll see across the entire
frontend, so it's worth locking in here, on its simplest example, before it
shows up dozens more times. `services` is a plain array of objects.
`.map(...)` — the same idea as Python's list comprehensions from Part 2,
just spelled differently — transforms each item in that array into a piece
of JSX, producing a new array of components, which React then renders one
after another. `key={service.title}` is required by React whenever you
render a list this way: React needs a way to tell which rendered item
corresponds to which array entry, especially if the list ever changes, and
the `key` is how you tell it — normally something guaranteed unique, like an
ID.

Notice `const Icon = service.icon; ... <Icon .../>` — this looks unusual the
first time you see it, but it's a real, valid pattern: `service.icon` holds
an actual component (imported at the top of the file from an icon
library, `lucide-react`), assigned to a capitalized variable name (React
requires a capital letter to treat something as a component rather than a
plain HTML tag), and then used directly as JSX. This is exactly how the
`services` array can drive both the icon and the text for each card from one
single data source, rather than writing four nearly-identical blocks of JSX
by hand.

`href={service.href}` combined with `Link` — imported from `next/link` — is
Next.js's way of handling internal navigation without a full page reload:
clicking it swaps the visible page instantly, keeping the app feeling fast,
rather than the browser doing a traditional full-page fetch the way a plain
HTML `<a>` tag would.

## Real UI, fake data — and why that's completely normal

Look closely at the top of `page.tsx`:

```tsx
const dashboardMetrics = [
  { label: "Total Complaints", value: "324", change: "+12%" },
  { label: "Under Review", value: "87", change: "+8%" },
  ...
];

const recentComplaints = [
  { title: "Noise Disturbance", id: "#CP-2025-0187", status: "Under Review" },
  ...
];

const trustMetrics = [
  { label: "1,247+ Departments", icon: Building2 },
  { label: "99.99% Uptime", icon: Activity },
  ...
];
```

This is genuinely worth pausing on, because it's an easy thing to miss: none
of this comes from the backend. There is no `fetch(...)` call anywhere in
this file, no `getComplaints()` from `lib/types.ts` — you'll meet that
function properly in the next chapter, once you see it actually used for
real, on the officer dashboard. `LandingDashboardPreview`, the component
rendering what looks like a live command-center screen with real-looking
complaint numbers, is entirely hardcoded, illustrative data, built purely to
show a visitor what the *real* dashboard, on `/officer/dashboard`, generally
looks like, before they've ever logged in or filed anything.

This is a completely normal, extremely common real-world pattern, and it's
worth naming clearly so you never mistake one for the other again: a
marketing or landing page's job is to *look* like the product convincingly,
not to *be* the product. "1,247+ Departments" and "99.99% Uptime" are
placeholder trust-signal copy, not numbers pulled from any real measurement
anywhere in this system. Once you reach Chapter 4.7 and see the real officer
dashboard fetching real complaints with `getComplaints()`, the contrast will
make this distinction completely concrete: that page has a `useEffect` and an
actual network request; this page has neither, on purpose.

## The navbar: one component, two completely different looks

```tsx
export default function Navbar() {
  const pathname = usePathname();
  const isLandingPage = pathname === "/";

  if (isLandingPage) {
    return ( /* ... dark, marketing-style header ... */ );
  }

  return ( /* ... compact application header ... */ );
}
```

`usePathname()` is a React **hook** — a special kind of function, always
starting with `use`, that lets a component tap into some piece of
React or Next.js machinery, here, the current URL path. `Navbar` is rendered
by `layout.tsx` on every single page, as you saw in the last chapter — but
it deliberately renders one of two genuinely different headers depending on
where the citizen currently is: a large, dark, marketing-style header with
"Services," "How It Works," and "Security" navigation links when they're on
the landing page, and a compact, functional application header, with direct
links to the citizen and officer sections, everywhere else. This is a real,
deliberate product decision made visible in code: a marketing page and a
working application have different jobs, and this project chooses to give
them visually distinct navigation instead of forcing one generic header to
serve both purposes adequately.

```tsx
{citizenLinks.map((link) => {
  const Icon = link.icon;
  const active = pathname.startsWith(link.href);
  return (
    <Link key={link.href} href={link.href} className={cn(
      buttonVariants({ variant: "ghost", size: "sm" }),
      "gap-1.5",
      active ? "bg-primary text-primary-foreground hover:bg-accent" : "text-muted-foreground hover:bg-card hover:text-white",
    )}>
      <Icon className="h-4 w-4" />
      <span className="hidden sm:inline">{link.label}</span>
    </Link>
  );
})}
```

`pathname.startsWith(link.href)` is how the navbar knows to visually
highlight whichever section the citizen is currently in — a small, genuinely
useful piece of interface feedback. `cn(...)`, imported from
`frontend/src/lib/utils.ts`, is worth understanding on its own, since you'll
see it in nearly every frontend file from here on:

```ts
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}
```

This project styles everything with **Tailwind CSS** — instead of writing
separate CSS files, you apply pre-built utility classes directly in your
markup (`text-sm`, `rounded-lg`, `bg-primary`, and so on, each doing exactly
one small styling job). `clsx(inputs)` takes a list of class names — some
always applied, some only conditionally, exactly like the `active ? "..." :
"..."` above — and joins them into one string, safely skipping over any
`false` or `undefined` values that come from a condition that didn't match.
`twMerge(...)` then resolves conflicts between Tailwind classes
specifically — if two of the combined classes would both try to set the
element's background color, for instance, it keeps only the one that should
actually win, rather than leaving both in and letting the browser's own,
much less predictable rules decide. `cn` is simply this project's own
short, memorable name for "combine class names safely," used constantly
because nearly every component here has some styling that depends on
its state or props.

## A logo pointing at a file that no longer exists

```tsx
const LOGO_SRC = "/assets/ChatGPT%20Image%20Jul%2026,%202026,%2002_09_42%20AM.png";
```

This is worth flagging honestly rather than glossing over, because it's a
real, current problem in this codebase, not a hypothetical teaching example.
`PortalLogo.tsx` — used by the navbar on every single page — points at a
specific image file under `frontend/public/assets/`. As of this guide being
written, that exact file has been deleted from the project (a new file,
`logo.png`, has been added alongside it, but nothing in the code has been
updated to actually reference it yet). The practical effect: right now, the
logo shown throughout this application would fail to load — a broken image
placeholder — until either the old filename is restored, or `LOGO_SRC` is
updated to point at the new `logo.png` instead. It's flagged here rather than
quietly fixed, because it's exactly the kind of small, easy-to-miss
mismatch between "what the code references" and "what actually exists on
disk" that's worth learning to notice yourself — it will not be the last
time you encounter one in a real project.

## Think about it

1. The landing page's `dashboardMetrics` and `trustMetrics` are hardcoded
   arrays sitting directly in `page.tsx`, not fetched from anywhere. If this
   product were being demoed to a real police department, what's one risk of
   leaving numbers like "99.99% Uptime" and "1,247+ Departments" as
   permanent hardcoded copy rather than clearly labeling them as
   illustrative?
2. `Navbar` decides which header to show using `pathname === "/"` — a single
   exact-match check. `citizenLinks`, further down the same file, use
   `pathname.startsWith(link.href)` instead. Why do you think one check uses
   exact equality and the other uses `startsWith`? What would break if they
   were swapped?
3. `PortalLogo`'s broken image reference wouldn't cause the app to crash —
   the page would still load, just with a missing logo. Contrast that with
   the syntax error you learned about in `backend/questions.py` back when
   this guide first described the state of the working tree. Why does one
   broken reference cause a hard crash and the other just a cosmetic bug?
   What does that tell you about the difference between a compiled/parsed
   error and a broken runtime reference?
