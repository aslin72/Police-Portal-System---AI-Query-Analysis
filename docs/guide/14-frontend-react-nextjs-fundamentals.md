# Chapter 4.1 — React and Next.js: The Mental Model

We now cross from the backend, which you've fully covered, into the
frontend — everything under `frontend/`, the half of this system a citizen or
an officer actually sees and touches. Before reading a single real screen,
this chapter builds the vocabulary every later frontend chapter assumes:
what React and Next.js actually are, and what it means to write UI as code.

## What React actually is

Recall from Chapter 0.2 that the frontend's whole job is showing information
clearly and collecting input correctly. **React** is the library this
project uses to do that. Its core idea is genuinely simple to state, even
though mastering it takes practice: instead of manually writing code that
pokes at a web page — "find this element, change its text, now show this
other element" — you write a description of what the page should look like
for a given set of data, called a **component**, and React handles actually
updating the real page to match whenever that data changes.

A component is a JavaScript (or, in this project, TypeScript) function that
returns a description of some UI. Here's the smallest real one in this
project, `frontend/src/lib/utils.ts` aside, from `PortalLogo.tsx`, trimmed:

```tsx
export function PortalLogo({ size = "nav", animated = false }: PortalLogoProps) {
  return (
    <span className="...">
      <Image src={LOGO_SRC} alt="" fill />
    </span>
  );
}
```

`PortalLogo` is a function, and it returns something that looks almost like
HTML but isn't quite — that's **JSX**, a syntax extension that lets you write
markup directly inside JavaScript/TypeScript code. It gets compiled into
regular function calls before it ever runs in a browser; the angle-bracket
syntax is purely a convenience for the person writing it. `{ size = "nav",
animated = false }` are this component's **props** — short for properties —
the inputs a component receives, exactly analogous to a Python function's
parameters, which you already understand deeply from Part 2. Someone using
this component elsewhere writes `<PortalLogo size="hero" animated />`, and
React runs the function with those exact values filled in.

## What "state" means

A component that only ever renders its props would be static — it could
never change in response to anything the user does. **State** is a
component's own private, changing data — like the text currently typed into
a search box, or whether a dropdown is open — and React re-renders whatever
part of the page depends on that state automatically the instant it changes.
You'll see this constantly, spelled `useState`, starting in the very next
chapter and continuing through the rest of Part 4. This is worth connecting
directly back to something you already learned deeply in Part 2: state is
exactly the frontend's equivalent of the backend keeping data in memory
during a request — except here, it persists for as long as the citizen keeps
that page open in their browser, not just for the length of one request.

## Server components, client components, and "use client"

You'll notice a line at the very top of most frontend files in this project:

```tsx
"use client";
```

This single line is worth understanding precisely, because Next.js draws a
real, meaningful distinction here that trips up almost everyone at first.
Next.js can render a component in two different places: on the **server**
(before the page is ever sent to the citizen's browser, producing plain
HTML) or on the **client** (running live, inside the citizen's actual
browser, able to react to clicks, hold state, and update the page instantly
without a full reload). By default, every component in this project's
`app/` folder is a **server component** — rendered once, on the server,
before delivery. `"use client"` at the top of a file opts that one file, and
everything it renders, into being a **client component** instead — meaning
it actually runs inside the browser, and, critically, is allowed to use
`useState`, respond to clicks, and do everything else that requires a live,
interactive page.

This is exactly why `layout.tsx`, coming up next, has no `"use client"` line
at the top, while nearly every page you'll read for the rest of Part 4
does — a citizen filing a complaint needs a form that responds to their
typing right now, in their browser; that's fundamentally a client-side job.

## The App Router: files as routes

```tsx
export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}>
      <body className="min-h-full flex flex-col bg-background">
        <Navbar />
        <main className="flex-1">{children}</main>
        <Toaster />
      </body>
    </html>
  );
}
```

This is `frontend/src/app/layout.tsx`, the outermost wrapper for every single
page in this project. Next.js uses **file-based routing** — the folder
structure under `frontend/src/app/` directly determines the site's URLs, no
separate routing configuration file needed. `app/officer/dashboard/page.tsx`
becomes the page at `/officer/dashboard`. `app/citizen/chat/page.tsx` becomes
`/citizen/chat`. A folder named in square brackets, like
`app/officer/complaints/[id]/page.tsx`, becomes a **dynamic route** —
`[id]` is a placeholder, matching `/officer/complaints/42`,
`/officer/complaints/107`, or any other value, exactly the same idea as
FastAPI's `{complaint_id}` path parameter from Chapter 2.4, just expressed
with square brackets instead of curly ones. `layout.tsx` is special: it
wraps every page inside the same folder (and, at the root, every single page
in the whole app), which is exactly why `<Navbar />` shows up on every screen
without every individual page needing to import and render it itself.
`{children}` is where whatever specific page the citizen navigated to
actually gets inserted — you'll recognize this as React's version of a
template with a slot in it.

`geistSans` and `geistMono` load specific fonts, using Next.js's built-in
font loading; `export const metadata` sets the page's title and description,
used by the browser tab and search engines — the direct frontend equivalent
of `FastAPI(title="...")` from Chapter 2.1, both existing to name the
application for whoever, or whatever, is looking at it from the outside.

## Think about it

1. `layout.tsx` has no `"use client"` at the top, but the `Navbar` component
   it renders does. Given what you now know about server versus client
   components, is that a contradiction, or is it fine for a server component
   to render a client component inside it? What do you think determines the
   answer?
2. React re-renders a component's output automatically whenever its state
   changes, without you writing code that manually updates the page. Compare
   that to `database.py`'s `create_table()` from Chapter 2.3, which only
   updates the database when explicitly called. Which model — "update
   automatically when data changes" or "update only when told to" — feels
   more natural to you right now, and why?
3. Next.js turns `app/officer/complaints/[id]/page.tsx` into a working route
   automatically, just from where the file sits in the folder structure,
   with no separate routing configuration to maintain. What's one advantage
   and one disadvantage you can think of for tying a URL's structure this
   directly to a project's file structure?
