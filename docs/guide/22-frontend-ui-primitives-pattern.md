# Chapter 4.9 — The Building Blocks Underneath Every Screen

Every page in Part 4 leaned on components you never actually saw
opened: `Button`, `Card`, `Input`, `Dialog`, `Select`, `Table`, `Badge`, and
more, all living under `frontend/src/components/ui/`. This closing chapter
of Part 4 explains that whole folder — not file by file, since, as flagged
back when this guide's plan was first agreed, these are largely generated,
boilerplate wrapper components with the same shape repeated roughly twenty
times. Instead, this chapter explains the *pattern* once, completely, using
`button.tsx` as the representative example, so you can read any of its
siblings yourself with genuine confidence.

## Why a "component library" exists at all

Think back to Chapter 4.4's form: `Input`, `Textarea`, `Button`, `Card` all
appeared with barely any styling code around them, and yet every one of them
already looked consistent with the rest of the app — same rounded corners,
same color palette, same hover behavior. That consistency isn't an accident,
and it isn't hand-tuned separately on every single page. It comes from a
**design system**: a shared, small set of building-block components,
defined once, styled once, and reused everywhere — so a form on the citizen
side and a table on the officer side, built by different people at
different times, still feel like they belong to the same product.

This project's design system is based on **shadcn/ui** — not a traditional
library you install and import from a package, but a collection of
component source code you copy directly into your own project (which is
exactly why you can find and read the full, real source of `Button` sitting
right there in `frontend/src/components/ui/button.tsx`, rather than buried
inside a `node_modules` folder you'd never normally look at). Underneath it,
these components are built on **Radix**-style primitives — here, a library
called `@base-ui/react` — which handle the genuinely hard, easy-to-get-wrong
parts of interactive UI: keyboard navigation, focus handling, accessibility
attributes for screen readers, correctly trapping focus inside an open
dialog. shadcn/ui's job is to take those solid, unstyled primitives and give
them this project's specific visual style.

## Reading button.tsx as the pattern

```tsx
import { Button as ButtonPrimitive } from "@base-ui/react/button"
import { cva, type VariantProps } from "class-variance-authority"
import { cn } from "@/lib/utils"

const buttonVariants = cva(
  "group/button inline-flex shrink-0 items-center justify-center rounded-lg border border-transparent ...",
  {
    variants: {
      variant: {
        default: "bg-primary text-primary-foreground hover:bg-primary/80",
        outline: "border-border bg-background hover:bg-muted ...",
        secondary: "bg-secondary text-secondary-foreground ...",
        ghost: "hover:bg-muted hover:text-foreground ...",
        destructive: "bg-destructive/10 text-destructive ...",
        link: "text-primary underline-offset-4 hover:underline",
      },
      size: {
        default: "h-8 gap-1.5 px-2.5 ...",
        sm: "h-7 gap-1 ... text-[0.8rem] ...",
        lg: "h-9 gap-1.5 px-2.5 ...",
        icon: "size-8",
        ...
      },
    },
    defaultVariants: { variant: "default", size: "default" },
  }
)

function Button({ className, variant = "default", size = "default", ...props }: ButtonPrimitive.Props & VariantProps<typeof buttonVariants>) {
  return (
    <ButtonPrimitive
      data-slot="button"
      className={cn(buttonVariants({ variant, size, className }))}
      {...props}
    />
  )
}

export { Button, buttonVariants }
```

`cva` — **class-variance-authority** — is worth understanding precisely,
because you saw its output, `buttonVariants(...)`, used directly in several
earlier chapters (Chapter 4.2's navbar links, for one). It takes a base set
of classes always applied to every button, plus a `variants` object
describing named, alternative sets of classes — here, `variant` (`default`,
`outline`, `secondary`, `ghost`, `destructive`, `link`) and `size`
(`default`, `sm`, `lg`, `icon`, and more) — and returns a function. Calling
that function, `buttonVariants({ variant: "outline", size: "lg" })`,
produces the exact right combination of class names for that specific
combination, falling back to `defaultVariants` for anything left
unspecified. This is precisely why, back in Chapter 4.2, you saw
`buttonVariants({ variant: "ghost", size: "lg" })` used directly to style a
plain `<Link>` as if it were a button — `buttonVariants` is genuinely
reusable independently of the `Button` component itself, because it's just a
function that produces a string of class names, usable anywhere.

`ButtonPrimitive.Props & VariantProps<typeof buttonVariants>` is TypeScript's
`&`, meaning "an intersection type" — a value that must satisfy *both* types
at once. In plain terms: this component accepts every prop the underlying
`@base-ui/react` button primitive already accepts (all its accessibility and
event-handling behavior), *plus* the `variant` and `size` props `cva`
defined. `{...props}` — the spread operator again, this time on the
receiving end of a function's arguments — forwards every other prop this
component wasn't specifically interested in straight through to the
underlying primitive untouched, so nothing about the primitive's real
behavior gets accidentally lost or blocked by this wrapping layer.

`cn(buttonVariants({ variant, size, className }))` is the final piece,
tying directly back to Chapter 4.2: `buttonVariants` produces this variant's
base classes, and `className` — whatever extra classes a specific usage
passed in, like `"h-12 gap-2 rounded-xl px-5"` on the landing page's big
"File Complaint" button — gets merged in through the same `cn`/`twMerge`
mechanism you already learned, so a one-off override on a specific button
cleanly wins over this component's own defaults, without silently leaving
both conflicting classes active at once.

## Why this chapter stops here

Every other file in this folder — `card.tsx`, `input.tsx`, `dialog.tsx`,
`select.tsx`, `table.tsx`, `badge.tsx`, and the rest — follows this same
shape: wrap an underlying primitive, define styling variants with `cva`
where relevant, forward props through with the spread operator, merge
classes with `cn`. Reading all eighteen in full would mean reading the same
lesson eighteen times with different CSS values. What's genuinely worth your
attention, if you're curious, is opening one or two of the more interactive
ones yourself now — `dialog.tsx`, which `FIRDraftPreview` from Chapter 4.6
builds on, or `select.tsx`, used throughout the officer screens in Chapters
4.7 and 4.8 — and seeing this exact same pattern in a slightly more involved
form.

`AnimatedComplaintLineChart`, used in Chapters 4.2 and 4.7, deserves one
last honest mention here too: it's not a shadcn primitive at all, but it's
similar in spirit — a single, focused, reusable component, taking plain data
in (`ChartPoint[]`) and handling its own internal complexity (in this case,
converting numeric values into SVG coordinates, and drawing an animated line
and gradient fill) so that nothing else in this codebase ever needs to know
or care how an SVG chart actually gets drawn. That's the whole underlying
value of every component in this chapter, in one sentence: hide real
complexity behind a small, stable, well-named surface, so the rest of the
application — everything you spent the whole of Part 4 reading — gets to
stay simple, readable, and focused on what it actually does.

## Think about it

1. shadcn/ui components are copied directly into this project's source code,
   rather than installed as a package from `node_modules` the way most
   libraries are. What's one real advantage of being able to open and edit
   `button.tsx` directly, and one real cost of that choice, compared to a
   traditional installed library you can't as easily change?
2. `buttonVariants` is used both to style the actual `<Button>` component
   and, separately, to style a plain `<Link>` to look like a button, back in
   Chapter 4.2. What does that tell you about the real difference between
   "how something looks" and "what component it technically is" in this
   codebase?
3. If you needed to add a brand-new `"success"` variant to `Button` — a
   green-tinted button for a confirmation action somewhere in this app —
   walk through exactly what you would change in `button.tsx`, using
   everything you now understand about how `cva` and `buttonVariants` work
   together.
