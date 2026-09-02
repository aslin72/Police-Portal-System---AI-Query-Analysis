# Chapter 6.3 — Planning a Feature From a Blank Page

Everything so far in this guide has been about understanding a system that
already exists. This chapter turns that around: given everything you now
know about how this project thinks — its architecture, its habit of
separating AI judgment from explainable rules, its instinct to always have a
safe fallback — how would you actually plan a brand-new feature for it,
starting from nothing but an idea?

## Start with the person, not the feature

Chapter 1.1 didn't open by describing screens or endpoints — it opened by
naming the citizen and the officer, and what each of them actually needs.
That's not a stylistic choice this guide made; it's the correct starting
point for planning any real feature, and it's worth internalizing as a
habit you carry into every project after this one. Before writing a single
line of code, or even a single database column, answer three questions in
plain language: who specifically is this for, what are they trying to
accomplish, and what does it cost them — in time, in stress, in risk — if
this feature doesn't exist or works badly?

Take a concrete example, one this project genuinely doesn't have yet:
letting a citizen receive a notification when their complaint's status
changes, instead of having to manually revisit `/citizen/track-complaint`
and check. Who is this for? A citizen who has already filed something and is
anxiously waiting to hear back — plausibly someone in a stressful, uncertain
situation, exactly the same person Chapter 1.1 described as the citizen
persona from the very start. What are they trying to accomplish? Not
checking a status page repeatedly out of anxiety; knowing, the moment
something changes, without having to ask. What does it cost them if this
doesn't exist? Genuine, ongoing uncertainty about something that may matter
a great deal to them.

## Define the smallest version that's actually real

A very common mistake, especially once you're excited about a new feature,
is designing the full, polished version first — every edge case handled,
every notification channel supported, a beautiful settings page for
preferences — before anything at all actually works. The far better
instinct, and the one this project's own AI service modeled for you
constantly throughout Part 3, is to find the smallest version of the
feature that's genuinely useful on its own, ship that, and learn from it
before building further.

For the notification example: the smallest real version might be nothing
more than an email sent whenever `update_triage` in `database.py` (Chapter
2.3) successfully changes a complaint's `status`, to whatever email address
is on file for that complaint's `reporter_email`. No preferences screen, no
SMS option, no in-app notification center — just one email, at one moment,
triggered by one specific, well-understood event you already fully
understand from Chapter 2.3. That's small enough to actually build,
test, and ship quickly, and it already delivers most of the real value:
a citizen no longer has to guess.

## Trace where it actually plugs into what already exists

This is where everything you learned in Parts 2 through 5 becomes directly,
practically useful, not just historically interesting. Before writing any
code, trace the feature through this project's existing architecture the
same way Chapter 5.2 traced a whole complaint's life: which existing file
does the triggering event already pass through? `update_triage` in
`database.py`, called from `patch_triage` in `routes.py` — you already know
both of these in complete detail. Does the data you need already exist?
`reporter_email` is already a field on every complaint, saved back in
Chapter 2.2's schema and Chapter 2.3's table — nothing new needed there.
What's genuinely new? Something that actually sends an email — a new,
small, focused piece of code, following the exact same instinct you saw
throughout `ai_service.py`: call an external service (an email-sending
provider, this time, instead of an AI provider), and handle the case where
that call fails without letting the failure break the actual status update
that triggered it in the first place — recall Chapter 2.4's evidence-upload
error handling as your model for exactly that shape of resilience.

## Decide, deliberately, what could go wrong

Every chapter in Part 3 modeled this instinct for AI features specifically;
it applies just as much to any feature at all. What happens if the email
provider is down? The status update itself should still succeed — an
officer's legitimate work should never be blocked by a notification
feature failing quietly in the background, the exact same principle you saw
protecting evidence uploads and complaint filing throughout Part 2. What
happens if `reporter_email` was never provided in the first place — recall
from Chapter 2.2 that it's optional? The feature should just quietly do
nothing for that complaint, not throw an error that somehow blocks the
officer's status update.

## A short, reusable planning checklist

Distilled from everything above, and directly informed by how this actual
project is built: name the specific person this serves and what they
actually need. Define the smallest version that delivers real value on its
own. Trace exactly which existing files and data this plugs into, and what,
concretely, is genuinely new. Decide what happens when the new part fails,
and make sure that failure can never take down something that already
worked. And, borrowing directly from Chapter 3.4's closing lesson: decide
which parts of this feature need real judgment, and which parts are
mechanical enough to be plain, predictable code — because that question
turns out to matter for almost any feature, not just ones that involve AI.

## Think about it

1. Using the six-question checklist above, plan a second, different small
   feature for this project: letting an officer flag a complaint as a
   suspected duplicate of another one already in the system. Who is it for,
   what's the smallest real version, and which existing files would it plug
   into?
2. The email-notification example deliberately avoided building a
   preferences screen in its first version. What real risk does skipping
   that carry — is there a citizen this smallest version might genuinely
   annoy or fail, and is that an acceptable trade-off for shipping something
   small first?
3. Think back to a feature or app you personally use regularly. Can you
   identify what its own "smallest real version" probably looked like,
   before everything else was added around it? What did that first version
   likely leave out on purpose?
