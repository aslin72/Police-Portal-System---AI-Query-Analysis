# Chapter 4.5 — The Chat Screen: Where Every React Hook Comes Together

`frontend/src/app/citizen/chat/page.tsx`, alongside `chat-input.tsx` and
`chat-message.tsx`, is the conversational filing flow from Chapter 1.1, and
it is the single richest piece of React in this entire project. Nearly
every hook you've met so far shows up here, plus two new ones — `useEffect`
and `useRef` — and, more importantly, they all work together to produce one
coherent experience: a live conversation with an assistant that remembers
context, shows what it's doing, and knows when it has enough to file.

## Reacting to something happening, automatically: useEffect

```tsx
useEffect(() => {
  messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
}, [messages, loading]);
```

`useEffect` is how a React component does something in reaction to a change,
rather than in reaction to a user directly clicking something. Read its two
arguments carefully: the function is the thing to actually do; the array at
the end, `[messages, loading]`, is its **dependency list** — React re-runs
this function every time any value in that list changes since the last
render, and does nothing otherwise. Here, that means: every time a new
message gets added to `messages`, or `loading` flips on or off, scroll the
chat window down to the newest message. `messagesEndRef.current` — introduced
properly below — is an empty marker `<div>` sitting at the very bottom of
the message list purely so this line has something concrete to scroll to.

A second `useEffect`, with an empty dependency array `[]`, runs the loading
logic when the page first opens:

```tsx
useEffect(() => {
  const init = async () => {
    const loadedSessions = await refreshSessions();
    const mostRecent = loadedSessions.find((s) => !s.is_filed);
    if (mostRecent) {
      setActiveSessionId(mostRecent.id);
      await loadMessages(mostRecent.id);
    } else {
      createNewSession();
    }
    setIsInitializing(false);
  };
  init();
}, []);
```

An empty dependency array, `[]`, means "there is nothing this effect
depends on that would need it to run again" — so React runs it exactly
once, right after the component first appears on screen, and never again
after that. This is the frontend's equivalent of a program's startup
routine, and it's making a real product decision worth noticing:
`loadedSessions.find((s) => !s.is_filed)` looks for a conversation the
citizen already started but hasn't finished filing yet, and resumes it
automatically, rather than always starting the citizen on a blank chat —
if none exists, only then does it fall back to `createNewSession()`. This
is a genuinely thoughtful piece of product design: a citizen who got
interrupted midway through describing something difficult shouldn't have to
start over from nothing when they come back.

## Reaching directly into the page: useRef

```tsx
const messagesEndRef = useRef<HTMLDivElement>(null);
...
<div ref={messagesEndRef} />
```

`useRef` creates a container that holds onto a value across renders without
causing a re-render itself when that value changes — genuinely different
from `useState`, which always triggers a re-render. Here, it's used for its
most common purpose: getting a direct handle on a real, rendered piece of
the page (`messagesEndRef.current` becomes the actual `<div>` DOM element
once it's rendered), so ordinary browser methods like `.scrollIntoView(...)`
can be called on it directly — something React's normal declarative
rendering doesn't otherwise give you a way to do.

## Avoiding wasted work: useMemo and useCallback

```tsx
const lastExtractedFields = useMemo(() => {
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i];
    if (msg.role === "agent" && msg.extracted_data && Object.keys(msg.extracted_data).length > 0) {
      return msg.extracted_data;
    }
  }
  return {};
}, [messages]);

const complaintInsight: ComplaintInsight = useMemo(
  () => deriveComplaintInsight(lastExtractedFields, messages),
  [lastExtractedFields, messages],
);
```

`useMemo` caches the *result* of a calculation, and only recomputes it when
something in its dependency list actually changes — here, `messages`.
Without it, this loop searching backward through every message for the most
recent extracted data would rerun on every single re-render of this
component, even ones triggered by something completely unrelated, like the
input box's focus state changing. `deriveComplaintInsight` — you'll meet
this function properly in the next chapter — does genuinely nontrivial work,
scanning message text for dozens of keyword patterns, so caching its result
until the underlying messages actually change is a real, meaningful
efficiency choice, not a micro-optimization for its own sake.

```tsx
const createNewSession = useCallback(() => {
  setActiveSessionId(createSessionId());
  setMessages([]);
  setReadyToFile(false);
  setShowDraftPreview(false);
}, []);
```

`useCallback` is the same underlying idea as `useMemo`, applied to a
function instead of a value: it returns the *same* function instance across
re-renders, as long as its dependencies (here, none — an empty array) haven't
changed, rather than creating a brand new function on every single render.
This matters specifically because `createNewSession` is passed down to, and
depended on by, other hooks and event handlers throughout this file — giving
those a stable, unchanging reference to depend on avoids a cascade of
unnecessary extra work every time this component re-renders for any reason
at all.

## Sending a message and living through the wait

```tsx
const handleSendMessage = async (content: string) => {
  if (!activeSessionId || !content.trim()) return;

  setMessages((prev) => [...prev, { role: "user", content, timestamp: new Date().toISOString() }]);
  setStreamingStatus("Preparing assistant response");
  setLoading(true);

  try {
    const chatResponse = await chatComplaintStream(activeSessionId, content, (event) => {
      if (event.event === "status") {
        setStreamingStatus(event.message);
      }
    });

    setMessages((prev) => [...prev, { role: "agent", content: chatResponse.agent_message, ... }]);
    setReadyToFile(!!chatResponse.ready_to_file);
    await refreshSessions();
  } catch (error) {
    console.error("Error sending message:", error);
    toast.error("Failed to send message. Please try again.");
  } finally {
    setLoading(false);
    setStreamingStatus("Preparing assistant response");
  }
};
```

This function is the payoff for everything you learned about SSE, both
sides, in Chapters 2.4 and 4.3, and it's worth tracing end to end one more
time, now completely concretely. The citizen's own message is added to
`messages` immediately, optimistically, before any network call has even
started — so the chat feels instantly responsive rather than freezing while
waiting on the network. `chatComplaintStream` is called with a callback as
its third argument — recall from Chapter 4.3 this is exactly the `onEvent`
parameter — and every time the backend's generator in Chapter 2.4 `yield`s a
new `"status"` event ("Extracting new complaint details," and so on), that
callback fires here, updating `streamingStatus`, which `SkeletonLoader`
displays live while the citizen waits — this is the whole reason this
project bothered with a streaming endpoint at all, made now fully visible,
front to back, in one function.

`try`/`catch`/`finally` — a pattern parallel to Python's `try`/`except`, with
one addition: `finally` runs no matter what, whether the `try` succeeded or
the `catch` caught an error, which is exactly why `setLoading(false)` lives
there — the loading indicator must always turn off, success or failure,
or the interface would stay stuck showing "typing" forever after a failed
request.

## Displaying a growing conversation

```tsx
{messages.map((msg, idx) =>
  msg.role === "user" ? (
    <UserMessage key={idx} content={msg.content} timestamp={msg.timestamp} />
  ) : (
    <AgentMessage key={idx} ... />
  )
)}
{loading && <SkeletonLoader status={streamingStatus} />}
```

This is `.map()` again, from Chapter 4.2, now choosing between two entirely
different components based on `msg.role` — a **ternary expression**
(`condition ? thenThis : elseThis`), JavaScript/TypeScript's compact form of
an if/else, used constantly throughout JSX because a full `if` statement
isn't valid directly inside markup like this. `key={idx}` uses the message's
position in the array as its key, a reasonable choice here specifically
because messages in this conversation only ever get appended to the end, and
never reordered or removed from the middle — the caveat from Chapter 4.2
about `key` needing to reliably identify each item doesn't cause problems in
this particular, append-only case.

Look at `chat-message.tsx`'s `AgentMessage`:

```tsx
const [showDetails, setShowDetails] = useState(false);
...
<button onClick={() => setShowDetails(!showDetails)}>
  {showDetails ? <ChevronUp /> : <ChevronDown />}
  Details collected so far
</button>
{showDetails && (
  <motion.div ...>
    {Object.entries(collected_fields).map(([key, value]) => (
      <div key={key}>
        <span>{key}:</span>
        <span>{typeof value === "string" ? value : JSON.stringify(value)}</span>
      </div>
    ))}
  </motion.div>
)}
```

Every single `AgentMessage` on screen holds its own independent
`showDetails` state — expanding one message's "details collected so far"
panel has zero effect on any other message's panel, because each rendered
component instance gets its own completely separate copy of that state.
This is a genuinely important thing to internalize about how React state
works: it belongs to the specific component instance it was declared in,
not to the page as a whole.

`motion.div`, used throughout this file and its neighbors, comes from a
library called `motion` (previously known as Framer Motion) — it renders a
regular element, but lets you declare animations declaratively: `initial`
(the starting state), `animate` (the state to animate toward), and
`transition` (how long, and with what easing). You'll see this pattern
throughout this whole part of the frontend and can now recognize it
immediately without needing it re-explained every time it appears.

## Think about it

1. `handleSendMessage` adds the citizen's own message to `messages`
   immediately, before the network request has even started, rather than
   waiting for the backend to confirm it was received. What would the chat
   feel like to use if it waited for the backend's response before showing
   the citizen's own message at all? What's the actual risk of showing it
   early?
2. The very first `useEffect` in this chapter runs once, on page load, and
   tries to resume the most recent unfiled session. What do you think should
   happen if that citizen is using a different device or a different
   browser than the one they started their conversation on — would this
   logic find their earlier session? Why or why not, given what you know
   about how a browser stores things versus what actually lives in the
   database from Part 2?
3. `useMemo` and `useCallback` both exist to avoid unnecessary recalculation.
   Can you think of a reason you might *not* want to wrap every single value
   and function in a component with one of these — in other words, what's
   the cost of using them, not just the benefit?
