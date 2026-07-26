"use client";

import { motion } from "motion/react";
import { Bot, ChevronDown, ChevronUp } from "lucide-react";
import { useState } from "react";
import { Skeleton } from "@/components/ui/skeleton";

export interface ChatMessageData {
  role: "user" | "agent";
  content: string;
  timestamp?: string;
  extracted_data?: Record<string, unknown>;
  suggested_followups?: string[];
}

export function UserMessage({ content, timestamp }: { content: string; timestamp?: string }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, ease: "easeOut" }}
      className="flex justify-end mb-4"
    >
      <div className="max-w-[70%] rounded-lg border border-accent/60 bg-primary text-white px-4 py-3 shadow-sm">
        <p className="text-sm">{content}</p>
        {timestamp && <p className="text-xs text-white/70 mt-1">{new Date(timestamp).toLocaleTimeString()}</p>}
      </div>
    </motion.div>
  );
}

export function AgentMessage({
  content,
  timestamp,
  suggested_followups,
  collected_fields,
}: {
  content: string;
  timestamp?: string;
  suggested_followups?: string[];
  collected_fields?: Record<string, unknown>;
}) {
  const [showDetails, setShowDetails] = useState(false);

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, ease: "easeOut" }}
      className="flex justify-start mb-4"
    >
      <div className="max-w-[70%] rounded-lg border border-border bg-card text-card-foreground px-4 py-3 shadow-sm">
        <p className="text-sm leading-relaxed">{content}</p>
        {timestamp && <p className="text-xs text-muted-foreground mt-1">{new Date(timestamp).toLocaleTimeString()}</p>}

        {suggested_followups && suggested_followups.length > 0 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.2, duration: 0.3 }}
            className="mt-3 space-y-2"
          >
            {suggested_followups.map((followup, i) => (
              <motion.button
                key={i}
                initial={{ scale: 0.9, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ delay: 0.1 + i * 0.05 }}
                className="block w-full text-left text-xs px-2 py-1.5 rounded border border-accent/60 text-white hover:bg-primary transition-colors"
              >
                {followup}
              </motion.button>
            ))}
          </motion.div>
        )}

        {collected_fields && Object.keys(collected_fields).length > 0 && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 }} className="mt-3">
            <button
              onClick={() => setShowDetails(!showDetails)}
              className="flex items-center gap-1 text-xs text-muted-foreground hover:text-white transition-colors"
            >
              {showDetails ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
              Details collected so far
            </button>
            {showDetails && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: "auto", opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.2 }}
                className="mt-2 p-2 rounded border border-border bg-background/80 text-xs space-y-1 overflow-hidden"
              >
                {Object.entries(collected_fields).map(([key, value]) => (
                  <div key={key} className="flex justify-between gap-2">
                    <span className="text-muted-foreground">{key}:</span>
                    <span className="text-white truncate">
                      {typeof value === "string" ? value : JSON.stringify(value)}
                    </span>
                  </div>
                ))}
              </motion.div>
            )}
          </motion.div>
        )}
      </div>
    </motion.div>
  );
}

export function SkeletonLoader({ status = "Analyzing complaint details" }: { status?: string }) {
  const dots = [0, 1, 2];
  const lines = [
    { width: "88%", delay: 0 },
    { width: "72%", delay: 0.1 },
    { width: "94%", delay: 0.2 },
    { width: "58%", delay: 0.3 },
  ];

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25, ease: "easeOut" }}
      className="flex justify-start mb-4"
      role="status"
      aria-live="polite"
      aria-label="Assistant is preparing a response"
    >
      <div className="w-full max-w-[min(88%,40rem)] rounded-lg border border-accent/30 bg-card/95 px-4 py-3 shadow-[0_0_22px_rgba(190,24,93,0.14)]">
        <div className="flex items-start gap-3">
          <div className="relative mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-md border border-accent/40 bg-background text-accent">
            <Bot className="h-4 w-4" aria-hidden="true" />
            <motion.span
              className="absolute -right-0.5 -top-0.5 h-2.5 w-2.5 rounded-full bg-accent"
              animate={{ opacity: [0.35, 1, 0.35], scale: [0.85, 1, 0.85] }}
              transition={{ duration: 1.4, repeat: Infinity, ease: "easeInOut" }}
            />
          </div>

          <div className="min-w-0 flex-1 space-y-3">
            <div className="flex flex-wrap items-center gap-2">
              <p className="text-xs font-medium text-foreground">{status}</p>
              <div className="flex items-center gap-1" aria-hidden="true">
                {dots.map((dot) => (
                  <motion.span
                    key={dot}
                    className="h-1.5 w-1.5 rounded-full bg-accent"
                    animate={{ opacity: [0.25, 1, 0.25], y: [0, -2, 0] }}
                    transition={{ duration: 0.9, repeat: Infinity, delay: dot * 0.15 }}
                  />
                ))}
              </div>
            </div>

            <div className="space-y-2">
              {lines.map((line) => (
                <motion.div
                  key={line.width}
                  initial={{ opacity: 0.45 }}
                  animate={{ opacity: [0.45, 0.95, 0.45] }}
                  transition={{ duration: 1.6, repeat: Infinity, delay: line.delay, ease: "easeInOut" }}
                >
                  <Skeleton className="h-3 rounded-sm bg-muted/80" style={{ width: line.width }} />
                </motion.div>
              ))}
            </div>

            <div className="grid grid-cols-2 gap-2 sm:grid-cols-3" aria-hidden="true">
              <Skeleton className="h-7 rounded-md border border-border/60 bg-background/70" />
              <Skeleton className="h-7 rounded-md border border-border/60 bg-background/70" />
              <Skeleton className="hidden h-7 rounded-md border border-border/60 bg-background/70 sm:block" />
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  );
}
