"use client";

import { motion, AnimatePresence } from "motion/react";
import { AlertCircle, CheckCircle2, ChevronDown, ChevronUp } from "lucide-react";
import { useState } from "react";
import type { MissingField, CollectedField } from "@/lib/complaint-intelligence";

interface MissingDetailsDetectorProps {
  missingDetails: MissingField[];
  collectedDetails: CollectedField[];
  criticalMissing: MissingField[];
}

export function MissingDetailsDetector({
  missingDetails,
  collectedDetails,
  criticalMissing,
}: MissingDetailsDetectorProps) {
  const [expanded, setExpanded] = useState(false);
  const [showCollected, setShowCollected] = useState(false);

  if (missingDetails.length === 0 && collectedDetails.length === 0) return null;

  const criticalCount = criticalMissing.length;
  const totalMissing = missingDetails.length;

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="rounded-lg border border-border bg-card p-4 space-y-3"
    >
      {/* Header */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between text-left"
      >
        <div className="flex items-center gap-2">
          <div className="relative">
            {criticalCount > 0 ? (
              <AlertCircle className="h-4 w-4 text-red-400" />
            ) : totalMissing > 0 ? (
              <AlertCircle className="h-4 w-4 text-yellow-400" />
            ) : (
              <CheckCircle2 className="h-4 w-4 text-green-400" />
            )}
          </div>
          <span className="text-sm font-medium text-foreground">
            {criticalCount > 0
              ? `${criticalCount} critical field${criticalCount > 1 ? "s" : ""} needed`
              : totalMissing > 0
                ? `${totalMissing} field${totalMissing > 1 ? "s" : ""} remaining`
                : "All required fields collected"}
          </span>
        </div>
        {expanded ? (
          <ChevronUp className="h-4 w-4 text-muted-foreground" />
        ) : (
          <ChevronDown className="h-4 w-4 text-muted-foreground" />
        )}
      </button>

      {/* Missing fields list */}
      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="space-y-2 pt-1">
              {missingDetails.map((field) => {
                const isCritical = criticalMissing.some((c) => c.key === field.key);
                return (
                  <div
                    key={field.key}
                    className={`flex items-start gap-2 text-xs rounded px-2 py-1.5 ${
                      isCritical
                        ? "bg-red-500/10 border border-red-500/20"
                        : field.priority === "high"
                          ? "bg-yellow-500/10 border border-yellow-500/20"
                          : "bg-muted/50 border border-border"
                    }`}
                  >
                    <AlertCircle
                      className={`h-3 w-3 mt-0.5 flex-shrink-0 ${
                        isCritical ? "text-red-400" : field.priority === "high" ? "text-yellow-400" : "text-muted-foreground"
                      }`}
                    />
                    <div>
                      <span className="font-medium text-foreground">{field.label}</span>
                      <span className="text-muted-foreground ml-1">— {field.hint}</span>
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Collected details */}
            {collectedDetails.length > 0 && (
              <div className="mt-3 pt-2 border-t border-border">
                <button
                  onClick={() => setShowCollected(!showCollected)}
                  className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
                >
                  {showCollected ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
                  {collectedDetails.length} field{collectedDetails.length > 1 ? "s" : ""} collected
                </button>
                {showCollected && (
                  <div className="mt-2 space-y-1">
                    {collectedDetails.map((field) => (
                      <div key={field.key} className="flex justify-between text-xs">
                        <span className="text-muted-foreground">{field.label}:</span>
                        <span className="text-foreground truncate ml-2 max-w-[200px]">{field.value}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
