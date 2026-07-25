"use client";

import { motion, AnimatePresence } from "motion/react";
import { X, FileText, User, MapPin, AlertTriangle, Shield, ArrowRight } from "lucide-react";
import type { DraftData } from "@/lib/complaint-intelligence";

interface FIRDraftPreviewProps {
  open: boolean;
  onClose: () => void;
  draftData: DraftData;
}

const PRIORITY_COLORS: Record<string, string> = {
  Emergency: "bg-red-500/15 text-red-300 border-red-500/30",
  High: "bg-orange-500/15 text-orange-300 border-orange-500/30",
  Medium: "bg-yellow-500/15 text-yellow-300 border-yellow-500/30",
  Low: "bg-green-500/15 text-green-300 border-green-500/30",
};

function Section({
  icon: Icon,
  title,
  children,
}: {
  icon: typeof FileText;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2 text-xs font-semibold text-foreground uppercase tracking-wider">
        <Icon className="h-3.5 w-3.5 text-accent" />
        {title}
      </div>
      <div className="rounded-lg border border-border bg-background/50 p-3 space-y-2 text-xs">
        {children}
      </div>
    </div>
  );
}

function Row({ label, value, estimated }: { label: string; value: string; estimated?: boolean }) {
  if (!value) return null;
  return (
    <div className="flex justify-between gap-2">
      <span className="text-muted-foreground shrink-0">{label}:</span>
      <span className="text-foreground text-right">
        {value}
        {estimated && (
          <span className="text-muted-foreground/50 text-[10px] ml-1">(estimated)</span>
        )}
      </span>
    </div>
  );
}

export function FIRDraftPreview({ open, onClose, draftData }: FIRDraftPreviewProps) {
  return (
    <AnimatePresence>
      {open && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 z-40 bg-black/60 backdrop-blur-sm"
          />

          {/* Drawer */}
          <motion.div
            initial={{ x: "100%" }}
            animate={{ x: 0 }}
            exit={{ x: "100%" }}
            transition={{ type: "spring", damping: 25, stiffness: 200 }}
            className="fixed right-0 top-0 bottom-0 z-50 w-full max-w-md bg-sidebar border-l border-sidebar-border overflow-y-auto"
          >
            <div className="p-5 space-y-5">
              {/* Header */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <FileText className="h-5 w-5 text-accent" />
                  <h2 className="text-lg font-bold text-foreground">Complaint Draft Preview</h2>
                </div>
                <button
                  onClick={onClose}
                  className="p-1.5 rounded-md hover:bg-sidebar-accent text-muted-foreground hover:text-foreground transition-colors"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>

              {/* Disclaimer */}
              <div className="rounded-lg border border-yellow-500/20 bg-yellow-500/5 p-3 text-xs text-yellow-300/80">
                This is a structured complaint preview generated from your details. It is not legal advice
                and does not constitute a filed FIR. Review the details before submitting.
              </div>

              {/* Reporter Details */}
              <Section icon={User} title="Reporter Details">
                <Row label="Name" value={draftData.reporterName} />
                <Row label="Phone" value={draftData.reporterPhone} />
                <Row label="Email" value={draftData.reporterEmail} />
                {!draftData.reporterName && !draftData.reporterPhone && (
                  <p className="text-muted-foreground/50 italic">No reporter details provided yet</p>
                )}
              </Section>

              {/* Incident Summary */}
              <Section icon={FileText} title="Incident Summary">
                <p className="text-foreground leading-relaxed">
                  {draftData.complaintText || "No description provided yet."}
                </p>
              </Section>

              {/* Incident Details */}
              <Section icon={MapPin} title="Incident Details">
                <Row label="Location" value={draftData.incidentLocation} />
                <Row label="Time" value={draftData.incidentTime} />
                {draftData.personsInvolved.length > 0 && (
                  <div>
                    <span className="text-muted-foreground">Persons Involved:</span>
                    <ul className="mt-1 space-y-0.5 ml-2">
                      {draftData.personsInvolved.map((p, i) => (
                        <li key={i} className="text-foreground">
                          {p}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </Section>

              {/* Risk Information */}
              <Section icon={AlertTriangle} title="Risk Information">
                <Row label="Category" value={draftData.estimatedCategory} estimated />
                <div className="flex justify-between gap-2">
                  <span className="text-muted-foreground shrink-0">Priority:</span>
                  <span
                    className={`inline-flex px-2 py-0.5 rounded text-[10px] font-medium border ${
                      PRIORITY_COLORS[draftData.estimatedPriority] || PRIORITY_COLORS.Low
                    }`}
                  >
                    {draftData.estimatedPriority} (estimated)
                  </span>
                </div>
                {draftData.estimatedRiskFlags.length > 0 && draftData.estimatedRiskFlags[0] !== "none_identified" && (
                  <div className="flex flex-wrap gap-1 mt-1">
                    {draftData.estimatedRiskFlags.map((flag) => (
                      <span
                        key={flag}
                        className="inline-flex px-1.5 py-0.5 rounded text-[10px] bg-muted text-muted-foreground border border-border"
                      >
                        {flag.replace(/_/g, " ")}
                      </span>
                    ))}
                  </div>
                )}
              </Section>

              {/* Officer Routing */}
              <Section icon={Shield} title="Officer Routing">
                <Row label="Assigned Unit" value={draftData.estimatedAssignedUnit} estimated />
                <Row label="Action" value={draftData.estimatedRecommendedAction} estimated />
              </Section>

              {/* Footer */}
              <div className="flex items-center gap-2 pt-2">
                <button
                  onClick={onClose}
                  className="flex-1 px-4 py-2.5 rounded-lg border border-border bg-background text-sm text-foreground hover:bg-sidebar-accent transition-colors"
                >
                  Close Preview
                </button>
                <div className="flex items-center gap-1 text-xs text-muted-foreground">
                  <ArrowRight className="h-3 w-3" />
                  <span>File when ready</span>
                </div>
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
