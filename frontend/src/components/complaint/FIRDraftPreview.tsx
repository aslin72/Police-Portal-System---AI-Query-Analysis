"use client";

import { FileText, User, MapPin, AlertTriangle, Shield } from "lucide-react";
import type { DraftData } from "@/lib/complaint-intelligence";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

interface FIRDraftPreviewProps {
  open: boolean;
  onClose: () => void;
  draftData: DraftData;
  canFile: boolean;
  isFiling: boolean;
  onFileComplaint: () => void;
}

const PRIORITY_COLORS: Record<string, string> = {
  Emergency: "bg-red-500/15 text-red-300 border-red-500/30",
  High: "bg-orange-500/15 text-orange-300 border-orange-500/30",
  Medium: "bg-yellow-500/15 text-yellow-300 border-yellow-500/30",
  Low: "bg-slate-500/15 text-slate-300 border-slate-500/30",
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

export function FIRDraftPreview({
  open,
  onClose,
  draftData,
  canFile,
  isFiling,
  onFileComplaint,
}: FIRDraftPreviewProps) {
  return (
    <Dialog
      open={open}
      onOpenChange={(nextOpen) => {
        if (!nextOpen) onClose();
      }}
    >
      <DialogContent
        className="!fixed !right-0 !top-0 !bottom-0 !left-auto !h-dvh !w-full !max-w-md !translate-x-0 !translate-y-0 overflow-y-auto rounded-none border-l border-sidebar-border bg-sidebar p-0 text-foreground shadow-2xl shadow-black/30 sm:!max-w-md data-open:slide-in-from-right data-closed:slide-out-to-right"
      >
        <div className="p-5 space-y-5">
          <DialogHeader className="pr-8">
            <div className="flex items-center gap-2">
              <FileText className="h-5 w-5 text-accent" />
              <DialogTitle className="text-lg font-bold text-foreground">
                Complaint Draft Preview
              </DialogTitle>
            </div>
            <DialogDescription>
              Review the structured complaint before filing.
            </DialogDescription>
          </DialogHeader>

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
            <Row
              label="Category"
              value={
                draftData.estimateConfidence === "high"
                  ? draftData.estimatedCategory
                  : "Not enough details to estimate yet"
              }
              estimated
            />
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
        </div>

        <DialogFooter className="sticky bottom-0 mx-0 mb-0 rounded-none border-sidebar-border bg-sidebar/95 backdrop-blur-sm">
          <Button
            type="button"
            variant="outline"
            className="flex-1"
            onClick={onClose}
          >
            Continue Editing
          </Button>
          <Button
            type="button"
            className="flex-1"
            disabled={!canFile || isFiling}
            onClick={onFileComplaint}
          >
            {isFiling ? "Filing..." : "File Complaint"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
