"use client";

import { useEffect, useRef, useState } from "react";
import { use } from "react";
import Link from "next/link";
import { ArrowLeft, AlertCircle, FileText, Download } from "lucide-react";
import { Button, buttonVariants } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  getComplaint,
  getEvidenceByComplaint,
  getEvidenceDownloadUrl,
  updateTriage,
  type Complaint,
  type EvidenceRecord,
} from "@/lib/types"
import { getPriorityBadgeProps } from "@/lib/badge-utils";
import { cn } from "@/lib/utils";

const STATUSES = ["New", "Under Review", "Assigned", "Resolved", "Closed"];

export default function ComplaintDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);
  const complaintId = parseInt(id);
  const [complaint, setComplaint] = useState<Complaint | null>(null);
  const [evidence, setEvidence] = useState<EvidenceRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState("");
  const [notes, setNotes] = useState("");
  const [updating, setUpdating] = useState(false);
  const [updateMsg, setUpdateMsg] = useState<{ type: "success" | "error"; text: string } | null>(null);
  const initialLoad = useRef(true);

  useEffect(() => {
    async function load() {
      try {
        const [comp, ev] = await Promise.all([
          getComplaint(complaintId),
          getEvidenceByComplaint(complaintId).catch(() => ({ evidence: [] })),
        ]);
        setComplaint(comp);
        setEvidence(ev.evidence || []);
        if (initialLoad.current) {
          setStatus(comp.status);
          setNotes(comp.officer_notes || "");
          initialLoad.current = false;
        }
      } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : String(e);
        setError(msg);
      }
      setLoading(false);
    }
    load();
  }, [complaintId]);

  async function handleUpdate() {
    setUpdating(true);
    setUpdateMsg(null);
    try {
      const updated = await updateTriage(complaintId, {
        status: status || null,
        officer_notes: notes,
      });
      setComplaint(updated);
      setUpdateMsg({ type: "success", text: "Complaint updated successfully." });
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      setUpdateMsg({ type: "error", text: msg });
    }
    setUpdating(false);
  }

  if (loading) {
    return (
      <div className="mx-auto max-w-4xl px-4 py-8 space-y-3">
        <Skeleton className="h-8 w-64" />
        <Skeleton className="h-4 w-96" />
        <Skeleton className="h-48 w-full" />
      </div>
    );
  }

  if (error || !complaint) {
    return (
      <div className="mx-auto max-w-4xl px-4 py-8">
        <div className="flex items-center gap-2 text-destructive mb-4">
          <AlertCircle className="h-5 w-5" />
          <p>{error || "Complaint not found"}</p>
        </div>
        <Link href="/officer/dashboard" className={buttonVariants({ variant: "outline" })}>
          Back to Dashboard
        </Link>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-4xl px-4 py-8">
      <div className="flex items-center gap-3 mb-6">
        <Link
          href="/officer/dashboard"
          aria-label="Back to dashboard"
          className={buttonVariants({ variant: "ghost", size: "icon" })}
        >
          <ArrowLeft className="h-5 w-5" />
        </Link>
        <h2 className="text-2xl font-bold">Complaint #{complaint.id}</h2>
        <Badge {...getPriorityBadgeProps(complaint.priority)} className="ml-auto">
          {complaint.priority}
        </Badge>
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        {/* Main Content */}
        <div className="lg:col-span-2 space-y-6">
          {/* Original Complaint */}
          <Card className="border-border">
            <CardHeader className="pb-3">
              <CardTitle className="text-base">Original Complaint</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm whitespace-pre-wrap">{complaint.complaint_text}</p>
            </CardContent>
          </Card>

          {/* AI Analysis */}
          <Card className="border-border">
            <CardHeader className="pb-3">
              <CardTitle className="text-base">AI Analysis</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div>
                <p className="text-xs text-muted-foreground">Summary</p>
                <p className="text-sm">{complaint.summary}</p>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <p className="text-xs text-muted-foreground">Location</p>
                  <p className="text-sm">{complaint.location}</p>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground">Incident Time</p>
                  <p className="text-sm">{complaint.incident_time}</p>
                </div>
              </div>
              <div>
                <p className="text-xs text-muted-foreground">Persons Involved</p>
                <p className="text-sm">
                  {complaint.persons_involved?.length
                    ? complaint.persons_involved.join(", ")
                    : "None identified"}
                </p>
              </div>
            </CardContent>
          </Card>

          {/* Reporter & Citizen Info */}
          {(complaint.reporter_name || complaint.reporter_phone || complaint.reporter_email) && (
            <Card className="border-border">
              <CardHeader className="pb-3">
                <CardTitle className="text-base">Reporter Information</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid gap-2 sm:grid-cols-3">
                  {complaint.reporter_name && (
                    <div>
                      <p className="text-xs text-muted-foreground">Name</p>
                      <p className="text-sm">{complaint.reporter_name}</p>
                    </div>
                  )}
                  {complaint.reporter_phone && (
                    <div>
                      <p className="text-xs text-muted-foreground">Phone</p>
                      <p className="text-sm">{complaint.reporter_phone}</p>
                    </div>
                  )}
                  {complaint.reporter_email && (
                    <div>
                      <p className="text-xs text-muted-foreground">Email</p>
                      <p className="text-sm">{complaint.reporter_email}</p>
                    </div>
                  )}
                </div>
                <div className="grid grid-cols-2 gap-3 mt-3">
                  {complaint.citizen_incident_location && (
                    <div>
                      <p className="text-xs text-muted-foreground">Citizen Location</p>
                      <p className="text-sm">{complaint.citizen_incident_location}</p>
                    </div>
                  )}
                  {complaint.citizen_incident_time && (
                    <div>
                      <p className="text-xs text-muted-foreground">Citizen Time</p>
                      <p className="text-sm">{complaint.citizen_incident_time}</p>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          )}

          {/* Evidence */}
          <Card className="border-border">
            <CardHeader className="pb-3">
              <CardTitle className="text-base">Evidence ({evidence.length})</CardTitle>
            </CardHeader>
            <CardContent>
              {evidence.length === 0 ? (
                <p className="text-sm text-muted-foreground">No evidence uploaded.</p>
              ) : (
                <div className="space-y-2">
                  {evidence.map((ev) => (
                    <div key={ev.id} className="flex items-center justify-between text-sm bg-muted px-3 py-2 rounded">
                      <div className="flex items-center gap-2 truncate">
                        <FileText className="h-4 w-4 text-muted-foreground flex-shrink-0" />
                        <span className="truncate">{ev.original_filename}</span>
                        <span className="text-xs text-muted-foreground">
                          ({(ev.file_size / 1024).toFixed(0)} KB)
                        </span>
                      </div>
                      <a
                        href={getEvidenceDownloadUrl(ev.id)}
                        target="_blank"
                        rel="noopener noreferrer"
                        aria-label={`Download ${ev.original_filename}`}
                        className={cn(
                          buttonVariants({ variant: "ghost", size: "icon" }),
                          "h-7 w-7",
                        )}
                      >
                        <Download className="h-4 w-4" />
                      </a>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>

          {/* Follow-up Questions */}
          {complaint.followup_questions && complaint.followup_questions.length > 0 && (
            <Card className="border-border">
              <CardHeader className="pb-3">
                <CardTitle className="text-base">Follow-Up Questions</CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="space-y-1">
                  {complaint.followup_questions.map((q, i) => (
                    <li key={i} className="text-sm flex items-start gap-2">
                      <span className="text-muted-foreground mt-1">•</span>
                      {q}
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>
          )}
        </div>

        {/* Sidebar */}
        <div className="space-y-4">
          {/* Status & Triage */}
          <Card className="border-border">
            <CardHeader className="pb-3">
              <CardTitle className="text-base">Triage Details</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div>
                <p className="text-xs text-muted-foreground">Assigned Unit</p>
                <p className="font-medium">{complaint.assigned_unit || "-"}</p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground">Category</p>
                <p className="text-sm capitalize">{complaint.category}</p>
              </div>
              {complaint.triage_reason && (
                <div>
                  <p className="text-xs text-muted-foreground">Triage Reason</p>
                  <p className="text-xs">{complaint.triage_reason}</p>
                </div>
              )}
              {complaint.risk_flags && complaint.risk_flags.length > 0 && complaint.risk_flags[0] !== "none_identified" && (
                <div>
                  <p className="text-xs text-muted-foreground mb-1">Risk Flags</p>
                  <div className="flex flex-wrap gap-1">
                    {complaint.risk_flags.map((flag) => (
                      <Badge key={flag} variant="outline" className="text-xs">
                        {flag.replace(/_/g, " ")}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}
              {complaint.recommended_action && (
                <div className="bg-muted rounded-lg p-2 text-xs">
                  {complaint.recommended_action}
                </div>
              )}
            </CardContent>
          </Card>

          {/* Update Status */}
          <Card className="border-border">
            <CardHeader className="pb-3">
              <CardTitle className="text-base">Update</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div>
                <p className="text-xs text-muted-foreground mb-1">Status</p>
                <Select value={status} onValueChange={(value: string | null) => setStatus(value ?? "")}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {STATUSES.map((s) => (
                      <SelectItem key={s} value={s}>{s}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div>
                <p className="text-xs text-muted-foreground mb-1">Officer Notes</p>
                <Textarea
                  value={notes}
                  onChange={(e) => setNotes(e.target.value)}
                  placeholder="Add notes..."
                  rows={3}
                />
              </div>

              {updateMsg && (
                <div className={`text-xs px-3 py-2 rounded ${
                  updateMsg.type === "success" ? "bg-accent/10 text-accent" : "bg-destructive/10 text-destructive"
                }`}>
                  {updateMsg.text}
                </div>
              )}

              <Button
                onClick={handleUpdate}
                disabled={updating}
                className="w-full"
                size="sm"
              >
                {updating ? "Updating..." : "Update Complaint"}
              </Button>
            </CardContent>
          </Card>

          {/* Timestamps */}
          <Card className="border-border">
            <CardContent className="pt-4 space-y-1">
              <div>
                <p className="text-xs text-muted-foreground">Created</p>
                <p className="text-sm">{complaint.created_at || "-"}</p>
              </div>
              {complaint.updated_at && (
                <div>
                  <p className="text-xs text-muted-foreground">Last Updated</p>
                  <p className="text-sm">{complaint.updated_at}</p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
