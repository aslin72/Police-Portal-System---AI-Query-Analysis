"use client";

import { useEffect, useState } from "react";
import { FileSearch, Download, AlertCircle, FileText } from "lucide-react";
import { buttonVariants } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { getComplaints, getEvidenceByComplaint, getEvidenceDownloadUrl, type EvidenceRecord } from "@/lib/types";
import { cn } from "@/lib/utils";

interface EvidenceWithComplaint extends EvidenceRecord {
  complaintId: number;
  complaintText: string;
  category: string;
}

export default function EvidenceReviewPage() {
  const [allEvidence, setAllEvidence] = useState<EvidenceWithComplaint[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      try {
        const complaints = await getComplaints();
        const results: EvidenceWithComplaint[] = [];
        for (const c of complaints) {
          try {
            const ev = await getEvidenceByComplaint(c.id);
            for (const record of ev.evidence || []) {
              results.push({
                ...record,
                complaintId: c.id,
                complaintText: c.summary || c.complaint_text,
                category: c.category,
              });
            }
          } catch {
            // complaint has no evidence
          }
        }
        setAllEvidence(results);
      } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : String(e);
        setError(msg);
      }
      setLoading(false);
    }
    load();
  }, []);

  return (
    <div className="mx-auto max-w-5xl px-4 py-8">
      <div className="flex items-center gap-2 mb-6">
        <FileSearch className="h-6 w-6 text-accent" />
        <h2 className="text-2xl font-bold">Evidence Review</h2>
        <Badge variant="outline" className="ml-auto">{allEvidence.length} files</Badge>
      </div>

      {error && (
        <div className="flex items-center gap-2 text-sm text-destructive mb-4">
          <AlertCircle className="h-4 w-4" />
          {error}
        </div>
      )}

      {loading ? (
        <div className="space-y-3">
          {[1, 2, 3].map((i) => (
            <Skeleton key={i} className="h-16 w-full" />
          ))}
        </div>
      ) : allEvidence.length === 0 ? (
        <Card className="border-border">
          <CardContent className="py-8 text-center text-muted-foreground">
            No evidence files have been uploaded yet.
          </CardContent>
        </Card>
      ) : (
        <Card className="border-border">
          <CardHeader>
            <CardTitle className="text-base">All Evidence Files</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="divide-y divide-border">
              {allEvidence.map((ev) => (
                <div key={ev.id} className="flex items-center gap-4 py-3 first:pt-0 last:pb-0">
                  <FileText className="h-8 w-8 text-muted-foreground flex-shrink-0" />
                  <div className="min-w-0 flex-1">
                    <p className="text-sm font-medium truncate">{ev.original_filename}</p>
                    <div className="flex items-center gap-2 mt-0.5">
                      <Badge variant="outline" className="text-xs">#{ev.complaintId}</Badge>
                      <span className="text-xs text-muted-foreground capitalize">{ev.category}</span>
                      <span className="text-xs text-muted-foreground">
                        {(ev.file_size / 1024).toFixed(0)} KB
                      </span>
                      <span className="text-xs text-muted-foreground">
                        {ev.uploaded_at?.slice(0, 16) || "-"}
                      </span>
                    </div>
                    <p className="text-xs text-muted-foreground mt-0.5 truncate">{ev.complaintText}</p>
                  </div>
                  <a
                    href={getEvidenceDownloadUrl(ev.id)}
                    target="_blank"
                    rel="noopener noreferrer"
                    className={cn(
                      buttonVariants({ variant: "outline", size: "sm" }),
                      "gap-1 flex-shrink-0",
                    )}
                  >
                    <Download className="h-3 w-3" />
                    Download
                  </a>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
