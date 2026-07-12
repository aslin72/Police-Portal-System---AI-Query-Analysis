"use client";

import { useState } from "react";
import { Search, AlertCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { getComplaint, type Complaint } from "@/lib/types"
import { getPriorityBadgeProps } from "@/lib/badge-utils";

const STATUS_FLOW = ["New", "Under Review", "Assigned", "Resolved", "Closed"];

export default function TrackComplaintPage() {
  const [idInput, setIdInput] = useState("");
  const [complaint, setComplaint] = useState<Complaint | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleTrack() {
    const id = parseInt(idInput.trim());
    if (isNaN(id) || id < 1) {
      setError("Please enter a valid complaint ID");
      return;
    }
    setLoading(true);
    setError(null);
    setComplaint(null);
    try {
      const result = await getComplaint(id);
      setComplaint(result);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      setError(msg || "Complaint not found");
    }
    setLoading(false);
  }

  return (
    <div className="mx-auto max-w-3xl px-4 py-8">
      <h2 className="text-2xl font-bold mb-1">Track Complaint</h2>
      <p className="text-muted-foreground mb-6">
        Enter your complaint ID to check its current status.
      </p>

      <Card className="border-border mb-6">
        <CardContent className="pt-6">
          <div className="flex gap-2">
            <Input
              placeholder="Enter complaint ID (e.g. 42)"
              value={idInput}
              onChange={(e) => setIdInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleTrack()}
              type="number"
              min={1}
              className="max-w-xs"
            />
            <Button onClick={handleTrack} disabled={loading} className="gap-2">
              <Search className="h-4 w-4" />
              Track
            </Button>
          </div>

          {error && (
            <div className="mt-3 flex items-center gap-2 text-sm text-destructive">
              <AlertCircle className="h-4 w-4" />
              {error}
            </div>
          )}
        </CardContent>
      </Card>

      {loading && (
        <Card className="border-border">
          <CardContent className="pt-6 space-y-3">
            <Skeleton className="h-4 w-3/4" />
            <Skeleton className="h-4 w-1/2" />
            <Skeleton className="h-4 w-2/3" />
            <Skeleton className="h-20 w-full" />
          </CardContent>
        </Card>
      )}

      {complaint && (
        <Card className="border-border">
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              Complaint #{complaint.id}
              <Badge {...getPriorityBadgeProps(complaint.priority)}>
                {complaint.priority}
              </Badge>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Status Bar */}
            <div className="flex items-center gap-1 flex-wrap">
              {STATUS_FLOW.map((s, i) => {
                const currentIdx = STATUS_FLOW.indexOf(complaint.status);
                const done = i <= currentIdx;
                return (
                  <div key={s} className="flex items-center gap-1">
                    <Badge
                      variant={done ? "default" : "outline"}
                      className={done ? "bg-primary text-primary-foreground" : ""}
                    >
                      {s}
                    </Badge>
                    {i < STATUS_FLOW.length - 1 && (
                      <span className="text-muted-foreground text-xs">→</span>
                    )}
                  </div>
                );
              })}
            </div>

            <Separator />

            <div className="grid gap-3 sm:grid-cols-2">
              <div>
                <p className="text-xs text-muted-foreground">Category</p>
                <p className="font-medium capitalize">{complaint.category}</p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground">Assigned Unit</p>
                <p className="font-medium">{complaint.assigned_unit || "-"}</p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground">Created</p>
                <p className="text-sm">{complaint.created_at || "-"}</p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground">Last Updated</p>
                <p className="text-sm">{complaint.updated_at || "-"}</p>
              </div>
            </div>

            <Separator />

            <div>
              <p className="text-xs text-muted-foreground">AI Summary</p>
              <p className="text-sm">{complaint.summary}</p>
            </div>

          </CardContent>
        </Card>
      )}
    </div>
  );
}
