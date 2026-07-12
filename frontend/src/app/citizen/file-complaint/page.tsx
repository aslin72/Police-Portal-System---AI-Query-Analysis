"use client";

import { useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { useDropzone } from "react-dropzone";
import { AlertCircle, CheckCircle, FileWarning, Upload, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import {
  Form,
  FormField,
  FormItem,
  FormLabel,
  FormControl,
  FormMessage,
} from "@/components/ui/form";
import { Skeleton } from "@/components/ui/skeleton";
import {
  submitComplaint,
  uploadEvidence,
  isAllowedFileType,
  isAllowedFileSize,
  type Complaint,
} from "@/lib/types"
import { getPriorityBadgeProps } from "@/lib/badge-utils";

const formSchema = z.object({
  complaint_text: z.string().min(10, "Please provide at least 10 characters"),
  reporter_name: z.string().optional(),
  reporter_phone: z.string().optional(),
  reporter_email: z.string().email("Invalid email").optional().or(z.literal("")),
  incident_location: z.string().optional(),
  incident_time: z.string().optional(),
});

type FormValues = z.infer<typeof formSchema>;

export default function FileComplaintPage() {
  const [result, setResult] = useState<Complaint | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [evidenceFiles, setEvidenceFiles] = useState<File[]>([]);
  const [evidenceErrors, setEvidenceErrors] = useState<string[]>([]);
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState<string | null>(null);

  const form = useForm<FormValues>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      complaint_text: "",
      reporter_name: "",
      reporter_phone: "",
      reporter_email: "",
      incident_location: "",
      incident_time: "",
    },
  });

  const onDrop = (accepted: File[], rejected: { file: File; errors: readonly { message: string }[] }[]) => {
    const newFiles: File[] = [];
    const errors: string[] = [];
    for (const f of accepted) {
      if (!isAllowedFileType(f)) {
        errors.push(`${f.name}: Unsupported file type`);
      } else if (!isAllowedFileSize(f)) {
        errors.push(`${f.name}: Exceeds 10 MB limit`);
      } else {
        newFiles.push(f);
      }
    }
    for (const r of rejected) {
      errors.push(`${r.file.name}: ${r.errors[0]?.message || "Invalid file"}`);
    }
    setEvidenceFiles((prev) => [...prev, ...newFiles]);
    setEvidenceErrors((prev) => [...prev, ...errors]);
  };

  const removeFile = (idx: number) => {
    setEvidenceFiles((prev) => prev.filter((_, i) => i !== idx));
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/jpeg": [".jpg", ".jpeg"], "image/png": [".png"], "application/pdf": [".pdf"], "text/plain": [".txt"] },
    maxSize: 10 * 1024 * 1024,
  });

  async function onSubmit(values: FormValues) {
    setLoading(true);
    setError(null);
    setResult(null);
    setUploadResult(null);

    try {
      const payload: Record<string, string | null> = {
        complaint_text: values.complaint_text,
      };
      if (values.reporter_name) payload.reporter_name = values.reporter_name;
      if (values.reporter_phone) payload.reporter_phone = values.reporter_phone;
      if (values.reporter_email) payload.reporter_email = values.reporter_email;
      if (values.incident_location) payload.incident_location = values.incident_location;
      if (values.incident_time) payload.incident_time = values.incident_time;

      const complaint = await submitComplaint(payload as unknown as { complaint_text: string; reporter_name?: string | null; reporter_phone?: string | null; reporter_email?: string | null; incident_location?: string | null; incident_time?: string | null });
      setResult(complaint);

      if (evidenceFiles.length > 0) {
        setUploading(true);
        try {
          const ev = await uploadEvidence(complaint.id, evidenceFiles);
          setUploadResult(`Successfully uploaded ${ev.uploaded.length} file(s).`);
        } catch (e: unknown) {
          const msg = e instanceof Error ? e.message : String(e);
          setUploadResult(`Evidence upload failed: ${msg}. Complaint saved.`);
        }
        setUploading(false);
      }
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      setError(msg || "Failed to submit complaint");
    }
    setLoading(false);
  }

  return (
    <div className="mx-auto max-w-3xl px-4 py-8">
      <h2 className="text-2xl font-bold mb-1">File a Complaint</h2>
      <p className="text-muted-foreground mb-6">
        Describe the incident in detail. AI will classify and route your complaint automatically.
      </p>

      <Card className="border-border">
        <CardHeader>
          <CardTitle className="text-lg">Incident Details</CardTitle>
        </CardHeader>
        <CardContent>
          <Form {...form}>
            <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
              <FormField
                control={form.control}
                name="complaint_text"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Complaint Description *</FormLabel>
                    <FormControl>
                      <Textarea
                        placeholder="Describe what happened in detail..."
                        className="min-h-32"
                        {...field}
                      />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />

              <div className="grid gap-4 sm:grid-cols-2">
                <FormField
                  control={form.control}
                  name="reporter_name"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Your Name (optional)</FormLabel>
                      <FormControl>
                        <Input placeholder="Full name" {...field} />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <FormField
                  control={form.control}
                  name="reporter_phone"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Phone (optional)</FormLabel>
                      <FormControl>
                        <Input placeholder="Phone number" {...field} />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
              </div>

              <FormField
                control={form.control}
                name="reporter_email"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Email (optional)</FormLabel>
                    <FormControl>
                      <Input type="email" placeholder="Email address" {...field} />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />

              <div className="grid gap-4 sm:grid-cols-2">
                <FormField
                  control={form.control}
                  name="incident_location"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Incident Location (optional)</FormLabel>
                      <FormControl>
                        <Input placeholder="e.g. Jalan Tun Razak, KL" {...field} />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <FormField
                  control={form.control}
                  name="incident_time"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Incident Time (optional)</FormLabel>
                      <FormControl>
                        <Input placeholder="e.g. July 12, 2026 around 3pm" {...field} />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
              </div>

              {/* Evidence Dropzone */}
              <div>
                <FormLabel className="mb-2 block">Evidence Files (optional)</FormLabel>
                <div
                  {...getRootProps()}
                  className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors ${
                    isDragActive
                      ? "border-accent bg-accent/5"
                      : "border-border hover:border-muted-foreground/50"
                  }`}
                >
                  <input {...getInputProps()} />
                  <Upload className="mx-auto h-8 w-8 text-muted-foreground mb-2" />
                  <p className="text-sm text-muted-foreground">
                    Drag & drop files here, or click to browse
                  </p>
                  <p className="text-xs text-muted-foreground mt-1">
                    Accepted: JPG, PNG, PDF, TXT (max 10 MB each)
                  </p>
                </div>

                {evidenceErrors.length > 0 && (
                  <div className="mt-2 space-y-1">
                    {evidenceErrors.map((err, i) => (
                      <p key={i} className="text-xs text-destructive flex items-center gap-1">
                        <AlertCircle className="h-3 w-3" /> {err}
                      </p>
                    ))}
                  </div>
                )}

                {evidenceFiles.length > 0 && (
                  <div className="mt-2 space-y-1">
                    {evidenceFiles.map((f, i) => (
                      <div key={i} className="flex items-center justify-between text-sm bg-muted px-3 py-1.5 rounded">
                        <span className="truncate">{f.name} ({(f.size / 1024).toFixed(0)} KB)</span>
                        <Button
                          type="button"
                          variant="ghost"
                          size="icon"
                          className="h-6 w-6"
                          onClick={() => removeFile(i)}
                        >
                          <X className="h-3 w-3" />
                        </Button>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {error && (
                <div className="flex items-center gap-2 text-sm text-destructive bg-destructive/10 rounded-lg px-3 py-2">
                  <AlertCircle className="h-4 w-4" />
                  {error}
                </div>
              )}

              <Button type="submit" disabled={loading} size="lg" className="w-full">
                {loading ? "Submitting..." : "Submit Complaint"}
              </Button>
            </form>
          </Form>
        </CardContent>
      </Card>

      {loading && (
        <Card className="border-border mt-6">
          <CardContent className="pt-6 space-y-3">
            <Skeleton className="h-4 w-3/4" />
            <Skeleton className="h-4 w-1/2" />
            <Skeleton className="h-4 w-2/3" />
            <Skeleton className="h-20 w-full" />
          </CardContent>
        </Card>
      )}

      {result && (
        <Card className="border-border mt-6">
          <CardHeader>
            <div className="flex items-center gap-2">
              <CheckCircle className="h-5 w-5 text-accent" />
              <CardTitle>Complaint #{result.id} Filed</CardTitle>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid gap-3 sm:grid-cols-3">
              <div>
                <p className="text-xs text-muted-foreground">Category</p>
                <p className="font-medium capitalize">{result.category}</p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground">Priority</p>
                <Badge {...getPriorityBadgeProps(result.priority)}>
                  {result.priority}
                </Badge>
              </div>
              <div>
                <p className="text-xs text-muted-foreground">Assigned Unit</p>
                <p className="font-medium">{result.assigned_unit || "-"}</p>
              </div>
            </div>

            <Separator />

            <div>
              <p className="text-xs text-muted-foreground">AI Summary</p>
              <p className="text-sm">{result.summary}</p>
            </div>

            <div className="grid gap-2 sm:grid-cols-2">
              <div>
                <p className="text-xs text-muted-foreground">AI Location</p>
                <p className="text-sm">{result.location}</p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground">AI Time</p>
                <p className="text-sm">{result.incident_time}</p>
              </div>
            </div>

            {result.risk_flags && result.risk_flags.length > 0 && result.risk_flags[0] !== "none_identified" && (
              <div>
                <p className="text-xs text-muted-foreground mb-1">Risk Flags</p>
                <div className="flex flex-wrap gap-1">
                  {result.risk_flags.map((flag) => (
                    <Badge key={flag} variant="outline" className="text-xs">
                      {flag.replace(/_/g, " ")}
                    </Badge>
                  ))}
                </div>
              </div>
            )}

            {result.followup_questions && result.followup_questions.length > 0 && (
              <div>
                <p className="text-xs text-muted-foreground mb-2">Follow-Up Questions</p>
                <ul className="space-y-1">
                  {result.followup_questions.map((q, i) => (
                    <li key={i} className="text-sm flex items-start gap-2">
                      <span className="text-muted-foreground mt-1">•</span>
                      {q}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {uploading && (
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Skeleton className="h-4 w-4 rounded-full" />
                Uploading evidence...
              </div>
            )}

            {uploadResult && (
              <div className={`flex items-center gap-2 text-sm px-3 py-2 rounded-lg ${
                uploadResult.includes("Success") ? "bg-accent/10 text-accent" : "bg-destructive/10 text-destructive"
              }`}>
                <FileWarning className="h-4 w-4" />
                {uploadResult}
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}
