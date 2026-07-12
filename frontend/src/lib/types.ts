export const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface Complaint {
  id: number;
  complaint_text: string;
  category: string;
  location: string;
  incident_time: string;
  persons_involved: string[];
  summary: string;
  priority: string;
  followup_questions: string[];
  reporter_name: string | null;
  reporter_phone: string | null;
  reporter_email: string | null;
  citizen_incident_location: string | null;
  citizen_incident_time: string | null;
  status: string;
  assigned_unit: string | null;
  triage_reason: string | null;
  risk_flags: string[];
  recommended_action: string | null;
  officer_notes: string | null;
  created_at: string | null;
  updated_at: string | null;
}

export interface ComplaintPayload {
  complaint_text: string;
  reporter_name?: string | null;
  reporter_phone?: string | null;
  reporter_email?: string | null;
  incident_location?: string | null;
  incident_time?: string | null;
}

export interface TriageUpdate {
  status?: string | null;
  officer_notes?: string | null;
}

export interface EvidenceRecord {
  id: number;
  complaint_id: number;
  original_filename: string;
  stored_filename: string;
  content_type: string | null;
  file_size: number;
  uploaded_at: string;
}

export interface EvidenceUploadResult {
  uploaded: {
    id: number;
    complaint_id: number;
    original_filename: string;
    stored_filename: string;
    file_size: number;
  }[];
}

const ALLOWED_EXTENSIONS = ["jpg", "jpeg", "png", "pdf", "txt"];
const MAX_FILE_SIZE = 10 * 1024 * 1024;

export function isAllowedFileType(file: File): boolean {
  const ext = file.name.split(".").pop()?.toLowerCase();
  return ext ? ALLOWED_EXTENSIONS.includes(ext) : false;
}

export function isAllowedFileSize(file: File): boolean {
  return file.size <= MAX_FILE_SIZE;
}

export async function submitComplaint(
  payload: ComplaintPayload,
): Promise<Complaint> {
  const res = await fetch(`${API_URL}/complaints`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(err);
  }
  return res.json();
}

export async function getComplaints(): Promise<Complaint[]> {
  const res = await fetch(`${API_URL}/complaints`);
  if (!res.ok) throw new Error("Failed to fetch complaints");
  return res.json();
}

export async function getComplaint(id: number): Promise<Complaint> {
  const res = await fetch(`${API_URL}/complaints/${id}`);
  if (!res.ok) {
    if (res.status === 404) throw new Error("Complaint not found");
    throw new Error("Failed to fetch complaint");
  }
  return res.json();
}

export async function updateTriage(
  id: number,
  update: TriageUpdate,
): Promise<Complaint> {
  const res = await fetch(`${API_URL}/complaints/${id}/triage`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(update),
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(err);
  }
  return res.json();
}

export async function uploadEvidence(
  complaintId: number,
  files: File[],
): Promise<EvidenceUploadResult> {
  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));
  const res = await fetch(
    `${API_URL}/complaints/${complaintId}/evidence`,
    { method: "POST", body: formData },
  );
  if (!res.ok) {
    const err = await res.text();
    throw new Error(err);
  }
  return res.json();
}

export async function getEvidenceByComplaint(
  complaintId: number,
): Promise<{ evidence: EvidenceRecord[] }> {
  const res = await fetch(
    `${API_URL}/complaints/${complaintId}/evidence`,
  );
  if (!res.ok) throw new Error("Failed to fetch evidence");
  return res.json();
}

export function getEvidenceDownloadUrl(
  evidenceId: number,
  apiUrl?: string,
): string {
  return `${apiUrl || API_URL}/evidence/${evidenceId}`;
}
