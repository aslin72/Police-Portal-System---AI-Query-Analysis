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

// Chat API Types and Functions
export interface ChatMessage {
  role: "user" | "agent";
  content: string;
  timestamp?: string;
  extracted_data?: Record<string, unknown>;
}

export interface ChatSession {
  id: string;
  created_at: string;
  updated_at: string;
  is_filed: boolean;
  complaint_id?: number;
  complaint_id_fk?: number | null;
  title?: string | null;
}

export interface ChatRequest {
  session_id: string;
  user_message: string;
}

export interface ChatResponse {
  session_id: string;
  agent_message: string;
  suggested_followups?: string[];
  collected_fields?: Record<string, unknown>;
  ready_to_file?: boolean;
}

export type ChatStreamEvent =
  | { event: "status"; message: string }
  | { event: "metadata"; data: Record<string, unknown> }
  | { event: "final"; data: ChatResponse }
  | { event: "error"; message: string };

function parseSseBlock(block: string): { event: string; data: unknown } | null {
  const lines = block.split("\n");
  let event = "message";
  const dataLines: string[] = [];

  for (const line of lines) {
    if (!line || line.startsWith(":")) continue;
    if (line.startsWith("event:")) {
      event = line.slice("event:".length).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice("data:".length).trimStart());
    }
  }

  if (dataLines.length === 0) return null;

  return {
    event,
    data: JSON.parse(dataLines.join("\n")),
  };
}

function getStreamMessage(data: unknown, fallback: string): string {
  if (data && typeof data === "object" && "message" in data) {
    const value = (data as { message?: unknown }).message;
    if (typeof value === "string" && value.trim()) return value;
  }
  return fallback;
}

function isChatResponse(data: unknown): data is ChatResponse {
  return (
    !!data &&
    typeof data === "object" &&
    typeof (data as ChatResponse).session_id === "string" &&
    typeof (data as ChatResponse).agent_message === "string"
  );
}

// The backend reconstructs conversation context from its own message
// history for each session_id, so only the new message needs to be sent.
export async function chatComplaintStream(
  sessionId: string,
  userMessage: string,
  onEvent?: (event: ChatStreamEvent) => void,
): Promise<ChatResponse> {
  const res = await fetch(`${API_URL}/chat/complaint`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId,
      user_message: userMessage,
    }),
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(err);
  }

  if (!res.body) {
    throw new Error("Chat stream response was empty");
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finalResponse: ChatResponse | null = null;

  const dispatchBlock = (block: string) => {
    if (!block.trim()) return;

    const parsed = parseSseBlock(block);
    if (!parsed) return;

    if (parsed.event === "status") {
      onEvent?.({
        event: "status",
        message: getStreamMessage(parsed.data, "Processing complaint details"),
      });
      return;
    }

    if (parsed.event === "metadata" && parsed.data && typeof parsed.data === "object") {
      onEvent?.({
        event: "metadata",
        data: parsed.data as Record<string, unknown>,
      });
      return;
    }

    if (parsed.event === "final") {
      if (!isChatResponse(parsed.data)) {
        throw new Error("Chat stream final response was invalid");
      }
      finalResponse = parsed.data;
      onEvent?.({ event: "final", data: parsed.data });
      return;
    }

    if (parsed.event === "error") {
      const message = getStreamMessage(parsed.data, "Failed to process chat message");
      onEvent?.({ event: "error", message });
      throw new Error(message);
    }
  };

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");

    let separatorIndex = buffer.indexOf("\n\n");
    while (separatorIndex !== -1) {
      const block = buffer.slice(0, separatorIndex);
      buffer = buffer.slice(separatorIndex + 2);
      dispatchBlock(block);
      separatorIndex = buffer.indexOf("\n\n");
    }
  }

  buffer += decoder.decode().replace(/\r\n/g, "\n");
  if (buffer.trim()) {
    dispatchBlock(buffer);
  }

  if (!finalResponse) {
    throw new Error("Chat stream ended before final response");
  }

  return finalResponse;
}

export async function getChatSession(
  sessionId: string,
): Promise<{ session: ChatSession; messages: ChatMessage[] }> {
  const res = await fetch(`${API_URL}/chat/complaint/${sessionId}`);
  if (!res.ok) throw new Error("Failed to fetch chat session");
  return res.json();
}

export async function listChatSessions(): Promise<{ sessions: ChatSession[] }> {
  const res = await fetch(`${API_URL}/chat/complaint`);
  if (!res.ok) throw new Error("Failed to fetch chat sessions");
  return res.json();
}

export async function deleteChatSession(sessionId: string): Promise<void> {
  const res = await fetch(`${API_URL}/chat/complaint/${sessionId}`, {
    method: "DELETE",
  });
  if (!res.ok) throw new Error("Failed to delete chat session");
}

export async function fileComplaintFromChat(
  sessionId: string,
): Promise<Complaint> {
  const res = await fetch(`${API_URL}/chat/complaint/${sessionId}/file`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(err);
  }
  return res.json();
}
