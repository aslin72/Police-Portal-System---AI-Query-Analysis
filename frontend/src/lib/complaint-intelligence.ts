export interface ComplaintInsight {
  missingDetails: MissingField[];
  collectedDetails: CollectedField[];
  criticalMissing: MissingField[];
  emergencyMode: boolean;
  emergencyReason: string | null;
  requiredEmergencyFields: MissingField[];
  draftData: DraftData;
}

export interface MissingField {
  key: string;
  label: string;
  priority: "critical" | "high" | "medium";
  hint: string;
}

export interface CollectedField {
  key: string;
  label: string;
  value: string;
}

export interface DraftData {
  reporterName: string;
  reporterPhone: string;
  reporterEmail: string;
  incidentLocation: string;
  incidentTime: string;
  complaintText: string;
  personsInvolved: string[];
  estimatedCategory: string;
  estimatedPriority: string;
  estimatedRiskFlags: string[];
  estimatedAssignedUnit: string;
  estimatedRecommendedAction: string;
  estimateConfidence: "high" | "low";
  summary: string;
}

const EMERGENCY_KEYWORDS = [
  "missing child", "child missing", "child is missing", "son is missing", "daughter is missing", "minor is missing", "kidnapped", "kidnapping",
  "weapon", "knife", "gun", "shot", "shooting", "stabbed", "stabbing",
  "bleeding", "serious injury", "injured badly",
  "fire", "burning", "trapped", "active fire",
  "active threat", "ongoing assault", "assault",
  "suicide", "self-harm", "end my life", "kill myself",
  "immediate danger", "life threatening", "hostage",
  "murder", "death", "dead body", "found dead",
];

const CATEGORY_KEYWORDS: Record<string, string[]> = {
  "child safety": ["missing child", "child missing", "child is missing", "son is missing", "daughter is missing", "minor missing", "minor is missing", "kidnapped child", "child kidnapped", "abducted child"],
  "cyber crime incident": ["online fraud", "cyber crime", "phishing", "bank account", "upi", "otp", "hacked account", "fake profile"],
  "women help desk": ["domestic violence", "dowry", "stalking", "sexual harassment", "husband beat", "wife assaulted"],
  "public healthcare": ["food poisoning", "water poisoning", "contaminated water", "contaminated food", "disease outbreak", "public healthcare"],
  "road accident": [
    "road accident",
    "hit-and-run",
    "hit and run",
    "car hit",
    "truck hit",
    "bus hit",
    "bike hit",
    "motorcycle accident",
    "scooter accident",
    "traffic collision",
    "vehicle collision",
    "collided",
    "traffic signal",
    "ring road",
  ],
  "murder / serious crime incident": ["murder", "dead body", "found dead", "stabbed", "stabbing", "shot", "gunshot", "killed", "homicide"],
  "fire accident": ["active fire", "house fire", "building fire", "fire accident", "burning", "smoke", "explosion", "gas leak", "blaze"],
};

const PRIORITY_KEYWORDS: Record<string, string[]> = {
  Emergency: ["immediate danger", "life threatening", "hostage", "active fire", "murder", "death", "shooting", "stabbed", "bleeding badly", "trapped"],
  High: ["injury", "injured", "weapon", "knife", "gun", "assault", "missing", "fire"],
  Medium: ["fraud", "scam", "harassment", "stalking", "hit"],
};

const UNIT_MAP: Record<string, string> = {
  "child safety": "Child Protection Desk",
  "cyber crime incident": "Cyber Crime Cell",
  "women help desk": "Women Help Desk",
  "public healthcare": "Public Health Coordination",
  "road accident": "Traffic Police",
  "murder / serious crime incident": "Serious Crime Unit",
  "fire accident": "Fire and Emergency Coordination",
  "general issue recorded": "General Desk",
};

function hasAnyKeyword(text: string, keywords: string[]): boolean {
  const lower = text.toLowerCase();
  return keywords.some((kw) => lower.includes(kw));
}

function extractPersonsInvolved(text: string): string[] {
  const persons: string[] = [];
  const patterns = [
    /(?:victim|suspect|witness|neighbor|neighbour|husband|wife|child|son|daughter|friend|colleague)s?\b/gi,
    /(?:Mr|Mrs|Ms|Dr|Sir|Madam)\.?\s+[A-Z][a-z]+/g,
  ];
  for (const pattern of patterns) {
    const matches = text.match(pattern);
    if (matches) {
      for (const m of matches) {
        const cleaned = m.trim();
        if (!persons.includes(cleaned)) persons.push(cleaned);
      }
    }
  }
  return persons;
}

function estimateCategory(text: string): { category: string; confidence: "high" | "low" } {
  const lower = text.toLowerCase();

  for (const [category, keywords] of Object.entries(CATEGORY_KEYWORDS)) {
    if (hasAnyKeyword(lower, keywords)) {
      return { category, confidence: "high" };
    }
  }

  return { category: "general issue recorded", confidence: "low" };
}

function estimatePriority(text: string, category: string): string {
  const lower = text.toLowerCase();
  if (category === "child safety" && (lower.includes("missing") || lower.includes("kidnapped"))) return "Emergency";
  if (category === "fire accident" && (lower.includes("active fire") || lower.includes("spreading"))) return "Emergency";
  if (hasAnyKeyword(lower, PRIORITY_KEYWORDS["Emergency"])) return "Emergency";
  if (hasAnyKeyword(lower, PRIORITY_KEYWORDS["High"])) return "High";
  if (hasAnyKeyword(lower, PRIORITY_KEYWORDS["Medium"])) return "Medium";
  return "Low";
}

function estimateRiskFlags(text: string, category: string): string[] {
  const flags: string[] = [];
  const lower = text.toLowerCase();

  if (hasAnyKeyword(lower, ["injury", "injured", "bleeding", "wound", "hurt", "hitting", "beating", "abuse", "attacked"])) {
    flags.push("injury_reported");
  }
  if (hasAnyKeyword(lower, ["emergency", "ambulance", "hospital", "urgent"])) {
    flags.push("urgent_medical_attention");
  }
  if (hasAnyKeyword(lower, ["gun", "knife", "weapon", "armed", "stab", "blade"])) {
    flags.push("weapon_involved");
  }
  if (category === "child safety" || hasAnyKeyword(lower, ["child", "kid", "son", "daughter", "minor"])) {
    flags.push("child_involved");
  }
  if (hasAnyKeyword(lower, ["fire", "burning", "smoke", "explosion", "gas leak"])) {
    flags.push("fire_risk");
  }
  if (hasAnyKeyword(lower, ["missing", "disappeared", "not found"])) {
    flags.push("person_missing");
  }
  if (category === "cyber crime incident" || hasAnyKeyword(lower, ["fraud", "scam", "hacked", "phishing"])) {
    flags.push("digital_fraud");
  }

  if (flags.length === 0) flags.push("none_identified");
  return flags;
}

function buildDraftData(
  collectedFields: Record<string, unknown>,
  complaintText: string,
): DraftData {
  const text = complaintText || String(collectedFields.complaint_text || "");
  const { category, confidence } = estimateCategory(text);
  const priority = estimatePriority(text, category);
  const riskFlags = estimateRiskFlags(text, category);
  const persons = extractPersonsInvolved(text);
  const hasUrgent = riskFlags.some((f) =>
    ["injury_reported", "weapon_involved", "fire_risk", "person_missing", "urgent_medical_attention"].includes(f),
  );
  const assignedUnit = confidence === "high" ? UNIT_MAP[category] || "General Desk" : "Not enough details to estimate yet";
  const recommendedAction = confidence === "high"
    ? priority === "Emergency"
      ? "Respond immediately. Dispatch nearest unit and notify emergency services."
      : hasUrgent
        ? "Review immediately and contact emergency response."
        : priority === "High"
          ? "Prioritize review within 1 hour."
          : "Assign for standard processing within 24 hours."
    : "Collect more details before estimating routing.";

  return {
    reporterName: String(collectedFields.reporter_name || ""),
    reporterPhone: String(collectedFields.reporter_phone || ""),
    reporterEmail: String(collectedFields.reporter_email || ""),
    incidentLocation: String(collectedFields.incident_location || ""),
    incidentTime: String(collectedFields.incident_time || ""),
    complaintText: text,
    personsInvolved: persons,
    estimatedCategory: category,
    estimatedPriority: priority,
    estimatedRiskFlags: riskFlags,
    estimatedAssignedUnit: assignedUnit,
    estimatedRecommendedAction: recommendedAction,
    estimateConfidence: confidence,
    summary: text.length > 120 ? text.slice(0, 117) + "..." : text,
  };
}

function addMissingField(fields: MissingField[], field: MissingField): void {
  const existingIndex = fields.findIndex((item) => item.key === field.key);
  if (existingIndex === -1) {
    fields.push(field);
    return;
  }

  if (field.priority === "critical") {
    fields[existingIndex] = field;
  }
}

function hasImmediateDangerStatus(text: string): boolean {
  return hasAnyKeyword(text, [
    "safe now",
    "not in danger",
    "no immediate danger",
    "danger is over",
    "danger has passed",
    "threat stopped",
    "attacker left",
    "attacker ran away",
    "still in danger",
    "danger is still active",
    "attacker is still here",
    "fire is still active",
    "still trapped",
  ]);
}

function hasPeopleAffectedDetails(text: string): boolean {
  return (
    /\b\d+\s+(people|persons|families|children|victims|workers|students|passengers)\b/i.test(text) ||
    hasAnyKeyword(text, [
      "one person",
      "two people",
      "many people",
      "people affected",
      "people trapped",
      "no one else",
      "nobody else",
      "i am alone",
    ])
  );
}

export function deriveComplaintInsight(
  collectedFields: Record<string, unknown>,
  messages: Array<{ role: string; content: string }>,
): ComplaintInsight {
  const complaintText = String(collectedFields.complaint_text || "");
  const userMessagesText = messages.filter((m) => m.role === "user").map((m) => m.content).join(" ");
  const userText = [
    complaintText,
    userMessagesText,
  ].join(" ");
  const draftText = complaintText || userMessagesText;
  const emergencyMode = hasAnyKeyword(userText, EMERGENCY_KEYWORDS);

  const emergencyReason = emergencyMode
    ? EMERGENCY_KEYWORDS.find((kw) => userText.toLowerCase().includes(kw)) || null
    : null;

  const missingDetails: MissingField[] = [];
  const collectedDetails: CollectedField[] = [];

  // Core fields
  const name = String(collectedFields.reporter_name || "");
  const phone = String(collectedFields.reporter_phone || "");
  const location = String(collectedFields.incident_location || "");
  const time = String(collectedFields.incident_time || "");
  const text = String(collectedFields.complaint_text || "");

  if (name) {
    collectedDetails.push({ key: "reporter_name", label: "Reporter Name", value: name });
  } else {
    missingDetails.push({ key: "reporter_name", label: "Reporter Name", priority: "high", hint: "Your full name for the report" });
  }

  if (phone) {
    collectedDetails.push({ key: "reporter_phone", label: "Phone Number", value: phone });
  } else {
    missingDetails.push({ key: "reporter_phone", label: "Phone Number", priority: "high", hint: "Contact number for follow-up" });
  }

  if (location) {
    collectedDetails.push({ key: "incident_location", label: "Location", value: location });
  } else {
    missingDetails.push({ key: "incident_location", label: "Incident Location", priority: "high", hint: "Where did this happen?" });
  }

  if (time) {
    collectedDetails.push({ key: "incident_time", label: "Time", value: time });
  } else {
    missingDetails.push({ key: "incident_time", label: "Incident Time", priority: "medium", hint: "When did this happen?" });
  }

  if (text) {
    collectedDetails.push({ key: "complaint_text", label: "Complaint Description", value: text.length > 80 ? text.slice(0, 77) + "..." : text });
  } else {
    missingDetails.push({ key: "complaint_text", label: "Complaint Description", priority: "critical", hint: "Please describe what happened" });
  }

  // Context fields
  if (hasAnyKeyword(userText, ["suspect", "perpetrator", "attacker", "assailant"])) {
    collectedDetails.push({ key: "suspect", label: "Suspect Details", value: "Mentioned in complaint" });
  }
  if (hasAnyKeyword(userText, ["witness", "saw", "saw it"])) {
    collectedDetails.push({ key: "witness", label: "Witnesses", value: "Mentioned in complaint" });
  }

  // Emergency-specific missing fields
  const requiredEmergencyFields: MissingField[] = [];
  if (emergencyMode) {
    if (!location) {
      requiredEmergencyFields.push({ key: "incident_location", label: "Location (Urgent)", priority: "critical", hint: "Please provide the exact location immediately" });
    }
    if (!phone) {
      requiredEmergencyFields.push({ key: "reporter_phone", label: "Contact Number (Urgent)", priority: "critical", hint: "We need a number to reach you" });
    }
    if (!hasImmediateDangerStatus(userText)) {
      requiredEmergencyFields.push({ key: "immediate_danger_status", label: "Immediate Danger Status", priority: "critical", hint: "Are you or others in immediate danger right now?" });
    }
    if (hasAnyKeyword(userText, ["fire", "trapped", "burning"]) && !hasPeopleAffectedDetails(userText)) {
      requiredEmergencyFields.push({ key: "people_affected", label: "People Affected", priority: "critical", hint: "How many people are affected or trapped?" });
    }
  }

  requiredEmergencyFields.forEach((field) => addMissingField(missingDetails, field));
  const criticalMissing = missingDetails.filter((field) => field.priority === "critical");

  const draftData = buildDraftData(collectedFields, draftText);

  return {
    missingDetails,
    collectedDetails,
    criticalMissing,
    emergencyMode,
    emergencyReason,
    requiredEmergencyFields,
    draftData,
  };
}
