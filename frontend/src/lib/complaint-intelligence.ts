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
  summary: string;
}

const EMERGENCY_KEYWORDS = [
  "missing child", "child missing", "kidnapped", "kidnapping",
  "weapon", "knife", "gun", "shot", "shooting", "stabbed", "stabbing",
  "bleeding", "serious injury", "injured badly",
  "fire", "burning", "trapped", "active fire",
  "active threat", "ongoing assault", "assault",
  "suicide", "self-harm", "end my life", "kill myself",
  "immediate danger", "life threatening", "hostage",
  "murder", "death", "dead body", "found dead",
];

const CATEGORY_KEYWORDS: Record<string, string[]> = {
  "child safety": ["child", "kid", "son", "daughter", "minor", "missing child", "kidnapped"],
  "cyber crime incident": ["scam", "hacked", "phishing", "online fraud", "cyber", "email", "bank account", "upi", "otp"],
  "women help desk": ["husband", "wife", "domestic", "harassment", "stalking", "assault", "dowry"],
  "public healthcare": ["sick", "poisoning", "contaminated", "outbreak", "fever", "vomiting", "hospital"],
  "road accident": ["car", "bike", "accident", "collision", "hit", "vehicle", "highway", "road"],
  "murder / serious crime incident": ["murder", "dead", "body", "stab", "shot", "killed", "homicide"],
  "fire accident": ["fire", "burning", "smoke", "explosion", "gas leak", "blaze"],
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

function estimateCategory(text: string): string {
  const lower = text.toLowerCase();
  let bestCategory = "general issue recorded";
  let bestScore = 0;

  for (const [category, keywords] of Object.entries(CATEGORY_KEYWORDS)) {
    let score = 0;
    for (const kw of keywords) {
      if (lower.includes(kw)) score++;
    }
    if (score > bestScore) {
      bestScore = score;
      bestCategory = category;
    }
  }
  return bestCategory;
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
  const category = estimateCategory(text);
  const priority = estimatePriority(text, category);
  const riskFlags = estimateRiskFlags(text, category);
  const persons = extractPersonsInvolved(text);
  const hasUrgent = riskFlags.some((f) =>
    ["injury_reported", "weapon_involved", "fire_risk", "person_missing", "urgent_medical_attention"].includes(f),
  );

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
    estimatedAssignedUnit: UNIT_MAP[category] || "General Desk",
    estimatedRecommendedAction: priority === "Emergency"
      ? "Respond immediately. Dispatch nearest unit and notify emergency services."
      : hasUrgent
        ? "Review immediately and contact emergency response."
        : priority === "High"
          ? "Prioritize review within 1 hour."
          : "Assign for standard processing within 24 hours.",
    summary: text.length > 120 ? text.slice(0, 117) + "..." : text,
  };
}

export function deriveComplaintInsight(
  collectedFields: Record<string, unknown>,
  messages: Array<{ role: string; content: string }>,
): ComplaintInsight {
  const complaintText = String(collectedFields.complaint_text || "");
  const allText = [complaintText, ...messages.map((m) => m.content)].join(" ");
  const emergencyMode = hasAnyKeyword(allText, EMERGENCY_KEYWORDS);

  const emergencyReason = emergencyMode
    ? EMERGENCY_KEYWORDS.find((kw) => allText.toLowerCase().includes(kw)) || null
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
  if (hasAnyKeyword(allText, ["suspect", "perpetrator", "attacker", "assailant"])) {
    collectedDetails.push({ key: "suspect", label: "Suspect Details", value: "Mentioned in complaint" });
  }
  if (hasAnyKeyword(allText, ["witness", "saw", "saw it"])) {
    collectedDetails.push({ key: "witness", label: "Witnesses", value: "Mentioned in complaint" });
  }

  // Emergency-specific missing fields
  const requiredEmergencyFields: MissingField[] = [];
  if (emergencyMode) {
    if (!location) {
      requiredEmergencyFields.push({ key: "emergency_location", label: "Location (Urgent)", priority: "critical", hint: "Please provide the exact location immediately" });
    }
    if (!phone) {
      requiredEmergencyFields.push({ key: "emergency_phone", label: "Contact Number (Urgent)", priority: "critical", hint: "We need a number to reach you" });
    }
    requiredEmergencyFields.push({ key: "emergency_danger", label: "Immediate Danger Status", priority: "critical", hint: "Are you or others in immediate danger right now?" });
    if (hasAnyKeyword(allText, ["fire", "trapped", "burning"])) {
      requiredEmergencyFields.push({ key: "emergency_people", label: "People Affected", priority: "critical", hint: "How many people are affected or trapped?" });
    }
  }

  // Critical missing = emergency fields that are also in the general missing list
  const criticalMissing = requiredEmergencyFields.filter(
    (ef) => !collectedDetails.some((cd) => cd.key === ef.key.replace("emergency_", "")),
  );

  const draftData = buildDraftData(collectedFields, complaintText);

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
