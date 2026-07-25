"use client";

import type { ComplaintInsight } from "@/lib/complaint-intelligence";
import { EmergencyIntakeBanner } from "./EmergencyIntakeBanner";
import { MissingDetailsDetector } from "./MissingDetailsDetector";

interface ComplaintFeaturePanelProps {
  insight: ComplaintInsight;
}

export function ComplaintFeaturePanel({ insight }: ComplaintFeaturePanelProps) {
  return (
    <div className="space-y-3">
      {insight.emergencyMode && insight.requiredEmergencyFields.length > 0 && (
        <EmergencyIntakeBanner
          emergencyReason={insight.emergencyReason}
          requiredEmergencyFields={insight.requiredEmergencyFields}
        />
      )}

      <MissingDetailsDetector
        missingDetails={insight.missingDetails}
        collectedDetails={insight.collectedDetails}
        criticalMissing={insight.criticalMissing}
      />
    </div>
  );
}
