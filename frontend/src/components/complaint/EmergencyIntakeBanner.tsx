"use client";

import { motion } from "motion/react";
import { AlertTriangle, Phone, MapPin, ShieldAlert, Users } from "lucide-react";
import type { MissingField } from "@/lib/complaint-intelligence";

interface EmergencyIntakeBannerProps {
  emergencyReason: string | null;
  requiredEmergencyFields: MissingField[];
}

const FIELD_ICONS: Record<string, typeof Phone> = {
  incident_location: MapPin,
  reporter_phone: Phone,
  immediate_danger_status: ShieldAlert,
  people_affected: Users,
};

export function EmergencyIntakeBanner({
  emergencyReason,
  requiredEmergencyFields,
}: EmergencyIntakeBannerProps) {
  if (requiredEmergencyFields.length === 0) return null;

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.98 }}
      animate={{ opacity: 1, scale: 1 }}
      className="rounded-lg border border-red-500/30 bg-red-500/5 p-4 space-y-3"
    >
      {/* Header */}
      <div className="flex items-center gap-2">
        <motion.div
          animate={{ scale: [1, 1.1, 1] }}
          transition={{ duration: 1.5, repeat: Infinity }}
        >
          <AlertTriangle className="h-5 w-5 text-red-400" />
        </motion.div>
        <div>
          <h3 className="text-sm font-semibold text-red-300">
            Emergency Intake Mode Activated
          </h3>
          <p className="text-xs text-red-400/70 mt-0.5">
            {emergencyReason
              ? `Detected: ${emergencyReason.replace(/_/g, " ")}`
              : "This complaint may need immediate attention."}
          </p>
        </div>
      </div>

      {/* Urgency message */}
      <p className="text-xs text-red-300/80 leading-relaxed">
        Please provide <strong>location</strong> and <strong>contact number</strong> first so we can
        dispatch help immediately.
      </p>

      {/* Required fields */}
      <div className="space-y-2">
        {requiredEmergencyFields.map((field) => {
          const Icon = FIELD_ICONS[field.key] || AlertTriangle;
          return (
            <div
              key={field.key}
              className="flex items-center gap-2 text-xs bg-red-500/10 rounded px-3 py-2 border border-red-500/20"
            >
              <Icon className="h-3.5 w-3.5 text-red-400 flex-shrink-0" />
              <div>
                <span className="font-medium text-red-300">{field.label}</span>
                <span className="text-red-400/60 ml-1">- {field.hint}</span>
              </div>
            </div>
          );
        })}
      </div>
    </motion.div>
  );
}
