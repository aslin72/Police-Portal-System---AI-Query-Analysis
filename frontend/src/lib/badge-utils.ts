import type { VariantProps } from "class-variance-authority";
import type { badgeVariants } from "@/components/ui/badge";

const BADGE_VARIANTS: Record<string, VariantProps<typeof badgeVariants>["variant"]> = {
  Emergency: "destructive",
  High: "destructive",
  Medium: "secondary",
  Low: "outline",
};

const BADGE_COLORS: Record<string, string> = {
  Medium: "bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-400 border-amber-200 dark:border-amber-800",
};

export function getPriorityBadgeProps(priority: string) {
  return {
    variant: BADGE_VARIANTS[priority] || ("outline" as const),
    className: BADGE_COLORS[priority] || "",
  };
}
