import Image from "next/image";
import { cn } from "@/lib/utils";

const LOGO_SRC = "/assets/ChatGPT%20Image%20Jul%2026,%202026,%2002_09_42%20AM.png";

type PortalLogoSize = "compact" | "nav" | "hero" | "mark";

interface PortalLogoProps {
  size?: PortalLogoSize;
  animated?: boolean;
  showGlow?: boolean;
  priority?: boolean;
  className?: string;
  imageClassName?: string;
}

const sizeClasses: Record<PortalLogoSize, string> = {
  compact: "h-8 w-8 rounded-lg",
  nav: "h-10 w-10 rounded-xl",
  hero: "h-20 w-20 rounded-2xl",
  mark: "h-12 w-12 rounded-xl",
};

const imageSizes: Record<PortalLogoSize, string> = {
  compact: "32px",
  nav: "40px",
  hero: "80px",
  mark: "48px",
};

export function PortalLogo({
  size = "nav",
  animated = false,
  showGlow = true,
  priority = false,
  className,
  imageClassName,
}: PortalLogoProps) {
  return (
    <span
      className={cn(
        "portal-logo-shell relative inline-flex shrink-0 items-center justify-center overflow-hidden border border-accent/35 bg-[#09070D]",
        sizeClasses[size],
        showGlow && "portal-logo-glow",
        animated && "portal-logo-live",
        className,
      )}
      aria-hidden="true"
    >
      <Image
        src={LOGO_SRC}
        alt=""
        fill
        priority={priority}
        sizes={imageSizes[size]}
        className={cn("object-cover object-center scale-[1.82]", imageClassName)}
      />
      {showGlow && <span className="portal-logo-halo" aria-hidden="true" />}
      {animated && <span className="portal-logo-sheen" aria-hidden="true" />}
    </span>
  );
}
