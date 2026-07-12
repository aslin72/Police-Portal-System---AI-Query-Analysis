"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { ShieldCheck, FileText, ClipboardList, LayoutDashboard, FileSearch } from "lucide-react";
import { buttonVariants } from "@/components/ui/button";
import { cn } from "@/lib/utils";

const citizenLinks = [
  { href: "/citizen/file-complaint", label: "File Complaint", icon: FileText },
  { href: "/citizen/track-complaint", label: "Track Complaint", icon: ClipboardList },
];

const officerLinks = [
  { href: "/officer/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { href: "/officer/evidence-review", label: "Evidence", icon: FileSearch },
];

export default function Navbar() {
  const pathname = usePathname();

  return (
    <header className="sticky top-0 z-50 border-b border-border bg-card/95 backdrop-blur supports-[backdrop-filter]:bg-card/80">
      <div className="mx-auto flex h-14 max-w-7xl items-center gap-4 px-4">
        <Link href="/" className="flex items-center gap-2 font-semibold text-primary">
          <ShieldCheck className="h-6 w-6 text-accent" />
          <span className="hidden sm:inline">Police Complaint Portal</span>
        </Link>

        <nav className="ml-auto flex items-center gap-1">
          {citizenLinks.map((link) => {
            const Icon = link.icon;
            const active = pathname.startsWith(link.href);
            return (
              <Link
                key={link.href}
                href={link.href}
                className={cn(
                  buttonVariants({ variant: active ? "secondary" : "ghost", size: "sm" }),
                  "gap-1.5",
                  active && "bg-secondary text-secondary-foreground",
                )}
              >
                <Icon className="h-4 w-4" />
                <span className="hidden sm:inline">{link.label}</span>
              </Link>
            );
          })}

          <span className="mx-1 h-5 w-px bg-border" />

          {officerLinks.map((link) => {
            const Icon = link.icon;
            const active = pathname.startsWith(link.href);
            return (
              <Link
                key={link.href}
                href={link.href}
                className={cn(
                  buttonVariants({ variant: active ? "secondary" : "ghost", size: "sm" }),
                  "gap-1.5",
                  active && "bg-secondary text-secondary-foreground",
                )}
              >
                <Icon className="h-4 w-4" />
                <span className="hidden sm:inline">{link.label}</span>
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
