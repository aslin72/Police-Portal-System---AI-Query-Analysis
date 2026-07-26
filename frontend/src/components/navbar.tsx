"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { ArrowRight, ChevronDown, ClipboardList, FileSearch, FileText, LayoutDashboard } from "lucide-react";
import { PortalLogo } from "@/components/brand/PortalLogo";
import { buttonVariants } from "@/components/ui/button";
import { cn } from "@/lib/utils";

const citizenLinks = [
  { href: "/citizen/chat", label: "File Complaint", icon: FileText },
  { href: "/citizen/track-complaint", label: "Track Complaint", icon: ClipboardList },
];

const officerLinks = [
  { href: "/officer/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { href: "/officer/evidence-review", label: "Evidence", icon: FileSearch },
];

export default function Navbar() {
  const pathname = usePathname();
  const isLandingPage = pathname === "/";

  if (isLandingPage) {
    return (
      <header className="sticky top-0 z-50 border-b border-white/10 bg-[#08070D]/86 text-white backdrop-blur-xl supports-[backdrop-filter]:bg-[#08070D]/72">
        <div className="mx-auto flex h-[72px] max-w-[1440px] items-center gap-5 px-4 sm:px-6 lg:px-8">
          <Link href="/" className="flex shrink-0 items-center gap-3 font-semibold">
            <PortalLogo size="nav" animated priority />
            <span className="text-lg sm:text-xl">Police Complaint Portal</span>
          </Link>

          <nav className="mx-auto hidden items-center gap-8 text-sm text-muted-foreground lg:flex">
            <a href="#services" className="transition hover:text-white">Services</a>
            <a href="#how-it-works" className="transition hover:text-white">How It Works</a>
            <a href="#security" className="transition hover:text-white">Security</a>
            <a href="#resources" className="inline-flex items-center gap-1 transition hover:text-white">
              Resources
              <ChevronDown className="h-4 w-4" />
            </a>
          </nav>

          <div className="ml-auto flex items-center gap-2">
            <Link
              href="/officer/dashboard"
              className={cn(
                buttonVariants({ variant: "ghost", size: "lg" }),
                "hidden h-11 rounded-xl px-4 text-white hover:bg-white/[0.06] sm:inline-flex",
              )}
            >
              Log in
            </Link>
            <Link
              href="/officer/dashboard"
              className={cn(
                buttonVariants({ size: "lg" }),
                "h-11 gap-2 rounded-xl border border-accent/35 bg-primary/80 px-4 text-white shadow-[0_0_24px_rgba(190,24,93,0.22)] hover:bg-primary",
              )}
            >
              <span className="hidden sm:inline">Officer Dashboard</span>
              <span className="sm:hidden">Dashboard</span>
              <ArrowRight className="h-4 w-4" />
            </Link>
          </div>
        </div>
      </header>
    );
  }

  return (
    <header className="sticky top-0 z-50 border-b border-sidebar-border bg-sidebar/95 text-sidebar-foreground backdrop-blur supports-[backdrop-filter]:bg-sidebar/90">
      <div className="mx-auto flex h-14 max-w-7xl items-center gap-4 px-4">
        <Link href="/" className="flex items-center gap-2 font-semibold text-white">
          <PortalLogo size="compact" animated={false} showGlow={false} />
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
                  buttonVariants({ variant: "ghost", size: "sm" }),
                  "gap-1.5",
                  active
                    ? "bg-primary text-primary-foreground hover:bg-accent"
                    : "text-muted-foreground hover:bg-card hover:text-white",
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
                  buttonVariants({ variant: "ghost", size: "sm" }),
                  "gap-1.5",
                  active
                    ? "bg-primary text-primary-foreground hover:bg-accent"
                    : "text-muted-foreground hover:bg-card hover:text-white",
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
