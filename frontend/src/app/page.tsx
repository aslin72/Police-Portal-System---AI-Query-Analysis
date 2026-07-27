import Link from "next/link";
import {
  Activity,
  ArrowRight,
  Bell,
  Brain,
  Building2,
  Filter,
  FileSearch,
  FileText,
  FolderOpen,
  LayoutDashboard,
  LockKeyhole,
  Radio,
  ShieldCheck,
  Sparkles,
  Zap,
} from "lucide-react";
import { AnimatedComplaintLineChart, type ChartPoint } from "@/components/analytics/AnimatedComplaintLineChart";
import { buttonVariants } from "@/components/ui/button";
import { cn } from "@/lib/utils";

const previewChartData: ChartPoint[] = [
  { label: "Mon", value: 18 },
  { label: "Tue", value: 43 },
  { label: "Wed", value: 31 },
  { label: "Thu", value: 67 },
  { label: "Fri", value: 52 },
  { label: "Sat", value: 78 },
  { label: "Sun", value: 106 },
];

const dashboardMetrics = [
  { label: "Total Complaints", value: "324", change: "+12%" },
  { label: "Under Review", value: "87", change: "+8%" },
  { label: "Resolved", value: "182", change: "+15%" },
  { label: "Overdue", value: "23", change: "-5%" },
];

const recentComplaints = [
  { title: "Noise Disturbance", id: "#CP-2025-0187", status: "Under Review" },
  { title: "Vandalism", id: "#CP-2025-0186", status: "Under Review" },
  { title: "Traffic Violation", id: "#CP-2025-0185", status: "Resolved" },
  { title: "Harassment", id: "#CP-2025-0184", status: "Under Review" },
];

const services = [
  {
    title: "File a Complaint",
    body: "Submit structured complaints with location, time, details, and supporting evidence.",
    href: "/citizen/chat",
    icon: FileText,
  },
  {
    title: "AI-Powered Triage",
    body: "Classify, prioritize, and route complaints to the right unit using explainable rules.",
    href: "/citizen/chat",
    icon: Brain,
  },
  {
    title: "Officer Dashboard",
    body: "Review, filter, and manage complaints with priority queues and status updates.",
    href: "/officer/dashboard",
    icon: LayoutDashboard,
  },
  {
    title: "Evidence Management",
    body: "Upload, organize, and review evidence with clear complaint-level records.",
    href: "/officer/evidence-review",
    icon: FolderOpen,
  },
];

const trustMetrics = [
  { label: "1,247+ Departments", icon: Building2 },
  { label: "99.99% Uptime", icon: Activity },
  { label: "SOC 2 Compliant", icon: ShieldCheck },
];

const trustItems = [
  { label: "Secure & Private", detail: "Protected complaint intake", icon: LockKeyhole },
  { label: "Faster Reviews", detail: "Cleaner queues for officers", icon: Zap },
  { label: "Actionable Insights", detail: "Live metrics and case trends", icon: Sparkles },
];

const dashboardHighlights = [
  { label: "Secure & Private", icon: LockKeyhole },
  { label: "Faster Reviews", icon: Zap },
  { label: "Actionable Insights", icon: Sparkles },
];

const securityItems = [
  "Secure complaint records",
  "Evidence linked to case IDs",
  "Officer review workflow",
];

function LandingDashboardPreview() {
  return (
    <div className="relative rounded-[1.65rem] border border-white/10 bg-[#090C12]/90 p-4 shadow-[0_0_0_1px_rgba(190,24,93,0.28),0_30px_90px_rgba(0,0,0,0.5)] backdrop-blur-xl">
      <div className="portal-pulse pointer-events-none absolute -inset-px rounded-[1.65rem] border border-accent/25" />
      <div className="pointer-events-none absolute inset-x-8 top-0 h-px bg-gradient-to-r from-transparent via-accent to-transparent" />

      <div className="grid min-h-[500px] overflow-hidden rounded-[1.25rem] border border-white/10 bg-[#080B10] lg:grid-cols-[168px_1fr]">
        <aside className="hidden border-r border-white/10 bg-[#070A0E]/90 p-5 lg:block">
          <div className="space-y-1">
            <p className="text-sm font-semibold text-white">Command Center</p>
            <p className="text-[11px] text-muted-foreground">Live operations view</p>
          </div>
          <nav className="mt-8 space-y-2 text-sm text-muted-foreground">
            {["Overview", "Complaints", "Evidence", "Units", "Reports", "Alerts"].map((item, index) => (
              <div
                key={item}
                className={cn(
                  "rounded-lg px-3 py-2",
                  index === 0 ? "border border-accent/25 bg-accent/12 text-white" : "text-muted-foreground",
                )}
              >
                {item}
              </div>
            ))}
          </nav>
        </aside>

        <div className="p-5">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div>
              <h2 className="text-xl font-semibold text-white">Complaints Overview</h2>
              <p className="mt-1 text-xs text-muted-foreground">Preview data for complaint operations</p>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <div className="rounded-lg border border-white/10 bg-white/[0.03] px-3 py-2 text-xs text-muted-foreground">
                May 12 - May 18, 2025
              </div>
              <div className="rounded-lg border border-white/10 bg-white/[0.03] px-3 py-2 text-xs text-muted-foreground">
                All Units
              </div>
              <button
                type="button"
                aria-label="Filter"
                className="flex h-9 w-9 items-center justify-center rounded-lg border border-white/10 bg-white/[0.03] text-muted-foreground"
              >
                <Filter className="h-4 w-4" />
              </button>
              <Bell className="ml-1 h-4 w-4 text-muted-foreground" />
              <div className="flex h-9 w-9 items-center justify-center rounded-full bg-muted text-xs text-white">OP</div>
            </div>
          </div>

          <div className="mt-4 flex flex-wrap gap-2">
            {dashboardHighlights.map((item) => {
              const Icon = item.icon;
              return (
                <span
                  key={item.label}
                  className="inline-flex items-center gap-1.5 rounded-full border border-white/10 bg-white/[0.035] px-3 py-1.5 text-[11px] font-medium text-muted-foreground"
                >
                  <Icon className="h-3.5 w-3.5 text-ring" />
                  {item.label}
                </span>
              );
            })}
          </div>

          <div className="mt-5 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
            {dashboardMetrics.map((metric) => (
              <div key={metric.label} className="rounded-xl border border-white/10 bg-white/[0.035] p-4">
                <p className="text-xs text-muted-foreground">{metric.label}</p>
                <p className="mt-4 text-2xl font-semibold text-white">{metric.value}</p>
                <p className="mt-2 text-xs text-ring">{metric.change} vs last week</p>
              </div>
            ))}
          </div>

          <div className="mt-4 grid gap-4 xl:grid-cols-[1.45fr_0.85fr]">
            <div className="overflow-hidden rounded-xl border border-white/10 bg-white/[0.035] p-4">
              <p className="text-sm font-medium text-white">Complaints Over Time</p>
              <AnimatedComplaintLineChart data={previewChartData} height={250} />
            </div>

            <div className="rounded-xl border border-white/10 bg-white/[0.035] p-4">
              <p className="text-sm font-medium text-white">Recent Complaints</p>
              <div className="mt-5 space-y-4">
                {recentComplaints.map((complaint) => (
                  <div key={complaint.id} className="flex items-start justify-between gap-3">
                    <div>
                      <p className="text-sm font-medium text-white">{complaint.title}</p>
                      <p className="mt-0.5 text-xs text-muted-foreground">{complaint.id}</p>
                    </div>
                    <span
                      className={cn(
                        "rounded-md border px-2 py-1 text-[10px] font-medium",
                        complaint.status === "Resolved"
                          ? "border-ring/30 bg-ring/10 text-ring"
                          : "border-accent/30 bg-accent/10 text-accent",
                      )}
                    >
                      {complaint.status}
                    </span>
                  </div>
                ))}
              </div>
              <Link href="/officer/dashboard" className="mt-5 inline-flex items-center gap-2 text-sm text-white">
                View all complaints
                <ArrowRight className="h-4 w-4" />
              </Link>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function HomePage() {
  return (
    <main className="relative min-h-[calc(100dvh-3.5rem)] overflow-hidden bg-background text-foreground">
      <div className="portal-gradient-field portal-gradient-live pointer-events-none absolute inset-0 scale-105" />
      <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(rgba(250,247,251,0.035)_1px,transparent_1px),linear-gradient(90deg,rgba(250,247,251,0.035)_1px,transparent_1px)] bg-[size:72px_72px] opacity-20" />
      <div className="portal-scan-line pointer-events-none absolute left-0 top-24 h-[1px] w-full bg-gradient-to-r from-transparent via-accent/70 to-transparent" />

      <section className="relative mx-auto grid max-w-[1440px] gap-10 px-4 pb-10 pt-10 sm:px-6 lg:min-h-[calc(100dvh-3.5rem)] lg:grid-cols-[0.78fr_1.22fr] lg:items-center lg:px-8 lg:pb-14 lg:pt-12">
        <div className="max-w-2xl">
          <div className="mb-6 grid gap-3 sm:grid-cols-3">
            {trustMetrics.map((metric) => {
              const Icon = metric.icon;
              return (
                <div
                  key={metric.label}
                  className="flex items-center gap-2.5 rounded-xl border border-white/10 bg-white/[0.035] px-3 py-2.5"
                >
                  <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg border border-white/10 bg-white/[0.05]">
                    <Icon className="h-4 w-4 text-ring" />
                  </span>
                  <span className="text-xs font-medium leading-snug text-white sm:text-[13px]">{metric.label}</span>
                </div>
              );
            })}
          </div>

          <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-accent/30 bg-accent/10 px-4 py-2 text-sm text-white shadow-[inset_0_1px_0_rgba(255,255,255,0.08)]">
            <ShieldCheck className="h-4 w-4 text-ring" />
            CJIS & SOC 2 Compliant
          </div>

          <h1 className="text-4xl font-semibold leading-[1.03] tracking-normal text-white sm:text-5xl lg:text-6xl xl:text-7xl">
            Modern complaint management for{" "}
            <span className="text-accent">safer communities.</span>
          </h1>
          <p className="mt-6 max-w-xl text-base leading-7 text-muted-foreground sm:text-lg">
            File incidents, upload evidence, track status, and support faster officer review.
          </p>

          <div className="mt-8 flex flex-wrap gap-4">
            <Link
              href="/citizen/chat"
              className={cn(
                buttonVariants({ size: "lg" }),
                "h-12 gap-2 rounded-xl px-5 text-base shadow-[0_18px_48px_rgba(190,24,93,0.32)]",
              )}
            >
              <FileText className="h-5 w-5" />
              File Complaint
            </Link>
            <Link
              href="/citizen/track-complaint"
              className={cn(
                buttonVariants({ variant: "outline", size: "lg" }),
                "h-12 gap-2 rounded-xl border-white/16 bg-white/[0.025] px-5 text-base text-white shadow-[inset_0_1px_0_rgba(255,255,255,0.08)] hover:bg-white/[0.06]",
              )}
            >
              <FileSearch className="h-5 w-5" />
              Track Status
              <ArrowRight className="h-4 w-4" />
            </Link>
          </div>

          <div className="mt-8 grid gap-4 sm:grid-cols-3">
            {trustItems.map((item) => {
              const Icon = item.icon;
              return (
                <div key={item.label} className="flex items-center gap-3">
                  <span className="flex h-9 w-9 items-center justify-center rounded-lg border border-white/10 bg-white/[0.05]">
                    <Icon className="h-4 w-4 text-ring" />
                  </span>
                  <span>
                    <span className="block text-sm font-medium text-white">{item.label}</span>
                    <span className="block text-xs text-muted-foreground">{item.detail}</span>
                  </span>
                </div>
              );
            })}
          </div>
        </div>

        <LandingDashboardPreview />
      </section>

      <section id="services" className="relative mx-auto max-w-[1440px] px-4 pb-12 sm:px-6 lg:px-8">
        <div className="grid gap-5 md:grid-cols-2 xl:grid-cols-4">
          {services.map((service, index) => {
            const Icon = service.icon;
            return (
              <Link
                key={service.title}
                href={service.href}
                className={cn(
                  "group relative min-h-[230px] overflow-hidden rounded-[1.35rem] border border-white/10 bg-card/70 p-6 shadow-[0_24px_70px_rgba(0,0,0,0.24)] transition duration-300 hover:-translate-y-1 hover:border-accent/35 hover:bg-card",
                  index === 1 && "bg-[radial-gradient(circle_at_20%_0%,rgba(190,24,93,0.18),transparent_34%),rgba(23,17,31,0.72)]",
                  index === 2 && "bg-[radial-gradient(circle_at_85%_10%,rgba(214,168,90,0.12),transparent_30%),rgba(23,17,31,0.72)]",
                )}
              >
                <div className="flex h-16 w-16 items-center justify-center rounded-full border border-accent/25 bg-accent/12 shadow-[inset_0_1px_0_rgba(255,255,255,0.08)]">
                  <Icon className="h-7 w-7 text-accent" />
                </div>
                <h2 className="mt-8 text-xl font-semibold text-white">{service.title}</h2>
                <p className="mt-3 text-sm leading-6 text-muted-foreground">{service.body}</p>
                <span className="absolute bottom-6 right-6 flex h-10 w-10 items-center justify-center rounded-lg border border-white/10 bg-white/[0.03] text-white transition group-hover:border-accent/35 group-hover:text-accent">
                  <ArrowRight className="h-5 w-5" />
                </span>
              </Link>
            );
          })}
        </div>
      </section>

      <section id="security" className="relative mx-auto max-w-[1440px] px-4 pb-12 sm:px-6 lg:px-8">
        <div className="rounded-[1.35rem] border border-white/10 bg-[#0B0A10]/78 p-6 shadow-[0_24px_70px_rgba(0,0,0,0.22)]">
          <div className="grid gap-6 lg:grid-cols-[0.85fr_1.15fr] lg:items-center">
            <div>
              <h2 className="text-2xl font-semibold text-white sm:text-3xl">Designed for official handling.</h2>
              <p className="mt-3 max-w-xl text-sm leading-6 text-muted-foreground">
                Complaint intake, evidence review, and officer updates stay organized around the complaint record.
              </p>
            </div>
            <div className="grid gap-3 sm:grid-cols-3">
              {securityItems.map((item) => (
                <div key={item} className="rounded-2xl border border-white/10 bg-white/[0.035] p-5">
                  <LockKeyhole className="mb-4 h-5 w-5 text-ring" />
                  <p className="text-sm font-medium text-white">{item}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      <section id="how-it-works" className="relative border-t border-white/10 bg-sidebar/30">
        <span id="resources" className="absolute -top-20" aria-hidden="true" />
        <div className="mx-auto grid max-w-[1440px] gap-8 px-4 py-12 sm:px-6 lg:grid-cols-[0.95fr_1.05fr] lg:px-8">
          <div>
            <h2 className="max-w-xl text-3xl font-semibold leading-tight text-white sm:text-4xl">
              A clearer path from citizen report to officer action.
            </h2>
            <p className="mt-4 max-w-xl text-sm leading-6 text-muted-foreground">
              The portal keeps emergency details visible, checks missing information, and helps officers focus on the most urgent complaints first.
            </p>
          </div>
          <div className="grid gap-3 sm:grid-cols-2">
            {["Report incident", "Attach evidence", "Review triage", "Track status"].map((item) => (
              <div key={item} className="rounded-2xl border border-white/10 bg-white/[0.035] p-5">
                <Radio className="mb-4 h-5 w-5 text-ring" />
                <p className="text-base font-medium text-white">{item}</p>
              </div>
            ))}
          </div>
        </div>
      </section>
    </main>
  );
}
