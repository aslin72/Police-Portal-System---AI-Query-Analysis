import Link from "next/link";
import { ShieldCheck, ArrowRight, FileText, Shield } from "lucide-react";
import { buttonVariants } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";

export default function HomePage() {
  return (
    <div className="flex flex-col">
      <section className="border-b border-border bg-primary text-primary-foreground">
        <div className="mx-auto max-w-7xl px-4 py-16 sm:py-24">
          <div className="flex flex-col items-start gap-6 max-w-2xl">
            <div className="flex items-center gap-3">
              <ShieldCheck className="h-10 w-10 text-accent" />
              <h1 className="text-3xl sm:text-4xl font-bold tracking-tight">
                Police Complaint Portal
              </h1>
            </div>
            <p className="text-lg text-primary-foreground/80 leading-relaxed">
              A secure, AI-assisted platform for filing and managing police complaints.
              Submit incidents, upload evidence, and track case status through an
              enterprise-grade triage system.
            </p>
            <div className="flex flex-wrap gap-3">
              <Link
                href="/citizen/file-complaint"
                className={cn(buttonVariants({ variant: "secondary", size: "lg" }), "gap-2")}
              >
                <FileText className="h-5 w-5" />
                File Complaint
              </Link>
              <Link
                href="/officer/dashboard"
                className={cn(
                  buttonVariants({ size: "lg" }),
                  "gap-2 border border-primary-foreground/20 bg-transparent hover:bg-primary-foreground/10",
                )}
              >
                <Shield className="h-5 w-5" />
                Officer Dashboard
                <ArrowRight className="h-4 w-4" />
              </Link>
            </div>
          </div>
        </div>
      </section>

      <section className="mx-auto max-w-7xl px-4 py-12 sm:py-16">
        <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
          <Card className="border-border">
            <CardContent className="pt-6">
              <FileText className="h-8 w-8 text-accent mb-3" />
              <h3 className="font-semibold text-lg mb-1">File a Complaint</h3>
              <p className="text-sm text-muted-foreground">
                Submit structured complaints with reporter details, incident
                location, time, and supporting evidence.
              </p>
            </CardContent>
          </Card>

          <Card className="border-border">
            <CardContent className="pt-6">
              <ShieldCheck className="h-8 w-8 text-accent mb-3" />
              <h3 className="font-semibold text-lg mb-1">AI-Powered Triage</h3>
              <p className="text-sm text-muted-foreground">
                Complaints are automatically classified, prioritized, and
                routed to the right police unit using explainable rules.
              </p>
            </CardContent>
          </Card>

          <Card className="border-border">
            <CardContent className="pt-6">
              <Shield className="h-8 w-8 text-accent mb-3" />
              <h3 className="font-semibold text-lg mb-1">Officer Dashboard</h3>
              <p className="text-sm text-muted-foreground">
                Review, filter, and manage complaints with priority-based
                sorting, status updates, and evidence review.
              </p>
            </CardContent>
          </Card>
        </div>
      </section>
    </div>
  );
}
