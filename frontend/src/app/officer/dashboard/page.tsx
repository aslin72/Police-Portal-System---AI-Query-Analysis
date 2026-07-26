"use client";

import { useEffect, useMemo, useState } from "react";
import {
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  getFilteredRowModel,
  useReactTable,
  type ColumnDef,
  type SortingState,
  type ColumnFiltersState,
} from "@tanstack/react-table";
import { useRouter } from "next/navigation";
import { Activity, AlertCircle, BarChart3, ChevronDown, ChevronUp, LayoutDashboard, ListFilter } from "lucide-react";
import { AnimatedComplaintLineChart, type ChartPoint } from "@/components/analytics/AnimatedComplaintLineChart";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { getComplaints, type Complaint } from "@/lib/types";
import { getPriorityBadgeProps } from "@/lib/badge-utils";

const columns: ColumnDef<Complaint>[] = [
  {
    accessorKey: "id",
    header: "ID",
    size: 60,
    cell: ({ getValue }) => <span className="font-mono text-xs">#{getValue<number>()}</span>,
  },
  {
    accessorKey: "priority",
    header: "Priority",
    size: 100,
    cell: ({ getValue }) => {
      const val = getValue<string>();
      return <Badge {...getPriorityBadgeProps(val)}>{val}</Badge>;
    },
  },
  {
    accessorKey: "category",
    header: "Category",
    cell: ({ getValue }) => <span className="capitalize">{getValue<string>()}</span>,
  },
  {
    accessorKey: "summary",
    header: "Summary",
    cell: ({ getValue }) => (
      <span className="text-sm text-muted-foreground line-clamp-1 max-w-xs block">
        {getValue<string>()}
      </span>
    ),
  },
  {
    accessorKey: "status",
    header: "Status",
    size: 120,
    cell: ({ getValue }) => <Badge variant="outline">{getValue<string>()}</Badge>,
  },
  {
    accessorKey: "assigned_unit",
    header: "Assigned Unit",
    size: 160,
    cell: ({ getValue }) => <span className="text-sm">{getValue<string>() || "-"}</span>,
  },
  {
    accessorKey: "created_at",
    header: "Created",
    size: 140,
    cell: ({ getValue }) => {
      const val = getValue<string | null>();
      return val ? <span className="text-xs text-muted-foreground">{val.slice(0, 16)}</span> : "-";
    },
  },
];

const priorityLevels = ["Emergency", "High", "Medium", "Low"];
const statusLevels = ["New", "Under Review", "Assigned", "Resolved", "Closed"];

function formatChartLabel(date: Date) {
  return date.toLocaleDateString("en-US", { weekday: "short" });
}

function buildDailyComplaintData(complaints: Complaint[]): ChartPoint[] {
  if (complaints.length === 0) return [{ label: "No data", value: 0 }];

  const today = new Date();
  const days = Array.from({ length: 7 }, (_, index) => {
    const date = new Date(today);
    date.setHours(0, 0, 0, 0);
    date.setDate(today.getDate() - (6 - index));
    return date;
  });

  return days.map((date) => {
    const count = complaints.filter((complaint) => {
      if (!complaint.created_at) return false;
      const created = new Date(complaint.created_at);
      return (
        created.getFullYear() === date.getFullYear() &&
        created.getMonth() === date.getMonth() &&
        created.getDate() === date.getDate()
      );
    }).length;

    return {
      label: formatChartLabel(date),
      value: count,
    };
  });
}

function countByValue(complaints: Complaint[], key: "priority" | "status") {
  return complaints.reduce<Record<string, number>>((acc, complaint) => {
    const value = complaint[key] || "Unknown";
    acc[value] = (acc[value] || 0) + 1;
    return acc;
  }, {});
}

function getTopCategories(complaints: Complaint[]) {
  const counts = complaints.reduce<Record<string, number>>((acc, complaint) => {
    const category = complaint.category || "Uncategorized";
    acc[category] = (acc[category] || 0) + 1;
    return acc;
  }, {});

  return Object.entries(counts)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5);
}

export default function OfficerDashboard() {
  const router = useRouter();
  const [complaints, setComplaints] = useState<Complaint[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [sorting, setSorting] = useState<SortingState>([{ id: "id", desc: true }]);
  const [columnFilters, setColumnFilters] = useState<ColumnFiltersState>([]);
  const [globalFilter, setGlobalFilter] = useState("");
  const [priorityFilter, setPriorityFilter] = useState("all");
  const [statusFilter, setStatusFilter] = useState("all");
  const [activeView, setActiveView] = useState<"complaints" | "analytics">("complaints");

  useEffect(() => {
    getComplaints()
      .then(setComplaints)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  const metrics = useMemo(() => {
    const total = complaints.length;
    const emergencyHigh = complaints.filter((c) => c.priority === "Emergency" || c.priority === "High").length;
    const underReview = complaints.filter((c) => c.status === "Under Review").length;
    const closed = complaints.filter((c) => c.status === "Resolved" || c.status === "Closed").length;
    return { total, emergencyHigh, underReview, closed };
  }, [complaints]);

  const analytics = useMemo(() => {
    return {
      dailyData: buildDailyComplaintData(complaints),
      priorityCounts: countByValue(complaints, "priority"),
      statusCounts: countByValue(complaints, "status"),
      topCategories: getTopCategories(complaints),
      recent: [...complaints]
        .sort((a, b) => {
          const aTime = a.created_at ? new Date(a.created_at).getTime() : 0;
          const bTime = b.created_at ? new Date(b.created_at).getTime() : 0;
          return bTime - aTime;
        })
        .slice(0, 5),
    };
  }, [complaints]);

  const filteredData = useMemo(() => {
    let data = complaints;
    if (priorityFilter !== "all") data = data.filter((c) => c.priority === priorityFilter);
    if (statusFilter !== "all") data = data.filter((c) => c.status === statusFilter);
    return data;
  }, [complaints, priorityFilter, statusFilter]);

  const table = useReactTable({
    data: filteredData,
    columns,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    onSortingChange: setSorting,
    onColumnFiltersChange: setColumnFilters,
    onGlobalFilterChange: setGlobalFilter,
    state: { sorting, columnFilters, globalFilter },
  });

  return (
    <div className="mx-auto max-w-7xl px-4 py-8">
      <div className="mb-6 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-2">
          <LayoutDashboard className="h-6 w-6 text-accent" />
          <h2 className="text-2xl font-bold">Officer Dashboard</h2>
        </div>

        <div className="flex w-fit rounded-xl border border-border bg-card/70 p-1">
          <Button
            variant={activeView === "complaints" ? "default" : "ghost"}
            size="sm"
            className="gap-2 rounded-lg"
            onClick={() => setActiveView("complaints")}
          >
            <ListFilter className="h-4 w-4" />
            Complaints
          </Button>
          <Button
            variant={activeView === "analytics" ? "default" : "ghost"}
            size="sm"
            className="gap-2 rounded-lg"
            onClick={() => setActiveView("analytics")}
          >
            <BarChart3 className="h-4 w-4" />
            Analytics
          </Button>
        </div>
      </div>

      {/* Metrics */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4 mb-6">
        <Card className="border-border">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground">Total Complaints</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">{metrics.total}</p>
          </CardContent>
        </Card>
        <Card className="border-border">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground">Emergency / High</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-destructive">{metrics.emergencyHigh}</p>
          </CardContent>
        </Card>
        <Card className="border-border">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground">Under Review</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">{metrics.underReview}</p>
          </CardContent>
        </Card>
        <Card className="border-border">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground">Closed</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-accent">{metrics.closed}</p>
          </CardContent>
        </Card>
      </div>

      {activeView === "analytics" && (
        <div className="space-y-4">
          {error && (
            <div className="flex items-center gap-2 text-sm text-destructive">
              <AlertCircle className="h-4 w-4" />
              {error}
            </div>
          )}

          {loading ? (
            <div className="grid gap-4 lg:grid-cols-[1.35fr_0.65fr]">
              <Skeleton className="h-[360px] rounded-xl" />
              <Skeleton className="h-[360px] rounded-xl" />
            </div>
          ) : (
            <>
              <div className="grid gap-4 lg:grid-cols-[1.35fr_0.65fr]">
                <Card className="border-border bg-card/80">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-base">
                      <Activity className="h-5 w-5 text-accent" />
                      Complaint Volume
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <AnimatedComplaintLineChart data={analytics.dailyData} height={310} />
                  </CardContent>
                </Card>

                <Card className="border-border bg-card/80">
                  <CardHeader>
                    <CardTitle className="text-base">Priority Breakdown</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    {priorityLevels.map((priority) => (
                      <div key={priority} className="flex items-center justify-between rounded-xl border border-border bg-background/40 px-4 py-3">
                        <span className="text-sm text-muted-foreground">{priority}</span>
                        <span className="font-mono text-lg font-semibold text-white">{analytics.priorityCounts[priority] || 0}</span>
                      </div>
                    ))}
                  </CardContent>
                </Card>
              </div>

              <div className="grid gap-4 lg:grid-cols-3">
                <Card className="border-border bg-card/80">
                  <CardHeader>
                    <CardTitle className="text-base">Status Overview</CardTitle>
                  </CardHeader>
                  <CardContent className="grid gap-2">
                    {statusLevels.map((status) => (
                      <div key={status} className="flex items-center justify-between rounded-lg bg-background/40 px-3 py-2">
                        <span className="text-sm text-muted-foreground">{status}</span>
                        <span className="font-mono text-sm text-white">{analytics.statusCounts[status] || 0}</span>
                      </div>
                    ))}
                  </CardContent>
                </Card>

                <Card className="border-border bg-card/80">
                  <CardHeader>
                    <CardTitle className="text-base">Top Categories</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-2">
                    {analytics.topCategories.length === 0 ? (
                      <p className="rounded-xl border border-border bg-background/40 p-4 text-sm text-muted-foreground">
                        No category data yet.
                      </p>
                    ) : (
                      analytics.topCategories.map(([category, count]) => (
                        <div key={category} className="flex items-center justify-between rounded-lg bg-background/40 px-3 py-2">
                          <span className="text-sm capitalize text-muted-foreground">{category}</span>
                          <span className="font-mono text-sm text-white">{count}</span>
                        </div>
                      ))
                    )}
                  </CardContent>
                </Card>

                <Card className="border-border bg-card/80">
                  <CardHeader>
                    <CardTitle className="text-base">Recent Activity</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    {analytics.recent.length === 0 ? (
                      <p className="rounded-xl border border-border bg-background/40 p-4 text-sm text-muted-foreground">
                        No complaints have been filed yet.
                      </p>
                    ) : (
                      analytics.recent.map((complaint) => (
                        <button
                          key={complaint.id}
                          type="button"
                          className="block w-full rounded-xl border border-border bg-background/40 px-4 py-3 text-left transition hover:border-accent/40 hover:bg-muted/40"
                          onClick={() => router.push(`/officer/complaints/${complaint.id}`)}
                        >
                          <span className="block text-sm font-medium text-white">#{complaint.id} {complaint.category}</span>
                          <span className="mt-1 block line-clamp-1 text-xs text-muted-foreground">{complaint.summary || complaint.complaint_text}</span>
                        </button>
                      ))
                    )}
                  </CardContent>
                </Card>
              </div>
            </>
          )}
        </div>
      )}

      {/* Filters */}
      {activeView === "complaints" && (
        <Card className="border-border mb-4">
          <CardContent className="pt-4">
            <div className="flex flex-wrap gap-3">
              <Input
                placeholder="Search all fields..."
                value={globalFilter}
                onChange={(e) => setGlobalFilter(e.target.value)}
                className="max-w-xs"
              />
              <Select value={priorityFilter} onValueChange={(value: string | null) => setPriorityFilter(value ?? "")}>
                <SelectTrigger className="w-36">
                  <SelectValue placeholder="Priority" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Priorities</SelectItem>
                  <SelectItem value="Emergency">Emergency</SelectItem>
                  <SelectItem value="High">High</SelectItem>
                  <SelectItem value="Medium">Medium</SelectItem>
                  <SelectItem value="Low">Low</SelectItem>
                </SelectContent>
              </Select>
              <Select value={statusFilter} onValueChange={(value: string | null) => setStatusFilter(value ?? "")}>
                <SelectTrigger className="w-40">
                  <SelectValue placeholder="Status" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Statuses</SelectItem>
                  <SelectItem value="New">New</SelectItem>
                  <SelectItem value="Under Review">Under Review</SelectItem>
                  <SelectItem value="Assigned">Assigned</SelectItem>
                  <SelectItem value="Resolved">Resolved</SelectItem>
                  <SelectItem value="Closed">Closed</SelectItem>
                </SelectContent>
              </Select>
              {(priorityFilter !== "all" || statusFilter !== "all") && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => {
                    setPriorityFilter("all");
                    setStatusFilter("all");
                  }}
                >
                  Clear
                </Button>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Table */}
      {activeView === "complaints" && error && (
        <div className="flex items-center gap-2 text-sm text-destructive mb-4">
          <AlertCircle className="h-4 w-4" />
          {error}
        </div>
      )}

      {activeView === "complaints" &&
        (loading ? (
          <div className="space-y-3">
            {[1, 2, 3, 4].map((i) => (
              <Skeleton key={i} className="h-12 w-full" />
            ))}
          </div>
        ) : (
          <Card className="border-border">
            <div className="overflow-x-auto">
              <Table>
                <TableHeader>
                  {table.getHeaderGroups().map((headerGroup) => (
                    <TableRow key={headerGroup.id}>
                      {headerGroup.headers.map((header) => (
                        <TableHead
                          key={header.id}
                          style={{ width: header.getSize() !== 150 ? header.getSize() : undefined }}
                          className={header.column.getCanSort() ? "cursor-pointer select-none" : ""}
                          onClick={header.column.getToggleSortingHandler()}
                        >
                          <div className="flex items-center gap-1">
                            {flexRender(header.column.columnDef.header, header.getContext())}
                            {{
                              asc: <ChevronUp className="h-3 w-3" />,
                              desc: <ChevronDown className="h-3 w-3" />,
                            }[header.column.getIsSorted() as string] ?? null}
                          </div>
                        </TableHead>
                      ))}
                    </TableRow>
                  ))}
                </TableHeader>
                <TableBody>
                  {table.getRowModel().rows.length === 0 ? (
                    <TableRow>
                      <TableCell colSpan={columns.length} className="text-center text-muted-foreground py-8">
                        {complaints.length === 0 ? "No complaints in the system." : "No matching complaints."}
                      </TableCell>
                    </TableRow>
                  ) : (
                    table.getRowModel().rows.map((row) => (
                      <TableRow
                        key={row.id}
                        className="cursor-pointer hover:bg-muted/50"
                        onClick={() => router.push(`/officer/complaints/${row.original.id}`)}
                      >
                        {row.getVisibleCells().map((cell) => (
                          <TableCell key={cell.id}>
                            {flexRender(cell.column.columnDef.cell, cell.getContext())}
                          </TableCell>
                        ))}
                      </TableRow>
                    ))
                  )}
                </TableBody>
              </Table>
            </div>
          </Card>
        ))}
    </div>
  );
}
