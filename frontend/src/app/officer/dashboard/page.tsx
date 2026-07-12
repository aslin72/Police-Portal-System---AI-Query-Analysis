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
import { AlertCircle, ChevronDown, ChevronUp, LayoutDashboard } from "lucide-react";
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
import { getComplaints, type Complaint } from "@/lib/types"
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

  useEffect(() => {
    getComplaints()
      .then(setComplaints)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  const metrics = useMemo(() => {
    const total = complaints.length;
    const emergencyHigh = complaints.filter((c) => c.priority === "Emergency" || c.priority === "High").length;
    const assigned = complaints.filter((c) => c.status === "Assigned").length;
    const closed = complaints.filter((c) => c.status === "Resolved" || c.status === "Closed").length;
    return { total, emergencyHigh, assigned, closed };
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
      <div className="flex items-center gap-2 mb-6">
        <LayoutDashboard className="h-6 w-6 text-accent" />
        <h2 className="text-2xl font-bold">Officer Dashboard</h2>
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
            <CardTitle className="text-sm text-muted-foreground">Assigned</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">{metrics.assigned}</p>
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

      {/* Filters */}
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

      {/* Table */}
      {error && (
        <div className="flex items-center gap-2 text-sm text-destructive mb-4">
          <AlertCircle className="h-4 w-4" />
          {error}
        </div>
      )}

      {loading ? (
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
      )}
    </div>
  );
}
