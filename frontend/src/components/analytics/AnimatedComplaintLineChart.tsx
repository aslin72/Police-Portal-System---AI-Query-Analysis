"use client";

import { useId, useMemo } from "react";
import { motion, useReducedMotion } from "motion/react";
import { cn } from "@/lib/utils";

export interface ChartPoint {
  label: string;
  value: number;
}

interface AnimatedComplaintLineChartProps {
  data: ChartPoint[];
  className?: string;
  height?: number;
  showLabels?: boolean;
}

const VIEWBOX_WIDTH = 640;
const VIEWBOX_HEIGHT = 260;
const PADDING_X = 34;
const PADDING_TOP = 28;
const PADDING_BOTTOM = 38;

function toPath(points: { x: number; y: number }[]) {
  if (points.length === 0) return "";
  return points.map((point, index) => `${index === 0 ? "M" : "L"} ${point.x} ${point.y}`).join(" ");
}

function toAreaPath(points: { x: number; y: number }[]) {
  if (points.length === 0) return "";
  const baseline = VIEWBOX_HEIGHT - PADDING_BOTTOM;
  return `${toPath(points)} L ${points[points.length - 1].x} ${baseline} L ${points[0].x} ${baseline} Z`;
}

export function AnimatedComplaintLineChart({
  data,
  className,
  height = 260,
  showLabels = true,
}: AnimatedComplaintLineChartProps) {
  const reduceMotion = useReducedMotion();
  const rawId = useId().replace(/:/g, "");
  const gradientId = `complaint-line-${rawId}`;
  const areaId = `complaint-area-${rawId}`;
  const glowId = `complaint-glow-${rawId}`;

  const chart = useMemo(() => {
    const safeData = data.length ? data : [{ label: "No data", value: 0 }];
    const maxValue = Math.max(...safeData.map((point) => point.value), 1);
    const innerWidth = VIEWBOX_WIDTH - PADDING_X * 2;
    const innerHeight = VIEWBOX_HEIGHT - PADDING_TOP - PADDING_BOTTOM;
    const gap = safeData.length > 1 ? innerWidth / (safeData.length - 1) : 0;

    const points = safeData.map((point, index) => ({
      label: point.label,
      value: point.value,
      x: PADDING_X + gap * index,
      y: PADDING_TOP + innerHeight - (point.value / maxValue) * innerHeight,
    }));

    return {
      points,
      linePath: toPath(points),
      areaPath: toAreaPath(points),
    };
  }, [data]);

  const dotPoints = chart.points.length > 1 ? chart.points : [chart.points[0], chart.points[0]];
  const lastPoint = chart.points[chart.points.length - 1];

  return (
    <div className={cn("relative h-full min-h-[180px] w-full", className)} style={{ height }}>
      <svg
        className="h-full w-full overflow-visible"
        role="img"
        aria-label="Complaint volume line chart"
        viewBox={`0 0 ${VIEWBOX_WIDTH} ${VIEWBOX_HEIGHT}`}
        preserveAspectRatio="none"
      >
        <defs>
          <linearGradient id={gradientId} x1="0" x2="1" y1="0" y2="0">
            <stop offset="0%" stopColor="#D6A85A" />
            <stop offset="48%" stopColor="#F43F5E" />
            <stop offset="100%" stopColor="#BE185D" />
          </linearGradient>
          <linearGradient id={areaId} x1="0" x2="0" y1="0" y2="1">
            <stop offset="0%" stopColor="#BE185D" stopOpacity="0.36" />
            <stop offset="58%" stopColor="#BE185D" stopOpacity="0.12" />
            <stop offset="100%" stopColor="#BE185D" stopOpacity="0" />
          </linearGradient>
          <filter id={glowId} x="-20%" y="-40%" width="140%" height="180%">
            <feGaussianBlur stdDeviation="5" result="coloredBlur" />
            <feMerge>
              <feMergeNode in="coloredBlur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {[0, 1, 2, 3].map((line) => {
          const y = PADDING_TOP + ((VIEWBOX_HEIGHT - PADDING_TOP - PADDING_BOTTOM) / 3) * line;
          return (
            <line
              key={line}
              x1={PADDING_X}
              x2={VIEWBOX_WIDTH - PADDING_X}
              y1={y}
              y2={y}
              stroke="rgba(199, 191, 209, 0.12)"
              strokeDasharray="6 10"
            />
          );
        })}

        <path d={chart.areaPath} fill={`url(#${areaId})`} />
        <motion.path
          d={chart.linePath}
          fill="none"
          stroke={`url(#${gradientId})`}
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth="4"
          filter={`url(#${glowId})`}
          initial={reduceMotion ? false : { pathLength: 0, opacity: 0.55 }}
          animate={{ pathLength: 1, opacity: 1 }}
          transition={{ duration: 1.2, ease: [0.16, 1, 0.3, 1] }}
        />

        {!reduceMotion && (
          <motion.circle
            r="6"
            fill="#FAF7FB"
            stroke="#F43F5E"
            strokeWidth="4"
            filter={`url(#${glowId})`}
            animate={{
              cx: dotPoints.map((point) => point.x),
              cy: dotPoints.map((point) => point.y),
              opacity: [0.75, 1, 0.75],
            }}
            transition={{
              duration: 4.6,
              repeat: Infinity,
              ease: "easeInOut",
            }}
          />
        )}

        {reduceMotion && lastPoint && (
          <circle
            cx={lastPoint.x}
            cy={lastPoint.y}
            r="6"
            fill="#FAF7FB"
            stroke="#F43F5E"
            strokeWidth="4"
          />
        )}

        {showLabels &&
          chart.points.map((point, index) => (
            <g key={`${point.label}-${index}`}>
              <circle cx={point.x} cy={point.y} r="3" fill="#D6A85A" opacity="0.85" />
              <text
                x={point.x}
                y={VIEWBOX_HEIGHT - 10}
                textAnchor="middle"
                className="fill-muted-foreground text-[10px]"
              >
                {point.label}
              </text>
            </g>
          ))}
      </svg>
    </div>
  );
}
