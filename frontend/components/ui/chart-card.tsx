"use client";

import React from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  BarChart,
  Bar,
  ScatterChart,
  Scatter,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
} from "recharts";

type ChartSeries = { name: string; y_field: string; color?: string };
type ChartPayload = {
  title?: string;
  chart_type: "bar" | "column" | "stacked_column" | "line" | "scatter";
  x_field: string;
  x_label?: string;
  y_label?: string;
  series: ChartSeries[];
  data: any[];
  note?: string;
};

const palette = ["#0f766e", "#1d4ed8", "#7c3aed", "#ea580c", "#16a34a", "#be123c", "#0ea5e9", "#9333ea", "#f59e0b", "#475569"];

function colorAt(idx: number, fallback?: string) {
  return fallback || palette[idx % palette.length];
}

function formatXTick(value: any) {
  // Treat ISO-like strings or long numbers as dates; otherwise passthrough
  if (typeof value === "string") {
    const looksIso = /^\d{4}-\d{2}-\d{2}/.test(value);
    if (looksIso) {
      const d = new Date(value);
      if (!isNaN(d.getTime())) {
        return d.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
      }
    }
  }
  if (typeof value === "number" && value > 1e9) {
    const d = new Date(value);
    if (!isNaN(d.getTime())) {
      return d.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
    }
  }
  return value;
}

function formatTooltipLabel(label: any) {
  const formatted = formatXTick(label);
  if (formatted === label && typeof label === "string" && label.length > 40) {
    return `${label.slice(0, 37)}…`;
  }
  return formatted;
}

export default function ChartCard({ chart }: { chart: ChartPayload }) {
  if (!chart || !chart.data || chart.data.length === 0) {
    return <div className="rounded-lg border border-slate-200 bg-slate-50 p-4 text-sm text-slate-600">Chart unavailable.</div>;
  }

  const xLabel = chart.x_label || chart.x_field;
  const yLabel =
    chart.y_label ||
    (chart.series.length === 1 ? chart.series[0].name || chart.series[0].y_field : undefined);

  const isScatter = chart.chart_type === "scatter";
  const scatterYKey = chart.chart_type === "scatter" ? chart.series[0]?.y_field : undefined;

  const commonAxes = (
    <>
      <CartesianGrid stroke="#e2e8f0" strokeDasharray="3 3" />
      <XAxis
        dataKey={chart.x_field}
        type={isScatter ? "number" : undefined}
        tick={{ fill: "#475569", fontSize: 12 }}
        label={{ value: xLabel, position: "insideBottom", offset: -6, style: { fill: "#475569", fontSize: 12, fontWeight: 500 } }}
        tickFormatter={formatXTick}
      />
      <YAxis
        dataKey={scatterYKey}
        type={isScatter ? "number" : undefined}
        tick={{ fill: "#475569", fontSize: 12 }}
        label={
          yLabel
            ? { value: yLabel, angle: -90, position: "insideLeft", offset: -8, style: { fill: "#475569", fontSize: 12, fontWeight: 500 } }
            : undefined
        }
      />
      <Tooltip
        cursor={{ stroke: "#cbd5e1", strokeDasharray: "3 3" }}
        formatter={(value: any, name: any) => [value, name]}
        labelFormatter={formatTooltipLabel}
      />
      {chart.series.length > 1 && <Legend />}
    </>
  );

  const renderCartesian = () => {
    if (chart.chart_type === "bar" || chart.chart_type === "column") {
      return (
        <BarChart data={chart.data} margin={{ top: 12, right: 16, left: 8, bottom: 32 }}>
          {commonAxes}
          {chart.series.map((s, idx) => (
            <Bar key={s.y_field} dataKey={s.y_field} name={s.name} fill={colorAt(idx, s.color)} />
          ))}
        </BarChart>
      );
    }
    if (chart.chart_type === "stacked_column") {
      return (
        <BarChart data={chart.data} margin={{ top: 12, right: 16, left: 8, bottom: 32 }}>
          {commonAxes}
          {chart.series.map((s, idx) => (
            <Bar
              key={s.y_field}
              dataKey={s.y_field}
              name={s.name}
              stackId="stack"
              fill={colorAt(idx, s.color)}
            />
          ))}
        </BarChart>
      );
    }
    if (chart.chart_type === "scatter") {
      return (
        <ScatterChart data={chart.data} margin={{ top: 12, right: 16, left: 8, bottom: 32 }}>
          {commonAxes}
          {chart.series.map((s, idx) => (
            <Scatter key={s.y_field} name={s.name} data={chart.data} dataKey={s.y_field} fill={colorAt(idx, s.color)} />
          ))}
        </ScatterChart>
      );
    }
    return (
      <LineChart data={chart.data} margin={{ top: 12, right: 16, left: 8, bottom: 32 }}>
        {commonAxes}
        {chart.series.map((s, idx) => (
          <Line
            key={s.y_field}
            type="monotone"
            dataKey={s.y_field}
            name={s.name}
            stroke={colorAt(idx, s.color)}
            strokeWidth={2}
            dot={{ r: 2, stroke: colorAt(idx, s.color), strokeWidth: 1, fill: "#fff" }}
            activeDot={{ r: 4, stroke: colorAt(idx, s.color), strokeWidth: 2, fill: "#fff" }}
          />
        ))}
      </LineChart>
    );
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Chart</p>
          <h3 className="text-lg font-semibold text-slate-900">{chart.title || "Chart"}</h3>
        </div>
        {chart.note && <div className="text-xs text-slate-600">{chart.note}</div>}
      </div>
      {chart.series.length === 1 && (
        <div className="text-xs font-medium text-slate-600">{chart.series[0].name || chart.series[0].y_field}</div>
      )}
      <div className="h-[360px] w-full">
        <ResponsiveContainer>
          {renderCartesian()}
        </ResponsiveContainer>
      </div>
    </div>
  );
}
