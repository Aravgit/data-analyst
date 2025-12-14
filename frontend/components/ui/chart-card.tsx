"use client";

import React from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  BarChart,
  Bar,
  AreaChart,
  Area,
  PieChart,
  Pie,
  Cell,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
} from "recharts";

type ChartSeries = { name: string; y_field: string; color?: string };
type ChartPayload = {
  title?: string;
  chart_type: "bar" | "line" | "area" | "pie";
  x_field: string;
  series: ChartSeries[];
  data: any[];
  note?: string;
};

const palette = ["#0f766e", "#1d4ed8", "#7c3aed", "#ea580c", "#16a34a", "#be123c", "#0ea5e9", "#9333ea", "#f59e0b", "#475569"];

function colorAt(idx: number, fallback?: string) {
  return fallback || palette[idx % palette.length];
}

export default function ChartCard({ chart }: { chart: ChartPayload }) {
  if (!chart || !chart.data || chart.data.length === 0) {
    return <div className="rounded-lg border border-slate-200 bg-slate-50 p-4 text-sm text-slate-600">Chart unavailable.</div>;
  }

  const renderCartesian = () => {
    const commonAxes = (
      <>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey={chart.x_field} />
        <YAxis />
        <Tooltip />
        <Legend />
      </>
    );
    if (chart.chart_type === "bar") {
      return (
        <BarChart data={chart.data}>
          {commonAxes}
          {chart.series.map((s, idx) => (
            <Bar key={s.y_field} dataKey={s.y_field} name={s.name} fill={colorAt(idx, s.color)} />
          ))}
        </BarChart>
      );
    }
    if (chart.chart_type === "area") {
      return (
        <AreaChart data={chart.data}>
          {commonAxes}
          {chart.series.map((s, idx) => (
            <Area key={s.y_field} type="monotone" dataKey={s.y_field} name={s.name} fill={colorAt(idx, s.color)} stroke={colorAt(idx, s.color)} />
          ))}
        </AreaChart>
      );
    }
    return (
      <LineChart data={chart.data}>
        {commonAxes}
        {chart.series.map((s, idx) => (
          <Line key={s.y_field} type="monotone" dataKey={s.y_field} name={s.name} stroke={colorAt(idx, s.color)} dot={false} />
        ))}
      </LineChart>
    );
  };

  const renderPie = () => {
    const firstSeries = chart.series[0];
    return (
      <PieChart>
        <Tooltip />
        <Legend />
        <Pie dataKey={firstSeries.y_field} nameKey={chart.x_field} data={chart.data} label>
          {chart.data.map((_entry, idx) => (
            <Cell key={idx} fill={colorAt(idx, firstSeries.color)} />
          ))}
        </Pie>
      </PieChart>
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
      <div className="h-[360px] w-full">
        <ResponsiveContainer>
          {chart.chart_type === "pie" ? renderPie() : renderCartesian()}
        </ResponsiveContainer>
      </div>
    </div>
  );
}
