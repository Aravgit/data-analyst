"use client";

import React, { useEffect, useMemo, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { Button } from "../../components/ui/button";
import { Input } from "../../components/ui/input";
import { Card } from "../../components/ui/card";
import { cn } from "../../lib/utils";
import dynamic from "next/dynamic";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { decryptPayload } from "../../lib/decrypt";
import PaginatedTable from "../../components/ui/paginated-table";

const ChartCard = dynamic(() => import("../../components/ui/chart-card"), { ssr: false });

type Role = "user" | "assistant" | "status" | "error";
type Message = { role: Role; content: string };

type UploadResponse = { session_id: string; csv_name: string; path: string };
type ChatResponse = { session_id: string; reply: string; total_tokens: number; status: string };

const apiBase = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

type DataFrameEvent = {
  df_name: string;
  logical_name?: string;
  columns: string[];
  rows: any[];
  row_count?: number;
  dtypes?: Record<string, string>;
};

type DownloadEvent = {
  df_name: string;
  logical_name?: string;
  rows?: number;
  columns?: number;
  path: string;
};

type ChartSeries = { name: string; y_field: string; color?: string };
type ChartPayload = {
  title?: string;
  chart_type: "bar" | "line" | "area" | "pie";
  x_field: string;
  series: ChartSeries[];
  data: any[];
  note?: string;
  df_name?: string;
  logical_name?: string;
};

export default function ChatClient() {
  const search = useSearchParams();
  const router = useRouter();

  const [sessionId, setSessionId] = useState("");
  const [messages, setMessages] = useState<Message[]>([
    { role: "assistant", content: "Upload a CSV to begin, then ask me anything about it." }
  ]);
  const [activityOpen, setActivityOpen] = useState(true);
  const [activity, setActivity] = useState<Array<{ kind: string; detail: string }>>([]);
  const [tokens, setTokens] = useState<{ input: number; output: number; total: number }>({
    input: 0,
    output: 0,
    total: 0
  });
  const [file, setFile] = useState<File | null>(null);
  const [path, setPath] = useState("");
  const [input, setInput] = useState("");
  const [thinking, setThinking] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [streamingAssistant, setStreamingAssistant] = useState<{ idx: number; text: string } | null>(null);
  const [csvName, setCsvName] = useState<string>("");
  const [dataFrames, setDataFrames] = useState<DataFrameEvent[]>([]);
  const [downloads, setDownloads] = useState<DownloadEvent[]>([]);
  const [charts, setCharts] = useState<ChartPayload[]>([]);
  const [chartIdx, setChartIdx] = useState(0);
  const [activeTab, setActiveTab] = useState<"summary" | "data" | "charts">("summary");

  useEffect(() => {
    const qsSession = search.get("session");
    const csv = search.get("csv");
    if (qsSession && !sessionId) {
      setSessionId(qsSession);
      setCsvName(csv || "");
      setMessages([{ role: "assistant", content: `Ready. CSV loaded as '${csv || "your file"}'.` }]);
    }
  }, [search, sessionId]);

  const canChat = useMemo(() => Boolean(sessionId), [sessionId]);

  const push = (msg: Message) => setMessages((prev) => [...prev, msg]);
  const appendActivity = (kind: string, detail: string) =>
    setActivity((prev) => [{ kind, detail }, ...prev].slice(0, 50));

  async function handleUpload() {
    if (!file && !path.trim()) {
      push({ role: "error", content: "Choose a CSV file or enter a path." });
      return;
    }
    setUploading(true);
    try {
      const form = new FormData();
      if (file) form.append("file", file);
      if (path.trim()) form.append("path", path.trim());
      if (sessionId) form.append("session_id", sessionId);

      const res = await fetch(`${apiBase}/upload`, { method: "POST", body: form });
      if (!res.ok) throw new Error(await res.text());
      const data: UploadResponse = await res.json();
      setSessionId(data.session_id);
      setCsvName(data.csv_name);
      push({ role: "status", content: `Ready. CSV loaded as '${data.csv_name}'.` });
      appendActivity("upload", `Loaded ${data.csv_name}`);

      if ((data as any).session_reset) {
        setMessages([{ role: "assistant", content: "Upload a CSV to begin, then ask me anything about it." }]);
        setActivity([]);
        setInput("");
        setCharts([]);
        setChartIdx(0);
      }

      const searchParams = new URLSearchParams({ session: data.session_id, csv: data.csv_name }).toString();
      router.replace(`/chat?${searchParams}`);
    } catch (err: any) {
      push({ role: "error", content: err?.message || "Upload failed." });
    } finally {
      setUploading(false);
    }
  }

  async function sendMessage(e: React.FormEvent) {
    e.preventDefault();
    if (!input.trim()) return;
    if (!canChat) {
      push({ role: "error", content: "Upload a CSV first." });
      return;
    }
    const text = input.trim();
    setInput("");
    push({ role: "user", content: text });
    setThinking(true);
    appendActivity("status", "Model call started");
    await streamChat(text);
  }

  async function streamChat(text: string) {
    try {
      const payload = { message: text, session_id: sessionId };
      const res = await fetch(`${apiBase}/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      if (!res.ok || !res.body) {
        throw new Error(await res.text());
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      const idx = messages.length;
      push({ role: "assistant", content: "" });
      setStreamingAssistant({ idx, text: "" });

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split("\n\n");
        buffer = parts.pop() || "";
        for (const part of parts) await handleSseEvent(part);
      }
      if (buffer.trim().length) await handleSseEvent(buffer);
    } catch (err: any) {
      push({ role: "error", content: err?.message || "Agent error." });
      appendActivity("error", err?.message || "Agent error.");
    } finally {
      setThinking(false);
      setStreamingAssistant(null);
    }
  }

  async function handleSseEvent(block: string) {
    const lines = block.split("\n");
    let event = "message";
    let data = "";
    for (const line of lines) {
      if (line.startsWith("event:")) event = line.replace("event:", "").trim();
      else if (line.startsWith("data:")) data += line.replace("data:", "").trim();
    }
    let payload: any = data;
    try {
      payload = JSON.parse(data);
    } catch {
      /* plain */
    }

    if (event === "partial") {
      if (streamingAssistant) {
        const updated = (streamingAssistant.text + " " + (payload.text || payload)).trim();
        setStreamingAssistant({ ...streamingAssistant, text: updated });
        setMessages((prev) => prev.map((m, i) => (i === streamingAssistant.idx ? { ...m, content: updated } : m)));
      }
      return;
    }
    if (event === "reply") {
      const replyText = payload.text || "";
      const status = payload.status || "ok";
      if (streamingAssistant) {
        setMessages((prev) =>
          prev.map((m, i) => (i === streamingAssistant.idx ? { ...m, content: replyText } : m))
        );
      } else {
        push({ role: "assistant", content: replyText });
      }
      if (status === "token_limit") {
        push({ role: "status", content: "Token budget reached (50k). Continuing with compacted history." });
      }
      appendActivity("reply", "Reply received");
      return;
    }
    if (event === "status") {
      appendActivity("status", `Stage: ${payload.stage || payload}`);
      return;
    }
    if (event === "tool_call") {
      appendActivity("tool", `Calling ${payload.name || ""}`);
      return;
    }
    if (event === "tool_result") {
      appendActivity("tool", `Tool result (${payload.result_kind || "ok"})`);
      return;
    }
    if (event === "token_usage") {
      const inTok = Number(payload.input_tokens || 0);
      const outTok = Number(payload.output_tokens || 0);
      const totTok = Number(payload.total_tokens || 0);
      setTokens({ input: inTok, output: outTok, total: totTok });
      appendActivity("tokens", `In ${inTok}, out ${outTok}, total ${totTok}`);
      return;
    }
    if (event === "error") {
      push({ role: "error", content: payload.message || "Agent error." });
      appendActivity("error", payload.message || "Agent error.");
      return;
    }
    if (event === "data_frame") {
      try {
        const raw = payload?.payload ?? payload;
        const parsed = typeof raw === "string" ? decryptPayload(raw) : raw;
        if (!parsed || !Array.isArray(parsed.columns) || !Array.isArray(parsed.rows)) {
          appendActivity("error", "Bad data_frame payload");
          return;
        }
        setDataFrames((prev) => {
          const filtered = prev.filter((p) => p.df_name !== parsed.df_name);
          return [parsed as DataFrameEvent, ...filtered].slice(0, 5);
        });
        appendActivity("data", `Table ${parsed.df_name || "df"} (${parsed.row_count || parsed.rows?.length || 0} rows)`);
      } catch (err: any) {
        appendActivity("error", "Failed to decrypt table");
      }
      return;
    }
    if (event === "chart") {
      try {
        const raw = payload?.payload ?? payload;
        const parsed = typeof raw === "string" ? decryptPayload(raw) : raw;
        const validated = validateChart(parsed);
        if (validated.ok && validated.chart) {
          const chart = validated.chart;
          setCharts((prev) => {
            const existingIdx = prev.findIndex(
              (c) => (c.title || c.logical_name || "") === (chart.title || chart.logical_name || "")
            );
            let next: ChartPayload[] = existingIdx >= 0 ? [...prev] : [chart, ...prev];
            if (existingIdx >= 0) next[existingIdx] = chart;
            setChartIdx(0);
            return next.slice(0, 5);
          });
          appendActivity("data", `Chart ${validated.chart.title || validated.chart.logical_name || "chart"}`);
        } else if (validated.reason) {
          appendActivity("error", `Chart rejected: ${validated.reason}`);
        }
      } catch (err: any) {
        appendActivity("error", "Failed to decrypt chart");
      }
      return;
    }
    if (event === "chart_rejected") {
      try {
        const raw = payload?.payload ?? payload;
        const parsed = typeof raw === "string" ? decryptPayload(raw) : raw;
        appendActivity("error", `Chart rejected: ${parsed?.reason || "unknown reason"}`);
      } catch (err: any) {
        appendActivity("error", "Chart rejected (could not decrypt reason)");
      }
      return;
    }
    if (event === "data_download") {
      try {
        const raw = payload?.payload ?? payload;
        const parsed = typeof raw === "string" ? decryptPayload(raw) : raw;
        setDownloads((prev) => {
          const filtered = prev.filter((p) => p.path !== parsed.path);
          return [parsed as DownloadEvent, ...filtered].slice(0, 10);
        });
        appendActivity("data", `Download ready: ${parsed.logical_name || parsed.df_name || "table"}`);
      } catch (err: any) {
        appendActivity("error", "Failed to decrypt download info");
      }
      return;
    }
  }

  const bubble = (role: Role) =>
    cn(
      "max-w-3xl rounded-lg px-4 py-3 text-sm border whitespace-pre-wrap break-words overflow-x-auto",
      role === "user" ? "bg-white text-slate-900 border-slate-200 ml-auto text-right" : "",
      role === "assistant" ? "bg-slate-100 text-slate-900 border-slate-200 mr-auto" : "",
      role === "status" ? "bg-slate-50 text-slate-600 border-slate-200 mr-auto" : "",
      role === "error" ? "bg-rose-50 text-rose-800 border-rose-200 mr-auto" : ""
    );

  const badge = (role: Role) => {
    switch (role) {
      case "user":
        return "You";
      case "assistant":
        return "Agent";
      case "status":
        return "Status";
      case "error":
        return "Error";
    }
  };

  function validateChart(raw: any): { ok: boolean; chart?: ChartPayload; reason?: string } {
    if (!raw || typeof raw !== "object") return { ok: false, reason: "empty chart payload" };
    const chart_type = raw.chart_type;
    const x_field = raw.x_field;
    const data = Array.isArray(raw.data) ? raw.data.slice(0, 200) : [];
    const series = Array.isArray(raw.series)
      ? raw.series
          .slice(0, 10)
          .filter((s: any) => s && s.y_field)
          .map((s: any) => ({ name: s.name || s.y_field, y_field: s.y_field, color: s.color }))
      : [];
    if (!chart_type || !x_field || series.length === 0) return { ok: false, reason: "missing chart fields" };
    return {
      ok: true,
      chart: {
        chart_type,
        x_field,
        series,
        data,
        title: raw.title,
        note: raw.note,
        df_name: raw.df_name,
        logical_name: raw.logical_name,
      },
    };
  }

  return (
    <div className="mx-auto flex min-h-screen w-full max-w-screen-2xl flex-col gap-6 px-4 py-8">
      <div className="grid gap-4 lg:grid-cols-[1.1fr,3.2fr,1fr]">
        {/* Left rail: data selection */}
        <Card className="p-4">
          <div className="mb-3">
            <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Data</p>
            <h2 className="text-lg font-semibold text-slate-900">Current file</h2>
            <p className="text-sm text-slate-600">{csvName || "None loaded"}</p>
          </div>
          <div className="space-y-3">
            <input
              type="file"
              accept=".csv"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
              className="block w-full text-sm text-slate-700 file:mr-4 file:rounded-md file:border file:border-slate-300 file:bg-white file:px-3 file:py-2 file:text-slate-800 hover:file:bg-slate-50"
            />
            <Input
              value={path}
              onChange={(e) => setPath(e.target.value)}
              placeholder="Path on server (e.g. /data/sales.csv)"
              className="bg-white text-slate-900 border-slate-300"
            />
            <Button onClick={handleUpload} disabled={uploading} className="w-full">
              {uploading ? "Uploading..." : sessionId ? "Replace data" : "Load data"}
            </Button>
            <div className="rounded-lg border border-slate-200 bg-white p-3 text-xs text-slate-600">
              <p className="font-semibold text-slate-900">Tips</p>
              <ul className="mt-1 space-y-1">
                <li>- Max CSV size 400 MB.</li>
                <li>- Session stays via URL params.</li>
                <li>- Ask for samples, summaries, plots.</li>
              </ul>
            </div>
          </div>
        </Card>

        {/* Center: chat */}
        <Card className="p-5">
          <div className="mb-4 flex items-center justify-between">
            <div>
              <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Conversation</p>
              <h1 className="text-2xl font-semibold text-slate-900">Chat with your data</h1>
            </div>
            {thinking ? (
              <div className="flex items-center gap-2 text-xs text-emerald-600">
                <span className="h-2 w-2 animate-ping rounded-full bg-emerald-500" />
                Thinking
              </div>
            ) : (
              <div className="text-xs text-slate-500">Ready</div>
            )}
          </div>

          <div className="flex flex-col gap-3">
            <div className="space-y-3">
              <div className="grid w-full grid-cols-3 rounded-md border border-slate-200 bg-slate-50">
                <button
                  onClick={() => setActiveTab("summary")}
                  className={cn(
                    "py-2 text-sm font-semibold",
                    activeTab === "summary" ? "bg-white text-slate-900 shadow-inner" : "text-slate-600"
                  )}
                >
                  Summary
                </button>
                <button
                  onClick={() => setActiveTab("data")}
                  className={cn(
                    "py-2 text-sm font-semibold",
                    activeTab === "data" ? "bg-white text-slate-900 shadow-inner" : "text-slate-600"
                  )}
                >
                  Data
                </button>
                <button
                  onClick={() => setActiveTab("charts")}
                  className={cn(
                    "py-2 text-sm font-semibold",
                    activeTab === "charts" ? "bg-white text-slate-900 shadow-inner" : "text-slate-600"
                  )}
                >
                  Charts
                </button>
              </div>

              {activeTab === "summary" && (
                <div className="min-h-[620px] max-h-[860px] space-y-3 overflow-y-auto rounded-xl border border-slate-200 bg-white p-4 shadow-inner">
                  {messages.map((m, idx) => (
                    <div key={idx} className={cn(bubble(m.role), m.role === "user" ? "text-right" : "text-left")}>
                      <div className="mb-1 text-[11px] uppercase tracking-wide text-slate-500">{badge(m.role)}</div>
                      <div className="prose prose-sm prose-slate max-w-none text-slate-900">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.content}</ReactMarkdown>
                      </div>
                    </div>
                  ))}
                  {messages.length === 0 && (
                    <div className="text-sm text-slate-500">No messages yet - upload and ask a question.</div>
                  )}
                </div>
              )}

              {activeTab === "data" && (
                dataFrames.length === 0 ? (
                  <div className="rounded-xl border border-slate-200 bg-white p-4 text-sm text-slate-600">No data yet.</div>
                ) : (
                  <PaginatedTable frame={dataFrames[0]} download={downloads[0]} />
                )
              )}

              {activeTab === "charts" && (
                charts.length === 0 ? (
                  <div className="rounded-xl border border-slate-200 bg-white p-4 text-sm text-slate-600">
                    No charts yet. Ask for a bar or line chart after loading data.
                  </div>
                ) : (
                  <div className="space-y-3 rounded-xl border border-slate-200 bg-white p-4">
                    {charts.length > 1 && (
                      <div className="flex flex-wrap gap-2 text-sm">
                        {charts.map((c, idx) => (
                          <button
                            key={`${c.title || c.logical_name || "chart"}-${idx}`}
                            onClick={() => setChartIdx(idx)}
                            className={cn(
                              "rounded-md border px-2 py-1",
                              chartIdx === idx ? "border-slate-900 bg-slate-100" : "border-slate-200 bg-white"
                            )}
                          >
                            {c.title || c.logical_name || `Chart ${idx + 1}`}
                          </button>
                        ))}
                      </div>
                    )}
                    <ChartCard chart={charts[chartIdx] || charts[0]} />
                    <div className="flex items-center justify-between text-xs text-slate-600">
                      <div>
                        {charts[chartIdx]?.logical_name || charts[chartIdx]?.df_name || "chart"} ·{" "}
                        {charts[chartIdx]?.data?.length || 0} rows · x: {charts[chartIdx]?.x_field}
                      </div>
                      <button
                        type="button"
                        onClick={() => setActiveTab("data")}
                        className="text-emerald-700 hover:underline"
                      >
                        View as table
                      </button>
                    </div>
                  </div>
                )
              )}
            </div>

            <form onSubmit={sendMessage} className="flex gap-2">
              <Input
                className="flex-1 bg-white border-slate-300"
                placeholder="Ask a question about your data..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                disabled={thinking}
              />
              <Button type="submit" disabled={thinking}>
                Send
              </Button>
            </form>
          </div>
        </Card>

        {/* Right: activity */}
        <Card className="p-4">
          <div className="mb-2 flex items-center justify-between">
            <div>
              <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Activity</p>
              <h3 className="text-sm font-semibold text-slate-900">Model & tools</h3>
            </div>
            <Button variant="ghost" size="sm" onClick={() => setActivityOpen((o) => !o)}>
              {activityOpen ? "Hide" : "Show"}
            </Button>
          </div>
          <div className="mb-3 grid grid-cols-3 gap-2 text-xs text-slate-700">
            <div className="rounded-md border border-slate-200 bg-white p-2">
              <p className="text-[10px] uppercase tracking-wide text-slate-500">Input</p>
              <p className="font-semibold text-slate-900">{tokens.input}</p>
            </div>
            <div className="rounded-md border border-slate-200 bg-white p-2">
              <p className="text-[10px] uppercase tracking-wide text-slate-500">Output</p>
              <p className="font-semibold text-slate-900">{tokens.output}</p>
            </div>
            <div className="rounded-md border border-slate-200 bg-white p-2">
              <p className="text-[10px] uppercase tracking-wide text-slate-500">Total</p>
              <p className="font-semibold text-slate-900">{tokens.total}</p>
            </div>
          </div>
          {activityOpen && (
            <div className="space-y-2 max-h-[520px] overflow-y-auto text-sm text-slate-800">
              {activity.length === 0 && <div className="text-slate-500 text-xs">No activity yet.</div>}
              {activity.map((a, idx) => (
                <div key={idx} className="rounded-md border border-slate-200 bg-white px-2 py-1">
                  <span className="text-xs uppercase text-slate-500 mr-2">{a.kind}</span>
                  <span>{a.detail}</span>
                </div>
              ))}
              {downloads.length > 0 && (
                <div className="rounded-md border border-slate-200 bg-white px-2 py-2">
                  <div className="text-[10px] uppercase tracking-wide text-slate-500 mb-1">Downloads</div>
                  <div className="space-y-1">
                    {downloads.map((d, idx) => (
                      <div key={idx} className="text-xs">
                        <span className="font-semibold">{d.logical_name || d.df_name || "table"}</span>:{" "}
                        <code className="bg-slate-50 px-1 rounded">{d.path}</code>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}
