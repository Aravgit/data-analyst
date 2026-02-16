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

type UploadResponse = {
  session_id: string;
  csv_name: string;
  path: string;
  mode?: "append" | "replace_session";
  session_reset?: boolean;
  datasets?: string[];
  active_dataset?: string;
  uploaded_count?: number;
};
type DatasetListResponse = { session_id: string; datasets: string[]; active_dataset?: string };

const apiBase = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8105";

type DataFrameEvent = {
  df_name: string;
  logical_name?: string;
  columns: string[];
  rows: any[];
  row_count?: number;
  dtypes?: Record<string, string>;
};


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
  const [files, setFiles] = useState<File[]>([]);
  const [path, setPath] = useState("");
  const [input, setInput] = useState("");
  const [thinking, setThinking] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [streamingAssistant, setStreamingAssistant] = useState<{ idx: number; text: string } | null>(null);
  const [datasetNames, setDatasetNames] = useState<string[]>([]);
  const [activeDataset, setActiveDataset] = useState<string>("");
  const [uploadMode, setUploadMode] = useState<"append" | "replace_session">("append");
  const [dataFrames, setDataFrames] = useState<DataFrameEvent[]>([]);
  const [charts, setCharts] = useState<ChartPayload[]>([]);
  const [chartIdx, setChartIdx] = useState(0);
  const [activeTab, setActiveTab] = useState<"summary" | "data" | "charts">("summary");

  useEffect(() => {
    const qsSession = search.get("session");
    if (qsSession && qsSession !== sessionId) {
      setSessionId(qsSession);
    }
  }, [search, sessionId]);

  useEffect(() => {
    if (!sessionId) return;
    hydrateDatasets(sessionId);
  }, [sessionId]);

  const canChat = useMemo(() => Boolean(sessionId), [sessionId]);

  const push = (msg: Message) => setMessages((prev) => [...prev, msg]);
  const appendActivity = (kind: string, detail: string) =>
    setActivity((prev) => [{ kind, detail }, ...prev].slice(0, 50));

  async function parseError(res: Response): Promise<string> {
    try {
      const payload = await res.json();
      const detail = payload?.detail;
      if (typeof detail === "string") return detail;
      if (detail?.message) return detail.message;
      return JSON.stringify(detail || payload);
    } catch {
      return await res.text();
    }
  }

  async function hydrateDatasets(sid: string) {
    try {
      const res = await fetch(`${apiBase}/session/${encodeURIComponent(sid)}/datasets`);
      if (!res.ok) throw new Error(await parseError(res));
      const data: DatasetListResponse = await res.json();
      const names = Array.isArray(data.datasets) ? data.datasets : [];
      setDatasetNames(names);
      setActiveDataset(data.active_dataset || names[0] || "");
      if (messages.length === 1 && messages[0].content.includes("Upload a CSV") && names.length > 0) {
        setMessages([{ role: "assistant", content: `Ready. ${names.length} dataset(s) loaded in this session.` }]);
      }
    } catch (err: any) {
      appendActivity("error", err?.message || "Failed to load dataset list.");
    }
  }

  async function handleUpload() {
    if (files.length > 0 && path.trim()) {
      push({ role: "error", content: "Choose either CSV file(s) or a server path, not both." });
      return;
    }
    if (files.length === 0 && !path.trim()) {
      push({ role: "error", content: "Choose a CSV file or enter a path." });
      return;
    }
    setUploading(true);
    try {
      let finalUpload: UploadResponse | null = null;
      if (files.length > 1) {
        const form = new FormData();
        for (const f of files) form.append("files", f);
        if (sessionId) form.append("session_id", sessionId);
        form.append("mode", uploadMode);
        const res = await fetch(`${apiBase}/upload/batch`, { method: "POST", body: form });
        if (!res.ok) throw new Error(await parseError(res));
        finalUpload = await res.json();
      } else if (files.length === 1) {
        const form = new FormData();
        form.append("file", files[0]);
        if (sessionId) form.append("session_id", sessionId);
        form.append("mode", uploadMode);
        const res = await fetch(`${apiBase}/upload`, { method: "POST", body: form });
        if (!res.ok) throw new Error(await parseError(res));
        finalUpload = await res.json();
      } else {
        const form = new FormData();
        form.append("path", path.trim());
        if (sessionId) form.append("session_id", sessionId);
        form.append("mode", uploadMode);
        const res = await fetch(`${apiBase}/upload`, { method: "POST", body: form });
        if (!res.ok) throw new Error(await parseError(res));
        finalUpload = await res.json();
      }

      if (!finalUpload) throw new Error("Upload failed.");
      const nextSessionId = finalUpload.session_id;
      setSessionId(nextSessionId);
      const names = Array.isArray(finalUpload.datasets) ? finalUpload.datasets : [];
      setDatasetNames(names);
      setActiveDataset(finalUpload.active_dataset || finalUpload.csv_name || names[0] || "");
      const uploadedCount = Number(finalUpload.uploaded_count || (files.length > 0 ? files.length : 1));
      const statusText =
        uploadedCount > 1
          ? `Ready. Loaded ${uploadedCount} CSV files into this session.`
          : `Ready. CSV loaded as '${finalUpload.csv_name}'.`;
      push({ role: "status", content: statusText });
      appendActivity("upload", statusText);

      if (finalUpload.session_reset) {
        setMessages([{ role: "assistant", content: "Upload a CSV to begin, then ask me anything about it." }]);
        setActivity([]);
        setInput("");
        setDataFrames([]);
        setCharts([]);
        setChartIdx(0);
      }

      setFiles([]);
      setPath("");
      const searchParams = new URLSearchParams({ session: nextSessionId }).toString();
      router.replace(`/chat?${searchParams}`);
      await hydrateDatasets(nextSessionId);
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
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 60000); // 60 second timeout
    try {
      const payload = { message: text, session_id: sessionId };
      const res = await fetch(`${apiBase}/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: controller.signal
      });
      clearTimeout(timeoutId);
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
      clearTimeout(timeoutId);
      const errorMsg = err?.name === "AbortError"
        ? "Request timed out. Please try a simpler query."
        : (err?.message || "Agent error.");
      push({ role: "error", content: errorMsg });
      appendActivity("error", errorMsg);
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
        push({ role: "status", content: "Token budget reached (100k). Continuing with compacted history." });
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
      const windowTok = Number(payload.total_tokens || 0);
      const lifeTok = Number(payload.lifetime_total_tokens ?? windowTok);
      setTokens({ input: inTok, output: outTok, total: lifeTok });
      appendActivity(
        "tokens",
        lifeTok !== windowTok
          ? `In ${inTok}, out ${outTok}, session total ${lifeTok} (window ${windowTok})`
          : `In ${inTok}, out ${outTok}, session total ${lifeTok}`
      );
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
        let parsed = typeof raw === "string" ? decryptPayload(raw) : raw;
        if (typeof parsed === "string") {
          try {
            parsed = JSON.parse(parsed);
          } catch {
            /* ignore */
          }
        }
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
    if (event === "done") {
      appendActivity("status", "Agent turn complete");
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
    const allowedTypes = ["bar", "column", "stacked_column", "line", "scatter"];
    if (!chart_type || !x_field || series.length === 0) return { ok: false, reason: "missing chart fields" };
    if (!allowedTypes.includes(chart_type)) return { ok: false, reason: `unsupported chart_type '${chart_type}'` };
    if (chart_type === "scatter" && series.length !== 1) return { ok: false, reason: "scatter requires exactly one series" };
    return {
      ok: true,
      chart: {
        chart_type,
        x_field,
        x_label: raw.x_label,
        y_label: raw.y_label,
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
            <h2 className="text-lg font-semibold text-slate-900">Datasets</h2>
            <p className="text-sm text-slate-600">{datasetNames.length ? `${datasetNames.length} loaded` : "None loaded"}</p>
            <p className="text-xs text-slate-500">Active: {activeDataset || "none"}</p>
          </div>
          <div className="space-y-3">
            <input
              type="file"
              accept=".csv"
              multiple
              onChange={(e) => setFiles(Array.from(e.target.files || []))}
              className="block w-full text-sm text-slate-700 file:mr-4 file:rounded-md file:border file:border-slate-300 file:bg-white file:px-3 file:py-2 file:text-slate-800 hover:file:bg-slate-50"
            />
            {files.length > 0 && (
              <div className="rounded-md border border-slate-200 bg-slate-50 p-2 text-xs text-slate-700">
                <p className="font-semibold text-slate-900">{files.length} file(s) selected</p>
                <p className="mt-1 break-words">{files.slice(0, 6).map((f) => f.name).join(", ")}</p>
              </div>
            )}
            <Input
              value={path}
              onChange={(e) => setPath(e.target.value)}
              placeholder="Path on server (e.g. /data/sales.csv)"
              className="bg-white text-slate-900 border-slate-300"
            />
            <div className="grid grid-cols-2 gap-2 text-xs">
              <button
                type="button"
                onClick={() => setUploadMode("append")}
                className={cn(
                  "rounded border px-2 py-1",
                  uploadMode === "append" ? "border-slate-900 bg-slate-100" : "border-slate-300 bg-white"
                )}
              >
                Append
              </button>
              <button
                type="button"
                onClick={() => setUploadMode("replace_session")}
                className={cn(
                  "rounded border px-2 py-1",
                  uploadMode === "replace_session" ? "border-slate-900 bg-slate-100" : "border-slate-300 bg-white"
                )}
              >
                Replace all
              </button>
            </div>
            <Button onClick={handleUpload} disabled={uploading} className="w-full">
              {uploading ? "Uploading..." : uploadMode === "append" ? "Add dataset(s)" : "Replace session data"}
            </Button>
            {datasetNames.length > 0 && (
              <div className="rounded-md border border-slate-200 bg-slate-50 p-2 text-xs text-slate-700">
                <p className="font-semibold text-slate-900">Available datasets</p>
                <p className="mt-1 break-words">{datasetNames.join(", ")}</p>
              </div>
            )}
            <div className="rounded-lg border border-slate-200 bg-white p-3 text-xs text-slate-600">
              <p className="font-semibold text-slate-900">Tips</p>
              <ul className="mt-1 space-y-1">
                <li>- Max CSV size 400 MB.</li>
                <li>- Session stays via URL params.</li>
                <li>- Append adds file(s) to session, Replace all resets session data.</li>
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
                  <PaginatedTable frame={dataFrames[0]} />
                )
              )}

              {activeTab === "charts" && (
                charts.length === 0 ? (
                  <div className="rounded-xl border border-slate-200 bg-white p-4 text-sm text-slate-600">
                    No charts yet. Ask for a bar, column, stacked column, line, or scatter chart after loading data.
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
              <p className="text-[10px] uppercase tracking-wide text-slate-500">Session Total</p>
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
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}
