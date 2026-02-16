"use client";

import { useRouter } from "next/navigation";
import { type DragEvent, useState } from "react";

type UploadResponse = { session_id: string; csv_name: string; path: string; datasets?: string[] };

const apiBase = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8105";

export default function Home() {
  const router = useRouter();
  const [files, setFiles] = useState<File[]>([]);
  const [path, setPath] = useState("");
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string>("");
  const [dragActive, setDragActive] = useState(false);

  function collectCsvFiles(fileList: FileList | null): File[] {
    return Array.from(fileList || []).filter((f) => f.name.toLowerCase().endsWith(".csv"));
  }

  function handleDrop(e: DragEvent<HTMLLabelElement>) {
    e.preventDefault();
    setDragActive(false);
    const dropped = collectCsvFiles(e.dataTransfer.files);
    if (dropped.length > 0) setFiles(dropped);
  }

  async function handleUpload() {
    setError("");
    if (files.length > 0 && path.trim()) {
      setError("Choose either CSV file(s) or a server path, not both.");
      return;
    }
    if (files.length === 0 && !path.trim()) {
      setError("Choose a CSV file or enter a path on the server.");
      return;
    }

    setUploading(true);
    try {
      let final: UploadResponse | null = null;
      if (files.length > 0) {
        if (files.length > 1) {
          const form = new FormData();
          for (const file of files) form.append("files", file);
          form.append("mode", "append");
          const res = await fetch(`${apiBase}/upload/batch`, { method: "POST", body: form });
          if (!res.ok) throw new Error(await res.text());
          final = await res.json();
        } else {
          const form = new FormData();
          form.append("file", files[0]);
          form.append("mode", "append");
          const res = await fetch(`${apiBase}/upload`, { method: "POST", body: form });
          if (!res.ok) throw new Error(await res.text());
          final = await res.json();
        }
      } else {
        const form = new FormData();
        form.append("path", path.trim());
        form.append("mode", "append");
        const res = await fetch(`${apiBase}/upload`, { method: "POST", body: form });
        if (!res.ok) throw new Error(await res.text());
        final = await res.json();
      }
      if (!final) throw new Error("Upload failed.");
      setFiles([]);
      setPath("");
      // Pass session only; chat page hydrates datasets from backend.
      const search = new URLSearchParams({ session: final.session_id }).toString();
      router.push(`/chat?${search}`);
    } catch (err: any) {
      setError(err?.message || "Upload failed.");
    } finally {
      setUploading(false);
    }
  }

  return (
    <main className="mx-auto flex min-h-screen w-full max-w-screen-2xl flex-col gap-10 px-6 py-12">
      <section className="card grid gap-10 p-10 lg:grid-cols-2">
        <div className="space-y-6">
          <p className="text-sm uppercase tracking-[0.25em] text-indigo-600">Data Concierge</p>
          <h1 className="text-4xl font-semibold leading-tight text-slate-900 lg:text-5xl">
            Upload one or more CSVs, then chat with an agent that thinks in Python.
          </h1>
          <p className="text-lg text-slate-700">
            We load your file into a private Python sandbox, so you can ask real questions and get
            code-backed answers - no notebooks required.
          </p>

          <div className="grid gap-3 text-sm text-slate-700 sm:grid-cols-2">
            <div className="glass rounded-xl p-4">
              <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Limits</p>
              <p className="mt-1 font-semibold text-slate-900">400 MB max CSV · 100k token budget</p>
            </div>
            <div className="glass rounded-xl p-4">
              <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Safety</p>
              <p className="mt-1 font-semibold text-slate-900">Sandboxed Python with resource caps</p>
            </div>
          </div>
        </div>

        <div className="glass relative overflow-hidden rounded-2xl border border-slate-200 p-8 shadow-2xl bg-white">
          <div className="pointer-events-none absolute -left-12 -top-12 h-40 w-40 rounded-full bg-indigo-100 blur-3xl" />
          <div className="pointer-events-none absolute -right-10 bottom-6 h-32 w-32 rounded-full bg-emerald-100 blur-3xl" />

          <div className="relative space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Step 1</p>
                <h2 className="text-xl font-semibold text-slate-900">Drop your CSV</h2>
              </div>
              <span className="rounded-full bg-indigo-50 px-3 py-1 text-xs text-indigo-700">Instant</span>
            </div>

            <label
              onDragOver={(e) => {
                e.preventDefault();
                if (!dragActive) setDragActive(true);
              }}
              onDragLeave={() => setDragActive(false)}
              onDrop={handleDrop}
              className={`block cursor-pointer rounded-xl border border-dashed p-4 text-sm text-slate-800 transition ${
                dragActive
                  ? "border-indigo-400 bg-indigo-50/70"
                  : "border-slate-300 bg-slate-50 hover:border-indigo-300 hover:bg-indigo-50/60"
              }`}
            >
              <input
                type="file"
                accept=".csv"
                multiple
                className="hidden"
                onChange={(e) => setFiles(collectCsvFiles(e.target.files))}
              />
              {files.length > 0 ? (
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-slate-900">{files.length} file(s) selected</p>
                    <p className="text-xs text-slate-500">{files.slice(0, 4).map((f) => f.name).join(", ")}</p>
                  </div>
                  <button
                    type="button"
                    onClick={() => setFiles([])}
                    className="text-xs text-indigo-600 hover:text-indigo-800"
                  >
                    Clear
                  </button>
                </div>
              ) : (
                <div className="flex items-center gap-3 text-slate-900">
                  <span className="inline-flex h-10 w-10 items-center justify-center rounded-full bg-indigo-100 text-lg">
                    📄
                  </span>
                  <div>
                    <p className="font-medium">Choose CSV file(s)</p>
                    <p className="text-xs text-slate-500">Drag multiple files here or click to browse</p>
                  </div>
                </div>
              )}
            </label>

            <div className="text-center text-xs uppercase tracking-[0.25em] text-slate-500">or</div>

            <input
              value={path}
              onChange={(e) => setPath(e.target.value)}
              placeholder="Path on server (e.g. /data/sales.csv)"
              className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 focus:border-indigo-400 focus:outline-none"
            />

            {error && <p className="text-sm text-rose-500">{error}</p>}
            <p className="text-xs text-slate-600">
              Multi-CSV: select multiple files to append into one session.
            </p>

            <button
              onClick={handleUpload}
              disabled={uploading}
              className="mt-2 w-full rounded-lg bg-indigo-500 px-4 py-3 text-sm font-semibold text-white shadow-lg transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-70"
            >
              {uploading ? "Uploading..." : "Upload and start chatting"}
            </button>
          </div>
        </div>
      </section>
    </main>
  );
}
