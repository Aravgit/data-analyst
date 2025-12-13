"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";

type UploadResponse = { session_id: string; csv_name: string; path: string };

const apiBase = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

export default function Home() {
  const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
  const [path, setPath] = useState("");
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string>("");

  async function handleUpload() {
    setError("");
    if (!file && !path.trim()) {
      setError("Choose a CSV file or enter a path on the server.");
      return;
    }

    setUploading(true);
    try {
      const form = new FormData();
      if (file) form.append("file", file);
      if (path.trim()) form.append("path", path.trim());

      const res = await fetch(`${apiBase}/upload`, { method: "POST", body: form });
      if (!res.ok) throw new Error(await res.text());
      const data: UploadResponse = await res.json();

      // Pass session + csv via query so chat page can hydrate immediately
      const search = new URLSearchParams({ session: data.session_id, csv: data.csv_name }).toString();
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
            Upload a CSV, then chat with an agent that thinks in Python.
          </h1>
          <p className="text-lg text-slate-700">
            We load your file into a private Python sandbox, so you can ask real questions and get
            code-backed answers - no notebooks required.
          </p>

          <div className="grid gap-3 text-sm text-slate-700 sm:grid-cols-2">
            <div className="glass rounded-xl p-4">
              <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Limits</p>
              <p className="mt-1 font-semibold text-slate-900">400 MB max CSV · 20k token budget</p>
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

            <label className="block cursor-pointer rounded-xl border border-dashed border-slate-300 bg-slate-50 p-4 text-sm text-slate-800 transition hover:border-indigo-300 hover:bg-indigo-50/60">
              <input
                type="file"
                accept=".csv"
                className="hidden"
                onChange={(e) => setFile(e.target.files?.[0] || null)}
              />
              {file ? (
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-slate-900">{file.name}</p>
                    <p className="text-xs text-slate-500">Ready to upload</p>
                  </div>
                  <button
                    type="button"
                    onClick={() => setFile(null)}
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
                    <p className="font-medium">Choose a CSV file</p>
                    <p className="text-xs text-slate-500">Drag & drop or click to browse</p>
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
