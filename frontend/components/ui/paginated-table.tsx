import React, { useMemo, useState } from "react";

type Frame = {
  df_name: string;
  logical_name?: string;
  columns: string[];
  rows: any[];
  row_count?: number;
  dtypes?: Record<string, string>;
};

type Download = { path: string; logical_name?: string; df_name?: string };

type Props = { frame: Frame; download?: Download };

const PAGE_SIZE = 10;

export default function PaginatedTable({ frame, download }: Props) {
  const rows = Array.isArray(frame.rows) ? frame.rows : [];
  const [page, setPage] = useState(0);

  const totalPages = useMemo(
    () => Math.max(1, Math.ceil(rows.length / PAGE_SIZE)),
    [rows]
  );

  const pageRows = useMemo(() => {
    const start = page * PAGE_SIZE;
    return rows.slice(start, start + PAGE_SIZE);
  }, [rows, page]);

  const totalRowCount = frame.row_count ?? rows.length;

  return (
    <div className="space-y-3 rounded-xl border border-slate-200 bg-slate-50 p-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Data</p>
          <h2 className="text-lg font-semibold text-slate-900">{frame.logical_name || frame.df_name}</h2>
          <p className="text-xs text-slate-600">
            Showing {pageRows.length} of {rows.length} buffered rows · {frame.columns.length} cols · {totalRowCount} total
            rows
          </p>
        </div>
        {download && (
          <div className="text-xs text-slate-700">
            Download: <code className="text-[11px] bg-white px-1 rounded">{download.path}</code>
          </div>
        )}
      </div>
      <div className="overflow-x-auto rounded-lg border border-slate-200 bg-white">
        <table className="min-w-full text-xs text-slate-900">
          <thead className="bg-slate-100">
            <tr>
              {frame.columns.map((c) => (
                <th key={c} className="px-2 py-2 text-left font-semibold border-b border-slate-200">
                  {c}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {pageRows.map((row, ridx) => (
              <tr key={ridx} className="odd:bg-white even:bg-slate-50">
                {frame.columns.map((c) => (
                  <td key={c} className="px-2 py-1 border-b border-slate-100">
                    {String(row?.[c] ?? "")}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="flex items-center justify-between text-xs text-slate-700">
        <div>
          Page {page + 1} / {totalPages}
        </div>
        <div className="flex gap-2">
          <button
            className="rounded border border-slate-300 bg-white px-2 py-1 disabled:opacity-50"
            onClick={() => setPage((p) => Math.max(0, p - 1))}
            disabled={page === 0}
          >
            Prev
          </button>
          <button
            className="rounded border border-slate-300 bg-white px-2 py-1 disabled:opacity-50"
            onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
            disabled={page >= totalPages - 1}
          >
            Next
          </button>
        </div>
      </div>
    </div>
  );
}
