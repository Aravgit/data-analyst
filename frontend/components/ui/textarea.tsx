import * as React from "react";
import { cn } from "../../lib/utils";

export interface TextareaProps extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {}

export const Textarea = React.forwardRef<HTMLTextAreaElement, TextareaProps>(function Textarea(
  { className, ...props },
  ref
) {
  return (
    <textarea
      ref={ref}
      className={cn(
        "flex w-full rounded-md border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-white shadow-sm transition focus:border-indigo-400 focus:outline-none",
        className
      )}
      {...props}
    />
  );
});
