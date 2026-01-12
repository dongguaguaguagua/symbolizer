"use client";
import { useState } from "react";

async function copyToClipboard(text: string) {
    await navigator.clipboard.writeText(text);
}

export default function CopyableMono({
    text,
    tone = "blue",
}: {
    text: string;
    tone?: "blue" | "green";
}) {
    const [copied, setCopied] = useState(false);

    const color =
        tone === "blue"
            ? "text-blue-700 hover:text-blue-800"
            : "text-green-700 hover:text-green-800";

    const onClick = async () => {
        try {
            await copyToClipboard(text);
            setCopied(true);
            window.setTimeout(() => setCopied(false), 1200);
        } catch (e) {
            console.error("Copy failed:", e);
        }
    };

    return (
        <button
            type="button"
            onClick={onClick}
            className={[
                "relative group",
                "font-mono text-sm",
                "text-left",
                "cursor-pointer",
                "hover:underline",
                "select-text",
                color,
                "leading-6",
            ].join(" ")}
        >
            {text}

            <span
                className={[
                    "pointer-events-none",
                    "absolute left-0 bottom-full mb-1",
                    "whitespace-nowrap",
                    "rounded px-2 py-1 text-xs",
                    "bg-black text-white",
                    "opacity-0 group-hover:opacity-100",
                    "transition-opacity",
                    "z-50",
                ].join(" ")}
            >
                {copied ? "复制成功" : "点击复制"}
            </span>
        </button>
    );
}
