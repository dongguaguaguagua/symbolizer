import { useI18n } from "@/lib/i18n";
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
    const { t } = useI18n();

    const color =
        tone === "blue"
            ? "text-blue-700 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-500"
            : "text-green-700 dark:text-green-400 hover:text-green-800 dark:hover:text-green-500";

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
                    "bg-gray-700 text-white",
                    "opacity-0 group-hover:opacity-100",
                    "transition-opacity",
                    "z-50",
                ].join(" ")}
            >
                {copied ? t("copySuccess") : t("copy")}
            </span>
        </button>
    );
}
