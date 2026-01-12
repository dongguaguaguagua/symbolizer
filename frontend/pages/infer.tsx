import { useState, useCallback, useEffect, useMemo } from "react";
import DrawCanvas from "@/components/DrawCanvas";
import { inferTop5, preloadRuntimeAndModel } from "@/lib/onnx";
import SvgSymbol from "@/components/SvgSymbol";

type Mapping = {
    symbol: string;
    unicode: string;
    svg: string;
};

function unicodeLiteralToChar(unicode: string) {
    // "\\u{03B1}" -> "α"
    try {
        const hex = unicode.match(/\{([0-9A-Fa-f]+)\}/)?.[1];
        if (!hex) return "?";
        return String.fromCodePoint(parseInt(hex, 16));
    } catch {
        return "?";
    }
}

async function copyToClipboard(text: string) {
    await navigator.clipboard.writeText(text);
}

function CopyableMono({
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

            {/* Tooltip 放上方，且不拦截点击 */}
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

export default function InferPage() {
    const [top5, setTop5] = useState<{ i: number; v: number }[]>([]);
    const [inferLoading, setInferLoading] = useState(false);
    const [bootLoading, setBootLoading] = useState(true);

    const [mappings, setMappings] = useState<Record<string, Mapping>>({});
    const [clearSignal, setClearSignal] = useState(0);

    // 页面加载：先拉 mappings + 预热 ORT/模型
    useEffect(() => {
        let cancelled = false;

        (async () => {
            try {
                const [mp] = await Promise.all([
                    fetch("/mappings.json", { cache: "force-cache" }).then(
                        (r) => r.json(),
                    ),
                    preloadRuntimeAndModel(),
                ]);

                if (!cancelled) setMappings(mp);
            } catch (e) {
                console.error("boot failed:", e);
            } finally {
                if (!cancelled) setBootLoading(false);
            }
        })();

        return () => {
            cancelled = true;
        };
    }, []);

    const onChange = useCallback(async (gray: Uint8Array) => {
        try {
            setInferLoading(true);
            const res = await inferTop5(gray);
            setTop5(res);
        } catch (e) {
            console.error("infer error", e);
        } finally {
            setInferLoading(false);
        }
    }, []);

    const clearCanvas = () => {
        setTop5([]);
        setClearSignal((x) => x + 1);
    };

    const statusText = useMemo(() => {
        if (bootLoading) return "正在加载模型与运行时（首次会稍慢）…";
        if (inferLoading) return "正在推理…";
        return "就绪";
    }, [bootLoading, inferLoading]);

    return (
        <div className="min-h-screen bg-gray-50 p-8">
            <div className="max-w-5xl mx-auto grid grid-cols-1 md:grid-cols-2 gap-6 md:gap-10">
                {/* 左侧：画板 */}
                <div className="bg-white rounded-xl shadow p-6">
                    <div className="flex items-center justify-between mb-4">
                        <h1 className="text-xl font-semibold">手写符号输入</h1>
                        <button
                            onClick={clearCanvas}
                            className="px-3 py-2 text-sm bg-gray-900 text-white rounded hover:bg-gray-800"
                        >
                            清空画板
                        </button>
                    </div>

                    <DrawCanvas onChange={onChange} clearSignal={clearSignal} />

                    <p className="mt-3 text-sm text-gray-500">松开鼠标识别</p>
                </div>

                {/* 右侧：结果 */}
                <div className="bg-white rounded-xl shadow p-6">
                    <div className="flex items-center justify-between mb-4">
                        <h2 className="text-xl font-semibold">预测结果</h2>
                        <div className="text-sm text-gray-500">
                            {statusText}
                        </div>
                    </div>

                    {top5.length === 0 ? (
                        <div className="text-gray-400 text-sm">
                            请在左侧写一个符号
                        </div>
                    ) : (
                        <ul className="space-y-3">
                            {top5.map((p) => {
                                const m = mappings[String(p.i)];
                                const symbol = m?.symbol ?? "Unknown";
                                const unicode = m?.unicode ?? "N/A";

                                return (
                                    <li
                                        key={p.i}
                                        className="border rounded-lg px-4 py-3"
                                    >
                                        <div className="flex items-center justify-between gap-3 flex-wrap sm:flex-nowrap">
                                            {/* 左侧：文本 */}
                                            <div className="min-w-0">
                                                <div className="font-medium">
                                                    #{p.i}
                                                </div>

                                                <div className="mt-1 space-y-2">
                                                    <CopyableMono
                                                        text={symbol}
                                                        tone="blue"
                                                    />
                                                    <CopyableMono
                                                        text={unicode}
                                                        tone="green"
                                                    />
                                                </div>
                                            </div>

                                            <div className="text-4xl">
                                                {m?.svg ? (
                                                    <SvgSymbol
                                                        base64={m.svg}
                                                        size="2.25em"
                                                    />
                                                ) : (
                                                    unicodeLiteralToChar(
                                                        m.unicode,
                                                    )
                                                )}
                                            </div>

                                            {/* 右侧：概率 */}
                                            <div className="text-sm text-gray-600 tabular-nums">
                                                {(p.v * 100).toFixed(2)}%
                                            </div>
                                        </div>
                                    </li>
                                );
                            })}
                        </ul>
                    )}
                </div>
            </div>
        </div>
    );
}
