import { useState, useCallback, useEffect, useMemo } from "react";
import DrawCanvas from "@/components/DrawCanvas";
import { inferTop5, preloadRuntimeAndModel } from "@/lib/onnx";
import SvgSymbol from "@/components/SvgSymbol";
import { useI18n } from "@/lib/i18n";
import CopyableMono from "@/components/CopyableMono";

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

export default function InferPage() {
    const { t } = useI18n();

    const [top5, setTop5] = useState<{ i: number; v: number }[]>([]);
    const [inferLoading, setInferLoading] = useState(false);
    const [bootLoading, setBootLoading] = useState(true);

    const [mappings, setMappings] = useState<Record<string, Mapping>>({});
    const [clearSignal, setClearSignal] = useState(0);
    type InferStatus = "loading" | "infering" | "ready";

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

    const statusKey = useMemo<InferStatus>(() => {
        if (bootLoading) return "loading";
        if (inferLoading) return "infering";
        return "ready";
    }, [bootLoading, inferLoading]);

    return (
        <div className="min-h-screen bg-gray-50 dark:bg-gray-950 p-8">
            <div className="max-w-5xl mx-auto grid grid-cols-1 md:grid-cols-2 gap-6 md:gap-10">
                {/* 左侧：画板 */}
                <div className="bg-white dark:bg-gray-900 rounded-xl shadow p-6">
                    <div className="flex items-center justify-between mb-4">
                        <h1 className="text-xl font-semibold  text-gray-900 dark:text-gray-100">
                            {t("inferTitle")}
                        </h1>
                        <button
                            onClick={clearCanvas}
                            className="px-3 py-2 text-sm bg-gray-900 text-white  dark:bg-gray-50 dark:text-black rounded hover:bg-gray-800"
                        >
                            {t("clear")}
                        </button>
                    </div>

                    <DrawCanvas onChange={onChange} clearSignal={clearSignal} />

                    <p className="mt-3 text-sm text-gray-500">
                        {t("drawHint")}
                    </p>
                </div>

                {/* 右侧：结果 */}
                <div className="bg-white dark:bg-gray-900 rounded-xl shadow p-6">
                    <div className="flex items-center justify-between mb-4">
                        <h2 className="text-xl font-semibold  text-gray-900 dark:text-gray-100">
                            {t("inferRes")}
                        </h2>
                        <div className="text-sm text-gray-500">
                            {t(statusKey)}
                        </div>
                    </div>

                    {top5.length === 0 ? (
                        <div className="text-gray-400 text-sm">
                            {t("inferHint")}
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
                                                <div className="font-medium  text-gray-900 dark:text-gray-100">
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
