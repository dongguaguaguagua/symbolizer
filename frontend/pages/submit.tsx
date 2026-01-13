import { useState, useEffect, useCallback } from "react";
import DrawCanvas from "@/components/DrawCanvas";
import Preview32 from "@/components/Preview32";
import CopyableMono from "@/components/CopyableMono";
import SvgSymbol from "@/components/SvgSymbol";
import { useI18n } from "@/lib/i18n";
import { FiRefreshCw } from "react-icons/fi";

const WORKER_BASE = "https://symbol-collector.melonhu.cn";

type SymbolInfo = {
    label: string;
    symbol: string;
    unicode: string;
    svg: string;
};

function unicodeLiteralToChar(unicode: string) {
    const hex = unicode.match(/\{([0-9A-Fa-f]+)\}/)?.[1];
    return hex ? String.fromCodePoint(parseInt(hex, 16)) : "?";
}

function grayToBase64(gray: Uint8Array) {
    return btoa(String.fromCharCode(...gray));
}

export default function SubmitPage() {
    const { t } = useI18n();
    const [target, setTarget] = useState<SymbolInfo | null>(null);
    const [gray, setGray] = useState<Uint8Array | null>(null);
    const [clearSignal, setClearSignal] = useState(0);
    const [statusKey, setStatusKey] = useState<StatusKey>("ready");
    type StatusKey = "ready" | "uploading" | "submitSuccess";

    const fetchRandom = async () => {
        const res = await fetch(`${WORKER_BASE}/random-symbol`);
        const data = await res.json();
        setTarget(data);
    };

    useEffect(() => {
        fetchRandom();
    }, []);

    const onChange = useCallback((g: Uint8Array) => {
        setGray(g);
    }, []);

    const clear = () => {
        setGray(null);
        setClearSignal((x) => x + 1);
    };

    const submit = async () => {
        if (!gray || !target) return;

        setStatusKey("uploading");

        await fetch(`${WORKER_BASE}/submit`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                label: target.label,
                image: grayToBase64(gray),
            }),
        });

        setStatusKey("submitSuccess");
        clear();
        fetchRandom();
    };

    return (
        <div className="min-h-screen bg-gray-50 dark:bg-gray-950 p-8">
            <div className="max-w-5xl mx-auto grid grid-cols-1 md:grid-cols-2 gap-6 md:gap-10">
                {/* 左侧：画板 */}
                <div className="bg-white dark:bg-gray-900 rounded-xl shadow p-6">
                    <div className="flex items-center justify-between mb-4">
                        <h1 className="text-xl font-semibold text-gray-900 dark:text-gray-100">
                            {t("submitTitle")}
                        </h1>
                        <div className="text-sm text-gray-500">
                            {t(statusKey)}
                        </div>
                    </div>

                    <DrawCanvas onChange={onChange} clearSignal={clearSignal} />

                    <p className="mt-3 text-sm text-gray-500">
                        {t("submitThanks")}
                    </p>

                    <div className="mt-4 flex gap-3">
                        <button
                            onClick={clear}
                            className="px-3 py-2 text-sm bg-gray-900 text-white dark:bg-gray-50 dark:text-black rounded hover:bg-gray-800"
                        >
                            {t("clear")}
                        </button>

                        <button
                            onClick={submit}
                            disabled={!gray}
                            className="px-3 py-2 text-sm bg-gray-900 text-white dark:bg-gray-50 dark:text-black rounded hover:bg-gray-800 disabled:opacity-50"
                        >
                            {t("submitSample")}
                        </button>
                    </div>
                </div>

                {/* 右侧：目标符号 */}
                <div className="bg-white dark:bg-gray-900 rounded-xl shadow p-6">
                    <div className="flex items-center justify-between mb-4">
                        <h2 className="text-xl font-semibold  text-gray-900 dark:text-gray-100">
                            {t("submitTarget")}
                        </h2>

                        {/* 刷新按钮 */}
                        <button
                            onClick={fetchRandom}
                            className="p-1 rounded hover:bg-gray-200 dark:hover:bg-gray-800 relative group"
                        >
                            <FiRefreshCw className="h-5 w-5 text-gray-600 dark:text-gray-300" />
                            {/* Tooltip */}
                            <span className="absolute bottom-full mb-1 left-1/2 -translate-x-1/2 whitespace-nowrap rounded bg-gray-700 px-2 py-1 text-xs text-white opacity-0 group-hover:opacity-100 transition-opacity">
                                {t("refresh")}
                            </span>
                        </button>
                    </div>

                    {target && (
                        <div className="border rounded-lg px-4 py-3">
                            <div className="flex items-center justify-between gap-3 flex-wrap sm:flex-nowrap">
                                {/* 左侧：文本 */}
                                <div className="min-w-0">
                                    <div className="font-medium  text-gray-900 dark:text-gray-100">
                                        #{target.label}
                                    </div>

                                    <div className="mt-1 space-y-2">
                                        <CopyableMono
                                            text={target.symbol}
                                            tone="blue"
                                        />
                                        <CopyableMono
                                            text={target.unicode}
                                            tone="green"
                                        />
                                    </div>
                                </div>

                                <div className="text-4xl">
                                    {target.svg ? (
                                        <SvgSymbol
                                            base64={target.svg}
                                            size="2.25em"
                                        />
                                    ) : (
                                        unicodeLiteralToChar(target.unicode)
                                    )}
                                </div>

                                {/* 右侧：32x32 预览 */}
                                <Preview32 gray={gray} />
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
