import { useEffect, useMemo, useState } from "react";
import SvgSymbol from "@/components/SvgSymbol";
import CopyableMono from "@/components/CopyableMono";
import { useI18n } from "@/lib/i18n";

type Mapping = {
    symbol: string;
    unicode: string;
    svg: string;
};

export default function SymbolListPage() {
    const [mappings, setMappings] = useState<Record<string, Mapping>>({});
    const [query, setQuery] = useState("");
    const { t } = useI18n();
    useEffect(() => {
        fetch("/mappings.json", { cache: "force-cache" })
            .then((r) => r.json())
            .then(setMappings)
            .catch((e) => console.error("load mappings failed", e));
    }, []);

    const items = useMemo(() => {
        const q = query.trim().toLowerCase();
        const arr = Object.entries(mappings).map(([id, m]) => ({
            id,
            ...m,
        }));

        if (!q) return arr;

        return arr.filter(
            (m) =>
                m.symbol.toLowerCase().includes(q) ||
                m.unicode.toLowerCase().includes(q),
        );
    }, [mappings, query]);

    return (
        <div className="min-h-screen bg-gray-50 dark:bg-gray-950 p-8">
            <div className="max-w-7xl mx-auto">
                {/* 标题 + 搜索框 */}
                <div className="mb-6 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
                    <h1 className="text-2xl font-semibold  text-gray-900 dark:text-gray-100">
                        {t("symbolListTitle")}
                    </h1>

                    <input
                        type="text"
                        placeholder={t("search")}
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        className="w-full sm:w-80 px-3 py-2 border rounded-md text-sm focus:outline-none focus:ring
                        bg-gray-50 dark:bg-gray-950"
                    />
                </div>

                {/* 符号网格 */}
                <div
                    className="
                        grid
                        grid-cols-2
                        sm:grid-cols-3
                        md:grid-cols-4
                        lg:grid-cols-6
                        xl:grid-cols-8
                        gap-4
                    "
                >
                    {items.map((m) => (
                        <div
                            key={m.id}
                            className="border rounded-lg px-3 py-3 bg-white dark:bg-gray-900 shadow-sm"
                        >
                            <div className="flex items-center justify-between gap-3">
                                {/* 左侧：文本 */}
                                <div className="min-w-0">
                                    <div className="font-medium text-sm  text-gray-900 dark:text-gray-100">
                                        #{m.id}
                                    </div>

                                    <div className="mt-1 space-y-1">
                                        <CopyableMono
                                            text={m.symbol}
                                            tone="blue"
                                        />
                                        <CopyableMono
                                            text={m.unicode}
                                            tone="green"
                                        />
                                    </div>
                                </div>

                                {/* 中间：SVG 符号 */}
                                <div className="text-3xl">
                                    {m.svg ? (
                                        <SvgSymbol
                                            base64={m.svg}
                                            size="1.875em"
                                        />
                                    ) : (
                                        "?"
                                    )}
                                </div>
                            </div>
                        </div>
                    ))}
                </div>

                {/* 空状态 */}
                {items.length === 0 && (
                    <div className="text-gray-400 text-sm mt-12 text-center">
                        {t("searchFailed")}
                    </div>
                )}
            </div>
        </div>
    );
}
