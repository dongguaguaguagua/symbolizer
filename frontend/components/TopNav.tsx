"use client";
import Link from "next/link";
import { useRouter } from "next/router";
import { useTheme } from "@/lib/useTheme";
import { useI18n } from "@/lib/i18n";
import { FiSun, FiMoon, FiGithub, FiChevronDown } from "react-icons/fi";
import { useState } from "react";

export default function TopNav() {
    const router = useRouter();
    const { theme, toggle } = useTheme();
    const { t, lang, setLang } = useI18n();

    const linkClass = (path: string) =>
        router.pathname === path ? "font-bold underline" : "hover:underline";

    return (
        <nav className="w-full border-b bg-white dark:bg-gray-900 dark:border-gray-800">
            <div className="max-w-5xl mx-auto px-6 py-4 flex items-center gap-6 text-sm text-gray-800 dark:text-gray-100">
                <Link href="/" className={linkClass("/")}>
                    {t("home")}
                </Link>
                <Link href="/submit" className={linkClass("/submit")}>
                    {t("submit")}
                </Link>
                <Link href="/infer" className={linkClass("/infer")}>
                    {t("infer")}
                </Link>
                <Link href="/symbols" className={linkClass("/symbols")}>
                    {t("symbols")}
                </Link>

                <div className="ml-auto flex items-center gap-4 relative">
                    {/* 语言切换 Hover 下拉 */}
                    <div className="relative group">
                        <button className="flex items-center gap-1 px-2 py-1 rounded cursor-pointer hover:bg-gray-200 dark:hover:bg-gray-800 transition">
                            <span>{lang === "zh" ? "中文" : "English"}</span>
                            <FiChevronDown className="h-4 w-4" />
                        </button>
                        <div className="absolute right-0 mt-1 w-20 bg-white dark:bg-gray-800 border dark:border-gray-700 rounded shadow-md opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-auto z-20">
                            <button
                                onClick={() => setLang("zh")}
                                className="block w-full text-left px-2 py-1 hover:bg-gray-200 dark:hover:bg-gray-700 transition"
                            >
                                中文
                            </button>
                            <button
                                onClick={() => setLang("en")}
                                className="block w-full text-left px-2 py-1 hover:bg-gray-200 dark:hover:bg-gray-700 transition"
                            >
                                English
                            </button>
                        </div>
                    </div>

                    {/* 暗黑模式切换 */}
                    <button
                        onClick={toggle}
                        className="p-2 rounded cursor-pointer hover:bg-gray-200 dark:hover:bg-gray-800 transition-transform duration-500 flex items-center justify-center"
                        title="Toggle theme"
                    >
                        <span
                            className={`transform transition-transform duration-500 ${
                                theme === "dark" ? "rotate-0" : "rotate-180"
                            }`}
                        >
                            {theme === "dark" ? (
                                <FiMoon className="h-5 w-5" />
                            ) : (
                                <FiSun className="h-5 w-5" />
                            )}
                        </span>
                    </button>

                    {/* GitHub */}
                    <a
                        href="https://github.com/dongguaguaguagua/symbolizer"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="p-2 rounded cursor-pointer hover:bg-gray-200 dark:hover:bg-gray-800 transition"
                        title="GitHub"
                    >
                        <FiGithub className="h-5 w-5" />
                    </a>
                </div>
            </div>
        </nav>
    );
}
