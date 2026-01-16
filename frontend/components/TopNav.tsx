"use client";
import Link from "next/link";
import { useRouter } from "next/router";
import { useTheme } from "@/lib/useTheme";
import { useI18n } from "@/lib/i18n";
import {
    FiSun,
    FiMoon,
    FiGithub,
    FiChevronDown,
    FiMenu,
    FiX,
} from "react-icons/fi";
import { useState } from "react";

export default function TopNav() {
    const router = useRouter();
    const { theme, toggle } = useTheme();
    const { t, lang, setLang } = useI18n();

    const [mobileOpen, setMobileOpen] = useState(false);

    const linkClass = (path: string) =>
        router.pathname === path ? "font-bold underline" : "hover:underline";

    // 移动端左侧显示当前页面
    const currentPage = (() => {
        switch (router.pathname) {
            case "/":
                return t("home");
            case "/submit":
                return t("submit");
            case "/infer":
                return t("infer");
            case "/symbols":
                return t("symbols");
            default:
                return "";
        }
    })();

    return (
        <nav className="w-full border-b bg-white dark:bg-gray-900 dark:border-gray-800 relative">
            <div className="max-w-5xl mx-auto px-6 py-4 flex items-center justify-between text-sm text-gray-800 dark:text-gray-100">
                <div className="flex items-center gap-2">
                    <span className="md:hidden font-semibold">
                        {currentPage}
                    </span>
                    <div className="hidden md:flex items-center gap-6">
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
                    </div>
                </div>

                <div className="flex items-center gap-4">
                    {/* 桌面端 */}
                    <div className="hidden md:flex items-center gap-4">
                        {/* 语言切换 */}
                        <div className="relative group">
                            <button className="flex items-center gap-1 px-2 py-1 rounded cursor-pointer hover:bg-gray-200 dark:hover:bg-gray-800 transition">
                                <span>
                                    {lang === "zh" ? "中文" : "English"}
                                </span>
                                <FiChevronDown className="h-4 w-4" />
                            </button>
                            <div className="absolute right-0 mt-1 w-20 bg-white dark:bg-gray-800 border dark:border-gray-700 rounded shadow-md opacity-0 group-hover:opacity-100 transition-opacity duration-200 z-20">
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

                        {/* 暗黑模式 */}
                        <button
                            onClick={toggle}
                            className="p-2 rounded cursor-pointer hover:bg-gray-200 dark:hover:bg-gray-800 transition-transform duration-500 flex items-center justify-center relative group"
                        >
                            {theme === "dark" ? (
                                <FiMoon className="h-5 w-5" />
                            ) : (
                                <FiSun className="h-5 w-5" />
                            )}
                            {/* Tooltip */}
                            <span className="absolute top-full mb-1 left-1/2 -translate-x-1/2 whitespace-nowrap rounded bg-gray-700 px-2 py-1 text-xs text-white opacity-0 group-hover:opacity-100 transition-opacity">
                                {theme === "dark"
                                    ? t("lightToolTip")
                                    : t("darkToolTip")}
                            </span>
                        </button>

                        {/* GitHub */}
                        <a
                            href="https://github.com/zimya/symbolizer"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="p-2 rounded cursor-pointer hover:bg-gray-200 dark:hover:bg-gray-800 transition  relative group"
                            title="GitHub"
                        >
                            <FiGithub className="h-5 w-5" />
                            <span className="absolute top-full mb-1 left-1/2 -translate-x-1/2 whitespace-nowrap rounded bg-gray-700 px-2 py-1 text-xs text-white opacity-0 group-hover:opacity-100 transition-opacity">
                                {t("githubToolTip")}
                            </span>
                        </a>
                    </div>

                    {/* 移动端汉堡 */}
                    <button
                        onClick={() => setMobileOpen(!mobileOpen)}
                        className="md:hidden p-2 rounded hover:bg-gray-200 dark:hover:bg-gray-800"
                    >
                        {mobileOpen ? (
                            <FiX className="h-5 w-5" />
                        ) : (
                            <FiMenu className="h-5 w-5" />
                        )}
                    </button>
                </div>
            </div>

            {/* 移动端折叠菜单 */}
            {mobileOpen && (
                <div className="md:hidden w-full bg-white dark:bg-gray-900 border-t dark:border-gray-800 flex flex-col gap-2 px-6 py-4 text-gray-800 dark:text-gray-100 z-10">
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

                    {/* 语言切换 */}
                    <div className="flex flex-col gap-1 mt-2">
                        <button
                            onClick={() => setLang("zh")}
                            className={`text-left px-2 py-1 rounded hover:bg-gray-200 dark:hover:bg-gray-800 transition ${lang === "zh" ? "font-bold" : ""}`}
                        >
                            中文
                        </button>
                        <button
                            onClick={() => setLang("en")}
                            className={`text-left px-2 py-1 rounded hover:bg-gray-200 dark:hover:bg-gray-800 transition ${lang === "en" ? "font-bold" : ""}`}
                        >
                            English
                        </button>
                    </div>

                    <button
                        onClick={toggle}
                        className="flex items-center gap-2 px-2 py-1 rounded hover:bg-gray-200 dark:hover:bg-gray-800 mt-2"
                    >
                        {theme === "dark" ? <FiMoon /> : <FiSun />}
                        <span>{t("toggleTheme")}</span>
                    </button>

                    <a
                        href="https://github.com/zimya/symbolizer"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="flex items-center gap-2 px-2 py-1 rounded hover:bg-gray-200 dark:hover:bg-gray-800 mt-2"
                    >
                        <FiGithub /> GitHub
                    </a>
                </div>
            )}
        </nav>
    );
}
