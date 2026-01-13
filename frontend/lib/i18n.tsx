"use client";
import { createContext, useContext, useEffect, useState } from "react";

export type Lang = "zh" | "en";

const dict = {
    zh: {
        websiteTitle: "Symbolizer：LaTeX 字符识别",

        // nav
        home: "主页",
        submit: "提交",
        infer: "识别",
        symbols: "符号表",

        // common
        clear: "清空画板",
        ready: "就绪",
        copy: "点击复制",
        copySuccess: "复制成功",
        refresh: "刷新",

        // infer
        inferTitle: "画出您想识别的符号",
        loading: "正在加载模型与运行时（首次会稍慢）…",
        inferHint: "请在左侧写一个符号",
        infering: "正在推理…",
        drawHint: "松开鼠标识别",
        inferRes: "预测结果",

        // submit
        submitTitle: "提交手写样本",
        submitThanks: "感谢您对数据集的贡献",
        submitTarget: "目标符号",
        submitSample: "提交样本",
        uploading: "上传中...",
        submitSuccess: "提交成功！",

        // index
        homeTitle: "Symbolizer: LaTeX 字符识别",
        homeDescLine1: "这是一个完全离线的手写 LaTeX 符号识别 Demo。",
        homeDescLine2: "推理在浏览器本地完成，模型通过 WASM 加载，无需网络。",
        homeEnterInfer: "进入识别 →",
        homeContribute: "为数据集做贡献 →",
        homeFooter: "五个候选输出，识别率高达 98%",

        // symbol list
        symbolListTitle: "符号表",
        search: "搜索 LaTeX 或 Unicode...",
        searchFailed: "没有匹配的符号",
    },
    en: {
        websiteTitle: "Symbolizer: LaTeX Symbol Recognition",
        // nav
        home: "Home",
        submit: "Submit",
        infer: "Recognize",
        symbols: "Symbols",

        // common
        clear: "Clear Canvas",
        ready: "Ready",
        copy: "Copy",
        copySuccess: "Copied!",
        refresh: "Refresh",

        // infer
        inferTitle: "Draw to recognize",
        loading: "Loading model and runtime (first time may take longer)…",
        inferHint: "Please draw a symbol on the left",
        infering: "Running inference…",
        drawHint: "Release the mouse to recognize",
        inferRes: "Prediction Results",

        // submit
        submitTitle: "Submit Handwritten Sample",
        submitThanks: "Thank you for contributing to the dataset",
        submitTarget: "Target Symbol",
        submitSample: "Submit Sample",
        uploading: "Uploading...",
        submitSuccess: "Submit success!",

        // index
        homeTitle: "Symbolizer: LaTeX Symbol Recognition",
        homeDescLine1:
            "A fully offline handwritten LaTeX symbol recognition demo.",
        homeDescLine2:
            "Inference runs locally in the browser, with the model loaded via WASM and no network required.",
        homeEnterInfer: "Start Recognition →",
        homeContribute: "Contribute to the Dataset →",
        homeFooter: "Five candidate outputs with accuracy up to 98%",

        // symbol list
        symbolListTitle: "Symbol Table",
        search: "Search LaTeX or Unicode...",
        searchFailed: "No matching symbol",
    },
};

type DictKey = keyof (typeof dict)["zh"];

const I18nContext = createContext<{
    lang: Lang;
    setLang: (l: Lang) => void;
    t: (k: DictKey) => string;
} | null>(null);

export function I18nProvider({ children }: { children: React.ReactNode }) {
    const [lang, setLang] = useState<Lang>("zh");

    useEffect(() => {
        const saved = localStorage.getItem("lang") as Lang | null;
        if (saved === "zh" || saved === "en") {
            setLang(saved);
        }
    }, []);

    const changeLang = (l: Lang) => {
        setLang(l);
        localStorage.setItem("lang", l);
    };

    const t = (k: DictKey) => dict[lang][k];

    return (
        <I18nContext.Provider value={{ lang, setLang: changeLang, t }}>
            {children}
        </I18nContext.Provider>
    );
}

export function useI18n() {
    const ctx = useContext(I18nContext);
    if (!ctx) {
        throw new Error("useI18n must be used inside I18nProvider");
    }
    return ctx;
}
