import "@/styles/global.css";
import type { AppProps } from "next/app";
import TopNav from "@/components/TopNav";
import Head from "next/head";
import { I18nProvider } from "@/lib/i18n";
import { useI18n } from "@/lib/i18n";

export default function App({ Component, pageProps }: AppProps) {
    return (
        <I18nProvider>
            <Head>
                <title>Symbolizer：LaTeX 字符识别</title>
                <link rel="icon" href="/favicon.ico" />
            </Head>

            <TopNav />
            <Component {...pageProps} />
        </I18nProvider>
    );
}
