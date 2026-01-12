import "@/styles/global.css";
import type { AppProps } from "next/app";
import TopNav from "@/components/TopNav";
import Head from "next/head";

export default function App({ Component, pageProps }: AppProps) {
    return (
        <>
            <Head>
                <title>Symbolizer：LaTeX 字符识别</title>
                <link rel="icon" href="/favicon.ico" />
                <link rel="apple-touch-icon" href="/icon-192.png" />
            </Head>

            <TopNav />
            <Component {...pageProps} />
        </>
    );
}
