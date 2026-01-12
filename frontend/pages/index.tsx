import Head from "next/head";

export default function HomePage() {
    return (
        <>
            <Head>
                <meta
                    name="viewport"
                    content="width=device-width, initial-scale=1"
                />
            </Head>

            <main
                style={{
                    minHeight: "100vh",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                }}
            >
                <div
                    style={{
                        width: 420,
                        padding: 32,
                        borderRadius: 12,
                        boxShadow: "0 10px 30px rgba(0,0,0,0.12)",
                    }}
                >
                    <h1 style={{ fontSize: 28, marginBottom: 12 }}>
                        Symbolizer：LaTeX 字符识别
                    </h1>

                    <p style={{ marginBottom: 24, lineHeight: 1.6 }}>
                        这是一个<strong>完全离线</strong>的手写 LaTeX 符号识别
                        Demo。
                        <br />
                        推理在浏览器本地完成，模型通过 WASM 加载。
                    </p>
                    <a href="/infer">
                        <strong>进入识别 →</strong>
                    </a>
                    <br />
                    <a href="/submit">为数据集做贡献 →</a>

                    <div
                        style={{
                            marginTop: 24,
                            fontSize: 12,
                            color: "#666",
                        }}
                    >
                        • 五个候选输出，识别率高达 98%
                        <br />
                        • 无网络请求
                        <br />• 量化模型，推理速度快
                    </div>
                </div>
            </main>
        </>
    );
}
