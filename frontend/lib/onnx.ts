// lib/onnx.ts
import { DownloadProgress } from "@/components/DownloadProgressLine";

let session: any = null;
let ort: any = null;

export async function loadSession() {
    if (typeof window === "undefined") {
        throw new Error(
            "onnxruntime-web must be loaded in browser environment",
        );
    }

    if (session) return session;

    if (!ort) {
        ort = await import("onnxruntime-web");

        if (ort?.env?.wasm) {
            ort.env.wasm.wasmPaths = "/ort/";
            ort.env.wasm.numThreads = 1;
        }
    }

    session = await ort.InferenceSession.create(
        // "/model/model_int8.onnx",
        // "/model/residualcnn_int8.onnx",
        "/model/residualcnn_augment_int8.onnx",
        {
            executionProviders: ["wasm"],
            graphOptimizationLevel: "all",
        },
    );

    return session;
}

/**
 * 页面加载：尽早触发下载/缓存 + ORT 初始化
 */
async function fetchWithProgressAndCache(
    url: string,
    cache: Cache,
    onFileProgress: (loaded: number, total: number) => void,
) {
    const res = await fetch(url, { cache: "force-cache" });
    if (!res.ok || !res.body) {
        // 如果没有 body 或状态异常，仍尝试把原始 response 放入 cache（容错）
        try {
            await cache.put(url, res.clone());
        } catch {}
        // 报告 0/total
        const totalHeader = Number(res.headers.get("Content-Length")) || 0;
        onFileProgress(totalHeader, totalHeader);
        return;
    }

    const total = Number(res.headers.get("Content-Length")) || 0;
    const reader = res.body.getReader();
    const chunks: Uint8Array[] = [];
    let received = 0;

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        if (value) {
            chunks.push(value);
            received += value.length;
            onFileProgress(received, total);
        }
    }

    // 拼接为一个 ArrayBuffer
    const buf = new Uint8Array(received);
    let offset = 0;
    for (const c of chunks) {
        buf.set(c, offset);
        offset += c.length;
    }

    // 构造 Response 并写入 cache，
    // 写明 Content-Length 可让后续读取更稳定
    const headers = new Headers();
    headers.set("Content-Length", String(buf.byteLength));
    headers.set("Content-Type", contentTypeFor(url));
    const resp = new Response(buf.buffer, { headers });
    try {
        await cache.put(url, resp);
    } catch (e) {
        console.warn("cache.put failed for", url, e);
    }
}

function contentTypeFor(url: string) {
    if (url.endsWith(".wasm")) return "application/wasm";
    if (url.endsWith(".onnx")) return "application/octet-stream";
    if (url.endsWith(".json")) return "application/json";
    return "application/octet-stream";
}

function makePhaseDownloader(urls: string[]) {
    // 返回一个函数： (onProgress) => Promise<void>
    return async (onProgress?: (p: DownloadProgress) => void) => {
        const cache = await caches.open("preload-cache-v1");

        const fileLoaded = new Map<string, number>();
        const fileTotal = new Map<string, number>();

        const start = performance.now();

        const report = () => {
            const loaded = [...fileLoaded.values()].reduce((a, b) => a + b, 0);
            const total = [...fileTotal.values()].reduce((a, b) => a + b, 0);
            const elapsed = (performance.now() - start) / 1000;
            const speed = elapsed > 0 ? loaded / elapsed : 0;

            const percent =
                total > 0
                    ? Math.min(100, (loaded / total) * 100)
                    : loaded > 0
                      ? 0
                      : 0;
            onProgress?.({
                loaded,
                total,
                percent,
                speed,
            });
        };

        // 并行下载，但每个文件回调都会覆盖 fileLoaded[file]
        await Promise.allSettled(
            urls.map(async (url) => {
                // 先确保 fileTotal 有项（避免分母缺失）
                fileLoaded.set(url, 0);
                fileTotal.set(url, 0);

                try {
                    await fetchWithProgressAndCache(url, cache, (l, t) => {
                        fileLoaded.set(url, l);
                        // 仅在第一次拿到 non-zero total 时设置（防止覆盖）
                        if (!fileTotal.get(url) && t) {
                            fileTotal.set(url, t);
                        } else if (t) {
                            // 如果每次都有 t，也确保最新的 total 被记录（容错）
                            fileTotal.set(url, t);
                        }
                        report();
                    });
                } catch (e) {
                    // 忽略单文件错误，但保留 zeros
                    console.warn("fetchWithProgressAndCache failed", url, e);
                }
            }),
        );

        // 最终一次报告，保证 percent 会到 100（若 total>0）
        report();
    };
}

export async function preloadRuntimeAndModel(
    onOnnxProgress?: (p: DownloadProgress) => void,
    onWasmProgress?: (p: DownloadProgress) => void,
) {
    if (typeof window === "undefined") return;

    const onnxUrls = [
        "/model/model_int8.onnx",
        // "/model/residualcnn_int8.onnx",
        // "/model/residualcnn_augment_int8.onnx",
    ];

    const wasmUrls = [
        // "/ort/ort-wasm.wasm",
        "/ort/ort-wasm-simd.wasm",
        // "/ort/ort-wasm-threaded.wasm",
        // "/ort/ort-wasm-simd-threaded.wasm",
    ];

    const downloadOnnxPhase = makePhaseDownloader(onnxUrls);
    await downloadOnnxPhase(onOnnxProgress);
    const downloadWasmPhase = makePhaseDownloader(wasmUrls);
    await downloadWasmPhase(onWasmProgress);

    try {
        await loadSession();
    } catch (e) {
        console.warn("loadSession failed", e);
    }
}

/** 数值稳定 softmax */
function softmax(xs: number[]) {
    let max = -Infinity;
    for (const x of xs) max = Math.max(max, x);

    const exps = new Array<number>(xs.length);
    let sum = 0;
    for (let i = 0; i < xs.length; i++) {
        const v = Math.exp(xs[i] - max);
        exps[i] = v;
        sum += v;
    }

    if (!isFinite(sum) || sum <= 0) {
        const uni = 1 / xs.length;
        return xs.map(() => uni);
    }

    for (let i = 0; i < exps.length; i++) exps[i] /= sum;
    return exps;
}

export async function inferTop5(gray: Uint8Array) {
    const sess = await loadSession();

    // [1,3,32,32]
    const input = new Float32Array(3 * 32 * 32);
    for (let i = 0; i < 1024; i++) {
        const v = gray[i] / 255.0;
        input[i] = v;
        input[i + 1024] = v;
        input[i + 2048] = v;
    }

    const inputName =
        (sess?.inputNames && sess.inputNames[0]) ||
        (sess?.inputNames?.length ? sess.inputNames[0] : "input");

    const feeds: Record<string, any> = {
        [inputName]: new ort.Tensor("float32", input, [1, 3, 32, 32]),
    };

    const output = await sess.run(feeds);

    const outName =
        (sess?.outputNames && sess.outputNames[0]) || Object.keys(output)[0];

    const logitsTensor = output[outName];
    const logits = Array.from(logitsTensor.data as Float32Array);

    const probs = softmax(logits);
    const pairs = probs.map((v, i) => ({ i, v }));
    pairs.sort((a, b) => b.v - a.v);
    return pairs.slice(0, 5);
}
