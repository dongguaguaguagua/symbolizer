// lib/onnx.ts
import { DownloadProgress } from "@/components/DownloadProgressLine";

const MODEL_URL = "/model/residualcnn_augment_int8.onnx";
const WASM_URLS = ["/ort/ort-wasm-simd.wasm"];

let session: any = null;
let ort: any = null;
let modelBuffer: ArrayBuffer | null = null;

function contentTypeFor(url: string) {
    if (url.endsWith(".wasm")) return "application/wasm";
    if (url.endsWith(".onnx")) return "application/octet-stream";
    if (url.endsWith(".json")) return "application/json";
    return "application/octet-stream";
}

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

    if (!modelBuffer) {
        throw new Error("Model buffer not preloaded");
    }

    session = await ort.InferenceSession.create(modelBuffer, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
    });

    return session;
}
async function fetchWithProgressAndCache(
    url: string,
    cache: Cache,
    onFileProgress: (loaded: number, total: number) => void,
): Promise<ArrayBuffer | null> {
    const res = await fetch(url, { cache: "force-cache" });

    if (!res.ok || !res.body) {
        return null;
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

    const buf = new Uint8Array(received);
    let offset = 0;
    for (const c of chunks) {
        buf.set(c, offset);
        offset += c.length;
    }

    const headers = new Headers();
    headers.set("Content-Length", String(buf.byteLength));
    headers.set("Content-Type", contentTypeFor(url));

    const resp = new Response(buf.buffer, { headers });
    await cache.put(url, resp.clone());

    return buf.buffer;
}

function makePhaseDownloader(urls: string[]) {
    return async (
        onProgress?: (p: DownloadProgress) => void,
    ): Promise<Map<string, ArrayBuffer>> => {
        const cache = await caches.open("preload-cache-v1");

        const buffers = new Map<string, ArrayBuffer>();
        const fileLoaded = new Map<string, number>();
        const fileTotal = new Map<string, number>();
        const start = performance.now();

        const report = () => {
            const loaded = [...fileLoaded.values()].reduce((a, b) => a + b, 0);
            const total = [...fileTotal.values()].reduce((a, b) => a + b, 0);
            const elapsed = (performance.now() - start) / 1000;
            const speed = elapsed > 0 ? loaded / elapsed : 0;

            onProgress?.({
                loaded,
                total,
                percent: total ? Math.min(100, (loaded / total) * 100) : 0,
                speed,
            });
        };

        await Promise.all(
            urls.map(async (url) => {
                fileLoaded.set(url, 0);
                fileTotal.set(url, 0);

                const buf = await fetchWithProgressAndCache(
                    url,
                    cache,
                    (l, t) => {
                        fileLoaded.set(url, l);
                        if (t) fileTotal.set(url, t);
                        report();
                    },
                );

                if (buf) buffers.set(url, buf);
            }),
        );

        report();
        return buffers;
    };
}

export async function preloadRuntimeAndModel(
    onOnnxProgress?: (p: DownloadProgress) => void,
    onWasmProgress?: (p: DownloadProgress) => void,
) {
    if (typeof window === "undefined") return;

    const downloadOnnx = makePhaseDownloader([MODEL_URL]);
    const onnxBuffers = await downloadOnnx(onOnnxProgress);

    modelBuffer = onnxBuffers.get(MODEL_URL) ?? null;
    if (!modelBuffer) {
        throw new Error("Failed to preload ONNX model");
    }

    const downloadWasm = makePhaseDownloader(WASM_URLS);
    await downloadWasm(onWasmProgress);

    await loadSession();
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
