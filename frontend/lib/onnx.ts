// lib/onnx.ts
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
        "/model/residualcnn_augument_int8.onnx",
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
export async function preloadRuntimeAndModel() {
    if (typeof window === "undefined") return;

    const urls = [
        "/model/model_int8.onnx",
        "/model/residualcnn_int8.onnx",
        "/model/residualcnn_augument_int8.onnx",
        "/ort/ort-wasm.wasm",
        "/ort/ort-wasm-simd.wasm",
        "/ort/ort-wasm-threaded.wasm",
        "/ort/ort-wasm-simd-threaded.wasm",
    ];

    await Promise.allSettled(
        urls.map((u) =>
            fetch(u, { cache: "force-cache" }).catch(() => undefined),
        ),
    );

    try {
        await loadSession();
    } catch {}
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
