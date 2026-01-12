import mappings from "./mappings.json";

const corsHeaders = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
};

export default {
    async fetch(req: Request, env: Env) {
        const url = new URL(req.url);

        // 1. 处理 CORS 预检请求
        if (req.method === "OPTIONS") {
            return new Response(null, {
                status: 204,
                headers: corsHeaders,
            });
        }

        // 2. 随机符号
        if (req.method === "GET" && url.pathname === "/random-symbol") {
            const keys = Object.keys(mappings);
            const label = keys[Math.floor(Math.random() * keys.length)];

            return new Response(
                JSON.stringify({
                    label,
                    ...mappings[label],
                }),
                { headers: corsHeaders },
            );
        }

        // 3. 提交样本
        if (req.method === "POST" && url.pathname === "/submit") {
            const { label, image } = await req.json();
            const buffer = Uint8Array.from(atob(image), (c) => c.charCodeAt(0));

            await env.DB.prepare(
                "INSERT INTO samples (label, image, created_at) VALUES (?, ?, ?)",
            )
                .bind(label, buffer, Date.now())
                .run();

            return new Response(JSON.stringify({ ok: true }), {
                headers: corsHeaders,
            });
        }

        return new Response("Not found", {
            status: 404,
            headers: corsHeaders,
        });
    },
};
