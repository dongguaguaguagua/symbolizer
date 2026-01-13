"use client";
import { useRef, useEffect } from "react";

export default function DrawCanvas({
    onChange,
    clearSignal,
}: {
    onChange: (gray: Uint8Array) => void;
    clearSignal: number;
}) {
    const ref = useRef<HTMLCanvasElement>(null);

    useEffect(() => {
        const c = ref.current!;
        const ctx = c.getContext("2d")!;

        // **始终用模型期望的颜色绘制**
        const bg = "#fff";
        const fg = "#000";

        const clear = () => {
            ctx.fillStyle = bg;
            ctx.fillRect(0, 0, 256, 256);
        };

        clear();

        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        ctx.lineWidth = 12;
        ctx.strokeStyle = fg;

        let drawing = false;
        let lx = 0,
            ly = 0;
        let pending = false;

        const drawLine = (x: number, y: number) => {
            ctx.beginPath();
            ctx.moveTo(lx, ly);
            ctx.lineTo(x, y);
            ctx.stroke();
            lx = x;
            ly = y;
        };

        const onMouseDown = (e: MouseEvent) => {
            drawing = true;
            lx = e.offsetX;
            ly = e.offsetY;
        };

        const onMouseUp = () => {
            if (!drawing) return;
            drawing = false;
            onChange(downsampleAutoCrop(c));
        };

        const onMouseLeave = () => {
            if (!drawing) return;
            drawing = false;
            onChange(downsampleAutoCrop(c));
        };

        const onMouseMove = (e: MouseEvent) => {
            if (!drawing) return;
            if (pending) return;
            pending = true;

            requestAnimationFrame(() => {
                pending = false;
                drawLine(e.offsetX, e.offsetY);
            });
        };

        const getTouchPos = (e: TouchEvent) => {
            const rect = c.getBoundingClientRect();
            const t = e.touches[0];
            return {
                x: t.clientX - rect.left,
                y: t.clientY - rect.top,
            };
        };
        const onTouchStart = (e: TouchEvent) => {
            e.preventDefault();
            drawing = true;
            const { x, y } = getTouchPos(e);
            lx = x;
            ly = y;
        };

        const onTouchMove = (e: TouchEvent) => {
            e.preventDefault();
            if (!drawing || pending) return;
            pending = true;

            requestAnimationFrame(() => {
                pending = false;
                const { x, y } = getTouchPos(e);
                drawLine(x, y);
            });
        };

        const onTouchEnd = () => {
            if (!drawing) return;
            drawing = false;
            onChange(downsampleAutoCrop(c));
        };

        c.addEventListener("mousedown", onMouseDown);
        c.addEventListener("mouseup", onMouseUp);
        c.addEventListener("mouseleave", onMouseLeave);
        c.addEventListener("mousemove", onMouseMove);
        // mobile手机触摸屏适配
        c.addEventListener("touchstart", onTouchStart, { passive: false });
        c.addEventListener("touchmove", onTouchMove, { passive: false });
        c.addEventListener("touchend", onTouchEnd);
        return () => {
            c.removeEventListener("mousedown", onMouseDown);
            c.removeEventListener("mouseup", onMouseUp);
            c.removeEventListener("mouseleave", onMouseLeave);
            c.removeEventListener("mousemove", onMouseMove);
            // mobile手机触摸屏适配
            c.removeEventListener("touchstart", onTouchStart);
            c.removeEventListener("touchmove", onTouchMove);
            c.removeEventListener("touchend", onTouchEnd);
        };
    }, [onChange]);

    useEffect(() => {
        const c = ref.current!;
        const ctx = c.getContext("2d")!;
        ctx.fillStyle = "#fff";
        ctx.fillRect(0, 0, 256, 256);
    }, [clearSignal]);

    return (
        <canvas
            ref={ref}
            width={256}
            height={256}
            className="
            border rounded
            bg-white
            dark:invert dark:brightness-90
            border-gray-300 dark:border-gray-700
            w-full max-w-[256px] aspect-square
        "
        />
    );
}

/**
 * 自动裁剪最小正方形并缩放到 32x32
 */
function downsampleAutoCrop(src: HTMLCanvasElement) {
    const W = 256;
    const H = 256;

    const ctx = src.getContext("2d")!;
    const img = ctx.getImageData(0, 0, W, H);
    const data = img.data;

    let minX = W,
        minY = H,
        maxX = 0,
        maxY = 0;
    let hasInk = false;

    // 找黑色像素的 bounding box
    for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
            const i = (y * W + x) * 4;
            const r = data[i];
            const g = data[i + 1];
            const b = data[i + 2];

            // 非白色像素
            if (r < 250 || g < 250 || b < 250) {
                hasInk = true;
                minX = Math.min(minX, x);
                minY = Math.min(minY, y);
                maxX = Math.max(maxX, x);
                maxY = Math.max(maxY, y);
            }
        }
    }

    // 如果用户没画任何东西，返回空白
    if (!hasInk) {
        return new Uint8Array(1024);
    }

    const boxW = maxX - minX + 1;
    const boxH = maxY - minY + 1;
    const size = Math.max(boxW, boxH); // 正方形边长

    // 计算正方形裁剪区域（居中）
    const cx = Math.floor((minX + maxX) / 2);
    const cy = Math.floor((minY + maxY) / 2);

    let sx = cx - Math.floor(size / 2);
    let sy = cy - Math.floor(size / 2);

    // 防止越界
    sx = Math.max(0, Math.min(W - size, sx));
    sy = Math.max(0, Math.min(H - size, sy));

    // 裁剪到临时 canvas
    const crop = document.createElement("canvas");
    crop.width = size;
    crop.height = size;
    const cropCtx = crop.getContext("2d")!;
    cropCtx.fillStyle = "#fff";
    cropCtx.fillRect(0, 0, size, size);
    cropCtx.drawImage(src, sx, sy, size, size, 0, 0, size, size);

    // 缩放到 32x32
    const small = document.createElement("canvas");
    small.width = 32;
    small.height = 32;
    const smallCtx = small.getContext("2d")!;
    smallCtx.drawImage(crop, 0, 0, 32, 32);

    const smallData = smallCtx.getImageData(0, 0, 32, 32).data;
    const out = new Uint8Array(1024);

    for (let i = 0; i < 1024; i++) {
        out[i] =
            (smallData[i * 4] + smallData[i * 4 + 1] + smallData[i * 4 + 2]) /
            3;
    }

    return out;
}
