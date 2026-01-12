"use client";
import { useEffect, useRef } from "react";

export default function Preview32({ gray }: { gray: Uint8Array | null }) {
    const ref = useRef<HTMLCanvasElement>(null);

    useEffect(() => {
        const c = ref.current!;
        const ctx = c.getContext("2d")!;
        ctx.fillStyle = "#fff";
        ctx.fillRect(0, 0, 32, 32);

        if (!gray) return;

        const img = ctx.createImageData(32, 32);
        for (let i = 0; i < 1024; i++) {
            const v = gray[i];
            img.data[i * 4] = v;
            img.data[i * 4 + 1] = v;
            img.data[i * 4 + 2] = v;
            img.data[i * 4 + 3] = 255;
        }
        ctx.putImageData(img, 0, 0);
    }, [gray]);

    return (
        <canvas
            ref={ref}
            width={32}
            height={32}
            className="border bg-white w-24 h-24 sm:w-32 sm:h-32 image-rendering-pixelated"
        />
    );
}
