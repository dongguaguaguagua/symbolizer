"use client";

type Props = {
    base64: string;
    size?: string;
    className?: string;
};

export default function SvgSymbol({
    base64,
    size = "2.25em",
    className = "",
}: Props) {
    return (
        <img
            src={`data:image/svg+xml;base64,${base64}`}
            alt="math-symbol"
            style={{
                width: size,
                height: size,
                verticalAlign: "middle",
            }}
            className={[
                // 暗黑模式下反色
                "dark:invert",
                "dark:brightness-90",
                className,
            ].join(" ")}
        />
    );
}
