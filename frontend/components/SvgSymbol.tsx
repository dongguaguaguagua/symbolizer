"use client";

type Props = {
    base64: string;
    size?: string;
};

export default function SvgSymbol({ base64, size = "2.25em" }: Props) {
    return (
        <img
            src={`data:image/svg+xml;base64,${base64}`}
            alt="math-symbol"
            style={{
                width: size,
                height: size,
                verticalAlign: "middle",
            }}
        />
    );
}
