import Link from "next/link";
import { useRouter } from "next/router";

export default function TopNav() {
    const router = useRouter();

    const linkClass = (path: string) =>
        router.pathname === path ? "font-bold underline" : "hover:underline";

    return (
        <nav className="w-full border-b bg-white">
            <div className="max-w-5xl mx-auto px-6 py-4 flex items-center gap-6 text-sm">
                <Link href="/" className={linkClass("/")}>
                    主页
                </Link>
                <Link href="/submit" className={linkClass("/submit")}>
                    提交
                </Link>
                <Link href="/infer" className={linkClass("/infer")}>
                    识别
                </Link>
                <Link href="/symbols" className={linkClass("/symbols")}>
                    符号表
                </Link>
                <a
                    href="https://github.com/dongguaguaguagua/symbolizer"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="ml-auto hover:underline"
                >
                    GitHub
                </a>
            </div>
        </nav>
    );
}
