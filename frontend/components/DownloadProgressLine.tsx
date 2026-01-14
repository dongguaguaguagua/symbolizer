import { useI18n } from "@/lib/i18n";

export type DownloadProgress = {
    loaded: number; // bytes
    total: number; // bytes, 0 = unknown
    speed: number; // bytes / second
    percent: number; // 0~100，仅在 total>0 时有意义
};

type Props = {
    label: string;
    progress?: DownloadProgress | null;
};

export default function DownloadProgressLine({ label, progress }: Props) {
    const { t } = useI18n();
    if (!progress) {
        return (
            <div className="tabular-nums">
                {label}: {t("waiting")}
            </div>
        );
    }

    const speedMB = progress.speed / 1024 / 1024;

    // total 已知 → 百分比模式
    if (progress.total > 0) {
        return (
            <div className="tabular-nums">
                {label}：{progress.percent.toFixed(1)}% · {speedMB.toFixed(2)}{" "}
                MB/s
            </div>
        );
    }

    // total 未知 → 显示已下载体积
    const loadedMB = progress.loaded / 1024 / 1024;

    return (
        <div className="tabular-nums">
            {label}：{loadedMB.toFixed(2)} MB · {speedMB.toFixed(2)} MB/s
        </div>
    );
}
