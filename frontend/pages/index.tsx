import Head from "next/head";
import Link from "next/link";
import { useI18n } from "@/lib/i18n";

export default function HomePage() {
    const { t } = useI18n();

    return (
        <>
            <Head>
                <title>{t("homeTitle")}</title>
                <meta
                    name="viewport"
                    content="width=device-width, initial-scale=1"
                />
            </Head>

            <main className="min-h-screen bg-gray-50 dark:bg-gray-950 flex items-center justify-center">
                <div className="max-w-2xl w-full bg-white dark:bg-gray-900 rounded-xl shadow-lg p-8 [margin-top:-200px]">
                    <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-6">
                        <div>
                            <h1 className="text-2xl sm:text-3xl font-semibold text-gray-900 dark:text-gray-100">
                                {t("homeTitle")}
                            </h1>

                            <p className="mt-4 text-sm text-gray-700 dark:text-gray-300 leading-relaxed">
                                {t("homeDescLine1")}
                                <br />
                                {t("homeDescLine2")}
                            </p>

                            <div className="mt-6 flex flex-wrap gap-3">
                                <Link
                                    href="/infer"
                                    className="inline-flex items-center px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-md text-sm font-medium shadow-sm"
                                >
                                    {t("homeEnterInfer")}
                                </Link>

                                <Link
                                    href="/submit"
                                    className="inline-flex items-center px-4 py-2 bg-gray-100 hover:bg-gray-200 dark:bg-gray-800 dark:hover:bg-gray-700 text-gray-900 dark:text-gray-100 rounded-md text-sm font-medium"
                                >
                                    {t("homeContribute")}
                                </Link>
                            </div>
                        </div>
                    </div>

                    <div className="mt-8 text-xs text-gray-500 dark:text-gray-400">
                        {t("homeFooter")}
                    </div>
                </div>
            </main>
        </>
    );
}
