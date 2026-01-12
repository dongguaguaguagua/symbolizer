/** @type {import('next').NextConfig} */
// const nextConfig = {
//     webpack(config, { isServer }) {
//         if (!isServer) {
//             config.resolve.alias["onnxruntime-node"] = false;
//         }
//         return config;
//     },
// };

// module.exports = nextConfig;
const nextConfig = {
    output: "export",
    images: { unoptimized: true },
    webpack: (config) => {
        // 支持 node_modules 中的 .mjs 文件不被当作普通 script
        config.module.rules.push({
            test: /\.mjs$/,
            include: /node_modules/,
            type: "javascript/auto",
        });

        return config;
    },
};

module.exports = nextConfig;
