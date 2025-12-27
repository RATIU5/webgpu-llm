import { r as renderers } from './chunks/_@astro-renderers_BCHh4Q3R.mjs';
import { c as createExports, s as serverEntrypointModule } from './chunks/_@astrojs-ssr-adapter_Cp2PBUTe.mjs';
import { manifest } from './manifest_DyVf0Dfx.mjs';

const serverIslandMap = new Map();;

const _page0 = () => import('./pages/_image.astro.mjs');
const _page1 = () => import('./pages/_llms-txt/_slug_.txt.astro.mjs');
const _page2 = () => import('./pages/404.astro.mjs');
const _page3 = () => import('./pages/api/search.astro.mjs');
const _page4 = () => import('./pages/llms-full.txt.astro.mjs');
const _page5 = () => import('./pages/llms-small.txt.astro.mjs');
const _page6 = () => import('./pages/llms.txt.astro.mjs');
const _page7 = () => import('./pages/_---slug_.astro.mjs');
const pageMap = new Map([
    ["../node_modules/.pnpm/astro@5.16.6_@types+node@22.19.3_@vercel+functions@2.2.13_rollup@4.54.0_tsx@4.21.0_typescript@5.9.3_yaml@2.8.2/node_modules/astro/dist/assets/endpoint/generic.js", _page0],
    ["../node_modules/.pnpm/starlight-llms-txt@0.6.0_@astrojs+starlight@0.37.1_astro@5.16.6_@types+node@22.19.3_@vercel+f_vauonzymqefqlfmkvdnomgvevi/node_modules/starlight-llms-txt/llms-custom.txt.ts", _page1],
    ["../node_modules/.pnpm/@astrojs+starlight@0.37.1_astro@5.16.6_@types+node@22.19.3_@vercel+functions@2.2.13_rollup@4._xy3a4zocypmfv67nnw6v2uns3e/node_modules/@astrojs/starlight/routes/static/404.astro", _page2],
    ["src/pages/api/search.ts", _page3],
    ["../node_modules/.pnpm/starlight-llms-txt@0.6.0_@astrojs+starlight@0.37.1_astro@5.16.6_@types+node@22.19.3_@vercel+f_vauonzymqefqlfmkvdnomgvevi/node_modules/starlight-llms-txt/llms-full.txt.ts", _page4],
    ["../node_modules/.pnpm/starlight-llms-txt@0.6.0_@astrojs+starlight@0.37.1_astro@5.16.6_@types+node@22.19.3_@vercel+f_vauonzymqefqlfmkvdnomgvevi/node_modules/starlight-llms-txt/llms-small.txt.ts", _page5],
    ["../node_modules/.pnpm/starlight-llms-txt@0.6.0_@astrojs+starlight@0.37.1_astro@5.16.6_@types+node@22.19.3_@vercel+f_vauonzymqefqlfmkvdnomgvevi/node_modules/starlight-llms-txt/llms.txt.ts", _page6],
    ["../node_modules/.pnpm/@astrojs+starlight@0.37.1_astro@5.16.6_@types+node@22.19.3_@vercel+functions@2.2.13_rollup@4._xy3a4zocypmfv67nnw6v2uns3e/node_modules/@astrojs/starlight/routes/static/index.astro", _page7]
]);

const _manifest = Object.assign(manifest, {
    pageMap,
    serverIslandMap,
    renderers,
    actions: () => import('./noop-entrypoint.mjs'),
    middleware: () => import('./_astro-internal_middleware.mjs')
});
const _args = {
    "middlewareSecret": "b5e306c0-f7be-4c12-b558-5842e7c619e7",
    "skewProtection": false
};
const _exports = createExports(_manifest, _args);
const __astrojsSsrVirtualEntry = _exports.default;
const _start = 'start';
if (Object.prototype.hasOwnProperty.call(serverEntrypointModule, _start)) ;

export { __astrojsSsrVirtualEntry as default, pageMap };
