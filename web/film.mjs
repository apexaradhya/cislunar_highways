// Record the story mode in real time to a video (Playwright's screen capture), then ffmpeg -> MP4.
// usage: node web/film.mjs <html> <vendorDir> <outDir> <width> <height> <seconds>
import { createRequire } from "module";
const { chromium } = createRequire(import.meta.url)("playwright");
import { readFileSync, mkdirSync } from "fs";
import path from "path";
const [html, vendor, outDir, W = "1280", H = "720", secs = "112"] = process.argv.slice(2);
mkdirSync(outDir, { recursive: true });
const browser = await chromium.launch({ args: ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"] });
const ctx = await browser.newContext({ viewport: { width: +W, height: +H }, recordVideo: { dir: outDir, size: { width: +W, height: +H } } });
const page = await ctx.newPage();
await page.route("https://cdn.jsdelivr.net/npm/three@0.160.0/**", (r) =>
  r.fulfill({ body: readFileSync(path.join(vendor, r.request().url().split("three@0.160.0/")[1])), contentType: "application/javascript" }));
await page.goto("file://" + path.resolve(html) + "#film");
await page.waitForFunction(() => window.__cislunar);
await page.waitForTimeout(+secs * 1000);
const fps = await page.evaluate(() => new Promise((res) => { let n = 0; const t0 = performance.now(); (function f() { n++; if (performance.now() - t0 < 2000) requestAnimationFrame(f); else res(n / 2); })(); }));
console.log("render fps during capture ~", fps.toFixed(1));
const video = page.video();
await ctx.close();
console.log("raw video:", await video.path());
await browser.close();
