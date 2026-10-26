// Deterministic film: step the story clock by 1/fps and screenshot every frame (JPEG), for ffmpeg.
// usage: node web/frames.mjs <html> <vendorDir> <outDir> <width> <height> <fps> [maxFrames]
import { createRequire } from "module";
const { chromium } = createRequire(import.meta.url)("playwright");
import { readFileSync, mkdirSync } from "fs";
import path from "path";
const [html, vendor, outDir, W = "1280", H = "720", FPS = "24", MAX = "0"] = process.argv.slice(2);
mkdirSync(outDir, { recursive: true });
const browser = await chromium.launch({ args: ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"] });
const page = await browser.newPage({ viewport: { width: +W, height: +H } });
await page.route("https://cdn.jsdelivr.net/npm/three@0.160.0/**", (r) =>
  r.fulfill({ body: readFileSync(path.join(vendor, r.request().url().split("three@0.160.0/")[1])), contentType: "application/javascript" }));
await page.goto("file://" + path.resolve(html) + "#record");
await page.waitForFunction(() => window.__cislunar);
const total = await page.evaluate(() => window.__cislunar.chapters.reduce((a, b) => a + b, 0));
const n = +MAX || Math.ceil((total + 1.5) * +FPS);
const t0 = Date.now();
for (let i = 0; i < n; i++) {
  await page.evaluate((dt) => window.__cislunar.step(dt), 1 / +FPS);
  await page.screenshot({ path: `${outDir}/f${String(i).padStart(5, "0")}.jpg`, type: "jpeg", quality: 92, timeout: 120000 });
  if (i % 240 === 0) console.log(`frame ${i}/${n}  ${((Date.now() - t0) / 1000).toFixed(0)} s`);
}
console.log(`done: ${n} frames in ${((Date.now() - t0) / 1000).toFixed(0)} s`);
await browser.close();
