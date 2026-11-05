// headless explorer render plus cam adjustments
import { createRequire } from "module";
const { chromium } = createRequire(import.meta.url)("playwright");
import { readFileSync, mkdirSync } from "fs";
import path from "path";
const [html, vendor, outDir, W = "1600", H = "900"] = process.argv.slice(2);
mkdirSync(outDir, { recursive: true });
const browser = await chromium.launch({ args: ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"] });
const page = await browser.newPage({ viewport: { width: +W, height: +H } });
const errors = [];
page.on("pageerror", (e) => errors.push(String(e)));
page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
await page.route("https://cdn.jsdelivr.net/npm/three@0.160.0/**", (route) => {
  const rel = route.request().url().split("three@0.160.0/")[1];
  route.fulfill({ body: readFileSync(path.join(vendor, rel)), contentType: "application/javascript" });
});
await page.goto("file://" + path.resolve(html) + "#record");
await page.waitForFunction(() => window.__cislunar, null, { timeout: 60000 });
const shots = JSON.parse(process.env.SHOTS || "[[0,6],[2,9],[4,9],[6,12],[7,9],[8,6]]");
for (const [ch, secs] of shots) {
  await page.evaluate(([c, s]) => { window.__cislunar.go(c); for (let i = 0; i < s * 20; i++) window.__cislunar.step(0.05); }, [ch, secs]);
  await page.screenshot({ path: `${outDir}/ch${ch}.png`, timeout: 120000 });
}
if (process.env.EXPLORE) {
  await page.evaluate(() => { window.__cislunar.mode("explore"); window.__cislunar.fly(16); for (let i = 0; i < 120; i++) window.__cislunar.step(0.05); });
  await page.screenshot({ path: `${outDir}/explore.png`, timeout: 120000 });
}
console.log(errors.length ? "ERRORS:\n" + errors.join("\n") : "no page errors");
await browser.close();
