'use strict';
/**
 * demo_recorder.cjs
 * Records a CareVoice demo video by scrolling through demo_notebook.html.
 *
 * Usage:
 *   node demo_recorder.cjs            # record
 *   node demo_recorder.cjs --rehearse # dry-run (check anchors only)
 *
 * Output: demo_output/carevoice-demo.webm
 */

const { chromium } = require('playwright');
const path = require('path');
const fs = require('fs');

const HTML_PATH = path.resolve(__dirname, 'demo_notebook.html');
const OUTPUT_DIR = path.resolve(__dirname, 'demo_output');
const OUTPUT_FILE = 'carevoice-demo.webm';
const REHEARSAL = process.argv.includes('--rehearse');
const W = 1280, H = 720;

async function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

async function injectOverlays(page) {
  await page.evaluate(() => {
    if (!document.getElementById('demo-cursor')) {
      const c = document.createElement('div');
      c.id = 'demo-cursor';
      c.innerHTML = `<svg width="24" height="24" viewBox="0 0 24 24">
        <path d="M5 3L19 12L12 13L9 20L5 3Z" fill="white" stroke="black" stroke-width="1.5" stroke-linejoin="round"/>
      </svg>`;
      c.style.cssText = 'position:fixed;z-index:999999;pointer-events:none;width:24px;height:24px;transition:left 0.08s,top 0.08s;filter:drop-shadow(1px 1px 2px rgba(0,0,0,0.4));left:0;top:0;';
      document.body.appendChild(c);
      document.addEventListener('mousemove', e => {
        c.style.left = e.clientX + 'px';
        c.style.top = e.clientY + 'px';
      });
    }
    if (!document.getElementById('demo-sub')) {
      const s = document.createElement('div');
      s.id = 'demo-sub';
      s.style.cssText = 'position:fixed;bottom:0;left:0;right:0;z-index:999998;text-align:center;padding:14px 32px;background:rgba(0,0,0,0.82);color:#fff;font-family:-apple-system,"Segoe UI",sans-serif;font-size:15px;font-weight:500;letter-spacing:0.3px;transition:opacity 0.3s;pointer-events:none;opacity:0;';
      document.body.appendChild(s);
    }
  });
}

async function subtitle(page, text) {
  await page.evaluate(t => {
    const s = document.getElementById('demo-sub');
    if (s) { s.textContent = t; s.style.opacity = t ? '1' : '0'; }
  }, text);
  if (text) await sleep(600);
}

async function scrollTo(page, yTarget) {
  const current = await page.evaluate(() => window.scrollY);
  const steps = 40;
  const delta = (yTarget - current) / steps;
  for (let i = 0; i < steps; i++) {
    await page.evaluate(d => window.scrollBy(0, d), delta);
    await sleep(25);
  }
  await sleep(400);
}

async function scrollToAnchor(page, id) {
  const y = await page.evaluate(id => {
    const el = document.getElementById(id);
    return el ? Math.max(0, el.getBoundingClientRect().top + window.scrollY - 30) : 0;
  }, id);
  await scrollTo(page, y);
}

async function moveCursor(page, x, y) {
  await page.mouse.move(x, y, { steps: 20 });
  await sleep(300);
}

(async () => {
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  const browser = await chromium.launch({ headless: true });

  // ── Phase 2: Rehearsal ──────────────────────────────────────────────────────
  if (REHEARSAL) {
    console.log('=== REHEARSAL MODE ===');
    const ctx = await browser.newContext({ viewport: { width: W, height: H } });
    const page = await ctx.newPage();
    await page.goto('file://' + HTML_PATH);
    await page.waitForLoadState('domcontentloaded');

    const anchors = ['scene-anchor-1','scene-anchor-2','scene-anchor-3',
                     'scene-anchor-4','scene-anchor-5','scene-anchor-6',
                     'scene-anchor-7','scene-anchor-8'];
    let allOk = true;
    for (const id of anchors) {
      const exists = await page.evaluate(id => !!document.getElementById(id), id);
      console.log(`  ${exists ? '✅' : '❌'}  #${id}`);
      if (!exists) allOk = false;
    }
    await ctx.close();
    await browser.close();
    console.log(allOk ? '\nREHEARSAL PASSED ✅' : '\nREHEARSAL FAILED ❌');
    process.exit(allOk ? 0 : 1);
  }

  // ── Phase 3: Record ─────────────────────────────────────────────────────────
  console.log('=== RECORDING ===');
  const ctx = await browser.newContext({
    recordVideo: { dir: OUTPUT_DIR, size: { width: W, height: H } },
    viewport: { width: W, height: H },
  });
  const page = await ctx.newPage();

  try {
    await page.goto('file://' + HTML_PATH);
    await page.waitForLoadState('domcontentloaded');
    await injectOverlays(page);
    await sleep(1500);

    // Scene 1 — Problem
    await subtitle(page, 'CareVoice — Trimodal Clinical Intake with Gemma 4');
    await moveCursor(page, 640, 180);
    await sleep(2800);
    await subtitle(page, 'Scene 1 — The Problem');
    await sleep(2000);
    await subtitle(page, '40 patients · no internet · 6+ hours of manual intake per day');
    await sleep(3500);
    await subtitle(page, 'CareVoice gives those hours back — offline, in any language');
    await sleep(3000);

    // Scene 2 — Text red flag
    await subtitle(page, 'Scene 2 — Text Intake: Stroke Red Flag');
    await scrollToAnchor(page, 'scene-anchor-2');
    await moveCursor(page, 500, 320);
    await sleep(2000);
    await subtitle(page, 'A family member types three symptoms in broken English');
    await sleep(2800);
    await subtitle(page, 'urgent=true · triage=red — in under 5 minutes, CPU-only, no cloud');
    await sleep(3000);
    await subtitle(page, 'Same result in Spanish and French — zero configuration');
    await sleep(3000);

    // Scene 3 — Image
    await subtitle(page, 'Scene 3 — Image Triage: Wound Photo vs Surgeon Ground Truth');
    await scrollToAnchor(page, 'scene-anchor-3');
    await moveCursor(page, 400, 380);
    await sleep(2000);
    await subtitle(page, 'Patient photographs a surgical wound — model triages vs surgeon GT');
    await sleep(3000);
    await subtitle(page, '67% accuracy on SurgWound dataset (CC BY-SA 4.0, 697 images)');
    await sleep(3000);

    // Scene 4 — Audio
    await subtitle(page, 'Scene 4 — Audio: Respiratory Sound Analysis');
    await scrollToAnchor(page, 'scene-anchor-4');
    await moveCursor(page, 400, 380);
    await sleep(2000);
    await subtitle(page, '4-second recording on a basic phone — Gemma 4 native audio encoder');
    await sleep(2800);
    await subtitle(page, 'Normal detection: 2/2 — zero false positives  ·  SPRSound CC BY 4.0');
    await sleep(3000);

    // Scene 5 — Multilingual
    await subtitle(page, 'Scene 5 — Multilingual Auto-Detection');
    await scrollToAnchor(page, 'scene-anchor-5');
    await moveCursor(page, 640, 380);
    await sleep(2000);
    await subtitle(page, 'Tagalog · French · Spanish — no language setting, no configuration');
    await sleep(3000);
    await subtitle(page, '100+ languages via Gemma 4 multilingual pretraining');
    await sleep(3000);

    // Scene 6 — Summary
    await subtitle(page, 'Scene 6 — Evaluation Summary');
    await scrollToAnchor(page, 'scene-anchor-6');
    await moveCursor(page, 640, 400);
    await sleep(2000);
    await subtitle(page, 'Text ✅  Image 67% ✅  Audio ✅  Multilingual ✅  CPU-only ✅');
    await sleep(3500);

    // Scene 7 — Ollama
    await subtitle(page, 'Scene 7 — Edge Deployment: Ollama on 4 GB RAM');
    await scrollToAnchor(page, 'scene-anchor-7');
    await moveCursor(page, 640, 400);
    await sleep(2000);
    await subtitle(page, 'Two commands — $50 tablet, clinic Raspberry Pi, no internet');
    await sleep(3500);

    // Scene 8 — Close
    await subtitle(page, '');
    await scrollTo(page, 999999);
    await moveCursor(page, 640, 480);
    await sleep(1500);
    await subtitle(page, '1.8 billion people live without reliable access to clinical care');
    await sleep(3000);
    await subtitle(page, 'CareVoice — on the device they have, in the language they speak');
    await sleep(3500);
    await subtitle(page, '');
    await sleep(2500);

    console.log('Recording complete.');
  } catch (err) {
    console.error('RECORDING ERROR:', err.message);
  } finally {
    await ctx.close();
    const files = fs.readdirSync(OUTPUT_DIR).filter(f => f.endsWith('.webm'));
    if (files.length > 0) {
      const src = path.join(OUTPUT_DIR, files[0]);
      const dst = path.join(OUTPUT_DIR, OUTPUT_FILE);
      if (src !== dst) fs.renameSync(src, dst);
      const sizeMB = Math.round(fs.statSync(dst).size / 1024 / 1024 * 10) / 10;
      console.log(`\n✅ Video saved: ${dst}`);
      console.log(`   Size: ${sizeMB} MB`);
    } else {
      console.error('❌ No video file produced in', OUTPUT_DIR);
    }
    await browser.close();
  }
})();
