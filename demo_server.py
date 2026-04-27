"""CareVoice — Interactive Demo Server (FastAPI + mock model, no GPU required).

Usage:
    python solutions/gemma_hackathon/demo_server.py
    # then open http://localhost:8000

For real Gemma 4 inference (Kaggle GPU):
    python solutions/gemma_hackathon/demo_server.py --real-model
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

from solutions.gemma_hackathon.intake_assistant import (
    IntakeAssistant,
    IntakeRecord,
    _parse_json_response,
    _apply_extraction,
)
from solutions.gemma_hackathon.scenarios import SAMPLE_SCENARIOS

# Extended corpus — optional, graceful fallback
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from tests.corpus.generator import EXTENDED_CORPUS
except ImportError:
    EXTENDED_CORPUS = []

# ---------------------------------------------------------------------------
# App + shared state
# ---------------------------------------------------------------------------

app = FastAPI(title="CareVoice Demo")
_SESSIONS: dict[str, dict] = {}
_ASSISTANT: IntakeAssistant | None = None
_REMOTE_URL: str | None = None          # e.g. "https://abc.trycloudflare.com"

_SCRIPTED_IDS = {s["scenario_id"] for s in SAMPLE_SCENARIOS}


def _get_assistant() -> IntakeAssistant:
    global _ASSISTANT
    if _ASSISTANT is None:
        _ASSISTANT = IntakeAssistant.mock()
    return _ASSISTANT


# ---------------------------------------------------------------------------
# Smart keyword mock for synthetic gen_* scenarios
# Detects red flags and extracts chief complaint from raw patient text
# so the 500 extended scenarios are genuinely interactive in the demo.
# ---------------------------------------------------------------------------

_RF_KEYWORDS: list[tuple[list[str], str]] = [
    (["chest pain", "chest tight", "chest pressure", "crushing", "heart attack",
      "arm numb", "jaw pain", "palpitation", "heart racing"],
     "cardiac"),
    (["can't breathe", "cannot breathe", "difficulty breathing", "shortness of breath",
      "suffocating", "blue lips", "bluish", "oxygen 8", "gasping"],
     "respiratory_distress"),
    (["end my life", "ending things", "suicid", "kill myself", "don't want to live",
      "no reason to live", "goodbye note", "gave away my belonging", "no point in living"],
     "suicidal_ideation"),
    (["coughing blood", "coughing up blood", "blood won't stop", "heavy bleeding",
      "soaked through", "blood in stool", "vaginal bleeding", "haemorrhage"],
     "bleeding"),
    (["face drooping", "drooping", "slurred speech", "arm suddenly weak",
      "sudden confusion", "can't lift", "stroke", "worst headache of my life"],
     "stroke"),
    (["throat swelling", "throat closing", "anaphylaxis", "hives everywhere",
      "lips swelling", "difficulty swallowing", "allergic reaction", "epipen"],
     "anaphylaxis"),
]

_RF_MESSAGES: dict[str, str] = {
    "cardiac":           "This sounds like it could be a cardiac emergency. Are you experiencing any chest tightness, shortness of breath, or sweating right now?",
    "respiratory_distress": "Breathing difficulty can be very serious. Are your lips or fingernails turning blue? Do you have an inhaler available?",
    "suicidal_ideation": "I hear you, and I want you to know you are not alone. Are you having thoughts of harming yourself or ending your life right now?",
    "bleeding":          "Significant bleeding requires immediate attention. Is the bleeding ongoing? Are you feeling faint or dizzy?",
    "stroke":            "These symptoms could indicate a stroke. Can you smile, raise both arms, and repeat a simple sentence for me?",
    "anaphylaxis":       "This sounds like a severe allergic reaction. Do you have an EpiPen available? Are you having any difficulty breathing?",
}

_FOLLOWUP = [
    "Can you tell me more about when this started and how severe it is on a scale of 1 to 10?",
    "Do you have any relevant medical history, current medications, or allergies I should note?",
    "Thank you. I have noted your information and your intake record is ready for the clinician.",
]


def _smart_response(patient_input: str, turn_idx: int) -> dict:
    """Keyword-based mock response for gen_* scenarios."""
    text = patient_input.lower()

    # Detect red flag
    rf_cat = None
    rf_reason = None
    for keywords, category in _RF_KEYWORDS:
        if any(k in text for k in keywords):
            rf_cat = category
            rf_reason = {
                "cardiac":           "Chest pain / cardiac symptoms — urgent assessment required",
                "respiratory_distress": "Respiratory distress — urgent assessment required",
                "suicidal_ideation": "Suicidal ideation — immediate safety assessment required",
                "bleeding":          "Active bleeding — urgent assessment required",
                "stroke":            "Neurological emergency / possible stroke — FAST assessment required",
                "anaphylaxis":       "Possible anaphylaxis — urgent assessment required",
            }[category]
            break

    # Build message
    if rf_cat:
        message = _RF_MESSAGES[rf_cat]
    elif turn_idx == 0:
        message = "Thank you for telling me. Can you describe when these symptoms started and how severe they feel on a scale of 1 to 10?"
    elif turn_idx == 1:
        message = "I understand. Do you have any relevant medical history, current medications, or allergies I should know about?"
    else:
        message = "Thank you. I have noted your information. Is there anything else you would like to add before I prepare your intake summary?"

    # Extract chief complaint on turn 0
    extracted_field = None
    extracted_value = None
    if turn_idx == 0:
        extracted_field = "chief_complaint"
        extracted_value = patient_input.split(".")[0].strip()[:120]

    return {
        "message": message,
        "extracted_field": extracted_field,
        "extracted_value": extracted_value,
        "confidence": 0.78 if rf_cat else 0.72,
        "red_flag": bool(rf_cat),
        "red_flag_reason": rf_reason,
        "intake_complete": turn_idx >= 2,
    }


# ---------------------------------------------------------------------------
# Build unified scenario menu (original 10 + extended 500)
# ---------------------------------------------------------------------------

def _menu_entry(s: dict, source: str) -> dict:
    return {
        "id":               s["scenario_id"],
        "description":      s["description"],
        "language":         s.get("language", "en").upper(),
        "category":         s.get("category", ""),
        "expected_red_flag": s.get("expected_red_flag", False),
        "red_flag_category": s.get("red_flag_category") or "",
        "turns":            s["turns"],
        "source":           source,
        "demographic":      s.get("demographic", {}),
    }


SCENARIO_MENU = (
    [_menu_entry(s, "original") for s in SAMPLE_SCENARIOS] +
    [_menu_entry(s, "synthetic") for s in EXTENDED_CORPUS]
)

TOTAL = len(SCENARIO_MENU)
ORIG_COUNT = len(SAMPLE_SCENARIOS)
EXT_COUNT = len(EXTENDED_CORPUS)


# ---------------------------------------------------------------------------
# HTML (inline, no jinja2)
# ---------------------------------------------------------------------------

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>CareVoice — Clinical Intake Demo</title>
<style>
  :root{--bg:#0f1117;--surface:#1a1d27;--surface2:#22263a;--border:#2d3148;
        --accent:#4f8ef7;--accent2:#6ee7b7;--danger:#f87171;--warn:#fbbf24;
        --text:#e2e8f0;--muted:#64748b;--radius:10px;}
  *{box-sizing:border-box;margin:0;padding:0;}
  body{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;height:100vh;display:flex;flex-direction:column;}

  /* Header */
  header{background:var(--surface);border-bottom:1px solid var(--border);padding:10px 20px;display:flex;align-items:center;gap:12px;flex-shrink:0;}
  .logo{font-size:1.25rem;font-weight:700;color:var(--accent2);}
  .subtitle{font-size:0.76rem;color:var(--muted);}
  .hbadge{margin-left:auto;display:flex;gap:8px;}
  .badge{background:var(--surface2);border:1px solid var(--border);padding:3px 9px;border-radius:20px;font-size:0.7rem;color:var(--muted);}
  .badge.green{border-color:var(--accent2);color:var(--accent2);}

  /* Layout */
  .layout{display:grid;grid-template-columns:290px 1fr 295px;flex:1;overflow:hidden;min-height:0;}

  /* Sidebar */
  .sidebar{background:var(--surface);border-right:1px solid var(--border);display:flex;flex-direction:column;overflow:hidden;}
  .sb-filters{padding:10px 12px;border-bottom:1px solid var(--border);display:flex;flex-direction:column;gap:6px;flex-shrink:0;}
  .sb-search{background:var(--surface2);border:1px solid var(--border);border-radius:7px;padding:6px 10px;color:var(--text);font-size:0.8rem;width:100%;outline:none;}
  .sb-search:focus{border-color:var(--accent);}
  .sb-row{display:flex;gap:5px;}
  .sb-sel{background:var(--surface2);border:1px solid var(--border);border-radius:6px;padding:4px 6px;color:var(--text);font-size:0.72rem;flex:1;outline:none;cursor:pointer;}
  .sb-toggle{background:var(--surface2);border:1px solid var(--border);border-radius:6px;padding:4px 8px;color:var(--muted);font-size:0.72rem;cursor:pointer;white-space:nowrap;}
  .sb-toggle.on{border-color:var(--danger);color:var(--danger);}
  .sb-count{font-size:0.67rem;color:var(--muted);padding:0 12px 6px;}
  .sb-list{flex:1;overflow-y:auto;}

  /* Scenario buttons */
  .sc-btn{display:block;width:100%;text-align:left;padding:8px 12px;background:none;border:none;cursor:pointer;color:var(--text);border-left:3px solid transparent;transition:background .12s;}
  .sc-btn:hover{background:var(--surface2);}
  .sc-btn.active{background:var(--surface2);border-left-color:var(--accent);}
  .sc-id{font-size:0.65rem;font-weight:600;color:var(--accent);}
  .sc-desc{font-size:0.77rem;margin-top:1px;line-height:1.3;}
  .sc-meta{font-size:0.65rem;color:var(--muted);margin-top:2px;}
  .rf-dot{display:inline-block;width:6px;height:6px;border-radius:50%;background:var(--danger);margin-right:3px;vertical-align:middle;}
  .src-orig{color:var(--accent2);font-size:0.62rem;font-weight:600;}
  .src-syn{color:var(--muted);font-size:0.62rem;}
  .sb-more{padding:8px 12px;font-size:0.72rem;color:var(--accent);cursor:pointer;text-align:center;border-top:1px solid var(--border);}
  .sb-more:hover{background:var(--surface2);}

  /* Chat */
  .chat-pane{display:flex;flex-direction:column;overflow:hidden;min-height:0;}
  .chat-msgs{flex:1;overflow-y:auto;padding:18px 20px;display:flex;flex-direction:column;gap:12px;min-height:0;}
  .msg{max-width:76%;}
  .msg.patient{align-self:flex-end;}
  .msg.assistant{align-self:flex-start;}
  .bubble{padding:10px 14px;border-radius:var(--radius);font-size:0.85rem;line-height:1.55;}
  .msg.patient .bubble{background:var(--accent);color:#fff;border-bottom-right-radius:2px;}
  .msg.assistant .bubble{background:var(--surface2);border:1px solid var(--border);border-bottom-left-radius:2px;}
  .mmeta{font-size:0.67rem;color:var(--muted);margin-top:3px;padding:0 3px;}
  .msg.patient .mmeta{text-align:right;}

  .rf-alert{background:rgba(248,113,113,0.1);border:1px solid var(--danger);border-radius:var(--radius);padding:9px 13px;font-size:0.8rem;color:var(--danger);display:flex;gap:8px;align-items:flex-start;}
  .complete-banner{background:rgba(110,231,183,0.08);border:1px solid var(--accent2);border-radius:var(--radius);padding:9px;font-size:0.79rem;color:var(--accent2);text-align:center;}
  .syn-note{background:rgba(251,191,36,0.08);border:1px solid var(--warn);border-radius:var(--radius);padding:8px 12px;font-size:0.75rem;color:var(--warn);}

  .toolbar{border-top:1px solid var(--border);background:var(--surface);padding:11px 18px;display:flex;gap:9px;align-items:center;flex-shrink:0;}
  .turn-info{font-size:0.72rem;color:var(--muted);flex:1;}
  .btn{padding:7px 16px;border-radius:8px;border:none;cursor:pointer;font-size:0.82rem;font-weight:600;transition:opacity .15s;}
  .btn:disabled{opacity:0.35;cursor:not-allowed;}
  .btn-p{background:var(--accent);color:#fff;}
  .btn-s{background:var(--surface2);border:1px solid var(--border);color:var(--text);}

  /* Record pane */
  .rec-pane{background:var(--surface);border-left:1px solid var(--border);overflow-y:auto;padding:14px;}
  .rec-pane h3{font-size:0.67rem;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);margin-bottom:11px;}
  .rf{margin-bottom:9px;}
  .rl{font-size:0.64rem;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin-bottom:2px;}
  .rv{font-size:0.81rem;}
  .rv.empty{color:var(--muted);font-style:italic;}
  .tag{display:inline-block;background:var(--surface2);border:1px solid var(--border);border-radius:4px;padding:2px 6px;font-size:0.72rem;margin:1px;}
  .tag.danger{border-color:var(--danger);color:var(--danger);}
  .divider{border:none;border-top:1px solid var(--border);margin:10px 0;}
  .cbar{height:5px;background:var(--border);border-radius:3px;overflow:hidden;margin-top:4px;}
  .cfill{height:100%;background:var(--accent2);border-radius:3px;transition:width .5s ease;}

  .welcome{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:10px;color:var(--muted);text-align:center;padding:24px;}
  .welcome h2{color:var(--text);font-size:1rem;}
  .welcome p{font-size:0.82rem;max-width:320px;}

  ::-webkit-scrollbar{width:5px;}
  ::-webkit-scrollbar-track{background:transparent;}
  ::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px;}
</style>
</head>
<body>

<header>
  <div class="logo">🩺 CareVoice</div>
  <div class="subtitle">Offline Multilingual Clinical Intake &mdash; Gemma 4 Good Hackathon</div>
  <div class="hbadge">
    <div class="badge green" id="scenario-count-badge">TOTAL_PLACEHOLDER scenarios</div>
    <div class="badge" id="model-badge">mock model &bull; no GPU</div>
    <div class="badge" id="remote-badge" style="display:none">&#128308; remote offline</div>
  </div>
</header>

<div class="layout">

  <!-- Sidebar -->
  <div class="sidebar">
    <div class="sb-filters">
      <input class="sb-search" id="search" type="text" placeholder="&#128269; Search scenarios…" oninput="renderList()">
      <div class="sb-row">
        <select class="sb-sel" id="f-lang" onchange="renderList()">
          <option value="">All languages</option>
          <option value="EN">EN</option>
          <option value="ES">ES</option>
          <option value="FR">FR</option>
        </select>
        <select class="sb-sel" id="f-src" onchange="renderList()">
          <option value="">All scenarios</option>
          <option value="original">Original 10</option>
          <option value="synthetic">Synthetic 500</option>
        </select>
        <button class="sb-toggle" id="f-rf" onclick="toggleRf()">🚨 RF only</button>
      </div>
    </div>
    <div class="sb-count" id="sb-count"></div>
    <div class="sb-list" id="sb-list"></div>
    <div class="sb-more" id="sb-more" onclick="showMore()" style="display:none">Show more…</div>
  </div>

  <!-- Chat -->
  <div class="chat-pane">
    <div class="chat-msgs" id="chat-msgs">
      <div class="welcome">
        <h2>Select a patient scenario</h2>
        <p>CareVoice conducts structured multilingual intake conversations, extracts clinical fields in real time, and flags emergencies instantly.</p>
        <p style="margin-top:8px;font-size:0.73rem">TOTAL_PLACEHOLDER scenarios &bull; EN / ES / FR &bull; 6 red-flag categories</p>
      </div>
    </div>
    <div class="toolbar">
      <div class="turn-info" id="turn-info">&mdash;</div>
      <button class="btn btn-s" id="btn-reset" onclick="resetSess()" disabled>&#8635; Reset</button>
      <button class="btn btn-p" id="btn-next" onclick="nextTurn()" disabled>Next Turn &#9654;</button>
    </div>
  </div>

  <!-- Record -->
  <div class="rec-pane">
    <h3>Live Intake Record</h3>
    <div id="rec"><div style="color:var(--muted);font-size:0.8rem">Start a scenario to see the record fill in real time.</div></div>
  </div>

</div>

<script>
const ALL = SCENARIO_DATA_PLACEHOLDER;
const PAGE = 80;
let shown = PAGE, rfOnly = false, cur = null, sid = null, turn = 0;

// Replace placeholders in text nodes
document.querySelectorAll('.welcome p, #scenario-count-badge').forEach(el => {
  el.innerHTML = el.innerHTML.replace(/TOTAL_PLACEHOLDER/g, ALL.length);
});

function toggleRf() {
  rfOnly = !rfOnly;
  const b = document.getElementById('f-rf');
  b.classList.toggle('on', rfOnly);
  shown = PAGE;
  renderList();
}

function filtered() {
  const q = document.getElementById('search').value.toLowerCase();
  const lang = document.getElementById('f-lang').value;
  const src = document.getElementById('f-src').value;
  return ALL.filter(s =>
    (!q || s.description.toLowerCase().includes(q) || s.id.includes(q) || s.category.includes(q)) &&
    (!lang || s.language === lang) &&
    (!src  || s.source === src) &&
    (!rfOnly || s.expected_red_flag)
  );
}

function renderList() {
  shown = PAGE;
  _render();
}

function showMore() { shown += PAGE; _render(); }

function _render() {
  const list = document.getElementById('sb-list');
  const more = document.getElementById('sb-more');
  const matches = filtered();
  document.getElementById('sb-count').textContent =
    `Showing ${Math.min(shown, matches.length)} of ${matches.length} scenario${matches.length!==1?'s':''}`;

  list.innerHTML = '';
  matches.slice(0, shown).forEach(s => {
    const b = document.createElement('button');
    b.className = 'sc-btn' + (cur && cur.id === s.id ? ' active' : '');
    b.id = 'sb-' + s.id;
    b.onclick = () => select(s);
    const rf = s.expected_red_flag ? '<span class="rf-dot"></span>' : '';
    const srcBadge = s.source === 'original'
      ? `<span class="src-orig">★ scripted</span>`
      : `<span class="src-syn">synthetic</span>`;
    const cat = s.category ? ` &bull; ${s.category.replace(/_/g,' ')}` : '';
    b.innerHTML = `
      <div class="sc-id">${s.id.replace('_',' ').toUpperCase()} ${srcBadge}</div>
      <div class="sc-desc">${esc(s.description)}</div>
      <div class="sc-meta">${rf}${s.language}${cat}${s.expected_red_flag?' &bull; red flag':''}</div>`;
    list.appendChild(b);
  });

  more.style.display = shown < matches.length ? '' : 'none';
}

function select(s) {
  cur = s; sid = 'sess_' + Date.now(); turn = 0;
  _render(); // re-render so active highlight updates

  const msgs = document.getElementById('chat-msgs');
  const demo = s.source === 'original'
    ? '' : '<div class="syn-note">&#9432; Synthetic scenario — responses use keyword-based mock inference.</div>';
  const rf = s.expected_red_flag
    ? `<span style="color:var(--danger)"> &bull; &#128680; red flag expected</span>` : '';
  msgs.innerHTML = `${demo}<div style="color:var(--muted);font-size:0.81rem;align-self:center;text-align:center">
    <strong>${esc(s.description)}</strong><br>${s.language} &bull; ${s.turns.length} turn${s.turns.length!==1?'s':''}${rf}</div>`;

  updateRec(null);
  document.getElementById('turn-info').textContent = `Turn 1 of ${s.turns.length}`;
  document.getElementById('btn-next').disabled = false;
  document.getElementById('btn-next').textContent = 'Next Turn \u25B6';
  document.getElementById('btn-reset').disabled = false;
}

async function nextTurn() {
  if (!cur || turn >= cur.turns.length) return;
  const btn = document.getElementById('btn-next');
  btn.disabled = true; btn.textContent = 'Processing\u2026';
  const input = cur.turns[turn];
  addMsg('patient', input, `Turn ${turn + 1} of ${cur.turns.length}`);
  try {
    const r = await fetch('/turn', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({session_id:sid, scenario_id:cur.id, patient_input:input, turn, source:cur.source})
    });
    const d = await r.json();
    const meta = `conf ${Math.round(d.confidence*100)}%` + (d.extracted_field ? ` \u00b7 ${d.extracted_field}` : '');
    addMsg('assistant', d.message, meta);
    if (d.red_flag && d.red_flag_reason) addAlert(d.red_flag_reason);
    updateRec(d.record);
    turn++;
    if (turn >= cur.turns.length) {
      btn.textContent = '\u2713 Complete';
      document.getElementById('turn-info').textContent = `Complete (${cur.turns.length}/${cur.turns.length} turns)`;
      addBanner();
    } else {
      btn.disabled = false; btn.textContent = 'Next Turn \u25B6';
      document.getElementById('turn-info').textContent = `Turn ${turn+1} of ${cur.turns.length}`;
    }
  } catch(e) {
    addMsg('assistant','Error: '+e.message,'');
    btn.disabled=false; btn.textContent='Next Turn \u25B6';
  }
}

function resetSess() { if (cur) select(cur); }

function addMsg(role, text, meta) {
  const el = document.getElementById('chat-msgs');
  const d = document.createElement('div'); d.className='msg '+role;
  d.innerHTML=`<div class="bubble">${esc(text)}</div><div class="mmeta">${esc(meta)}</div>`;
  el.appendChild(d); el.scrollTop=el.scrollHeight;
}
function addAlert(r) {
  const el = document.getElementById('chat-msgs');
  const d = document.createElement('div'); d.className='rf-alert';
  d.innerHTML=`<span>&#128680;</span><strong>RED FLAG</strong> &mdash; ${esc(r)}`;
  el.appendChild(d); el.scrollTop=el.scrollHeight;
}
function addBanner() {
  const el = document.getElementById('chat-msgs');
  const d = document.createElement('div'); d.className='complete-banner';
  d.textContent='\u2713 Intake record ready for clinician handoff';
  el.appendChild(d); el.scrollTop=el.scrollHeight;
}

function updateRec(rec) {
  const el = document.getElementById('rec');
  if (!rec) { el.innerHTML='<div style="color:var(--muted);font-size:0.8rem">Start a scenario to see the record fill in real time.</div>'; return; }
  const conf = Math.round((rec.overall_confidence||0)*100);
  el.innerHTML =
    fld('Chief Complaint', rec.chief_complaint) +
    fld('Duration', rec.symptom_duration) +
    fld('Severity', rec.symptom_severity ? rec.symptom_severity+'/10' : '') +
    lf('Associated Symptoms', rec.associated_symptoms) +
    lf('Medical History', rec.medical_history) +
    lf('Medications', rec.current_medications) +
    lf('Allergies', rec.allergies) +
    '<hr class="divider">' +
    lf('\u26A0 Red Flags', rec.red_flags, true) +
    `<hr class="divider"><div class="rf"><div class="rl">Confidence</div>
     <div style="font-size:0.77rem;color:var(--accent2)">${conf}%</div>
     <div class="cbar"><div class="cfill" style="width:${conf}%"></div></div></div>`;
}

function fld(l, v) {
  const e=!v;
  return `<div class="rf"><div class="rl">${l}</div><div class="rv${e?' empty':''}">${esc(e?'\u2014':String(v))}</div></div>`;
}
function lf(l, a, d) {
  if (!a||!a.length) return `<div class="rf"><div class="rl">${l}</div><div class="rv empty">\u2014</div></div>`;
  return `<div class="rf"><div class="rl">${l}</div><div class="rv">${a.map(v=>`<span class="tag${d?' danger':''}">${esc(v)}</span>`).join('')}</div></div>`;
}
function esc(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}

// Initial render
renderList();

// Poll remote inference status every 10s
async function pollRemote() {
  try {
    const r = await fetch('/remote-status');
    const d = await r.json();
    const mb = document.getElementById('model-badge');
    const rb = document.getElementById('remote-badge');
    if (d.url) {
      rb.style.display = '';
      if (d.connected) {
        rb.style.borderColor = 'var(--accent2)'; rb.style.color = 'var(--accent2)';
        rb.textContent = '\uD83D\uDFE2 Gemma 4 GPU connected';
        mb.textContent = 'real model \u2022 Colab T4';
      } else {
        rb.style.borderColor = 'var(--danger)'; rb.style.color = 'var(--danger)';
        rb.textContent = '\uD83D\uDD34 remote offline \u2014 using mock';
        mb.textContent = 'mock fallback';
      }
    }
  } catch(_) {}
}
pollRemote();
setInterval(pollRemote, 10000);
</script>
</body>
</html>
"""


def _build_html() -> str:
    html = _HTML.replace("SCENARIO_DATA_PLACEHOLDER", json.dumps(SCENARIO_MENU, ensure_ascii=False))
    html = html.replace("TOTAL_PLACEHOLDER", str(TOTAL))
    return html


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index():
    return _build_html()


def _call_remote(conversation: list[dict], scenario_id: str, turn_idx: int) -> dict | None:
    """Call the Colab inference server. Returns parsed dict or None on failure."""
    import requests as _requests
    try:
        r = _requests.post(
            f"{_REMOTE_URL}/generate",
            json={"conversation": conversation, "scenario_id": scenario_id, "turn": turn_idx},
            timeout=60,
        )
        r.raise_for_status()
        raw = r.json().get("raw_response", "")
        return _parse_json_response(raw)
    except Exception as exc:
        print(f"[remote] error: {exc} — falling back to local mock")
        return None


@app.get("/remote-status")
async def remote_status():
    """UI polls this to show connection indicator."""
    if not _REMOTE_URL:
        return {"connected": False, "url": None}
    import requests as _requests
    try:
        _requests.get(f"{_REMOTE_URL}/health", timeout=5).raise_for_status()
        return {"connected": True, "url": _REMOTE_URL}
    except Exception:
        return {"connected": False, "url": _REMOTE_URL}


@app.post("/turn")
async def process_turn(request: Request):
    body = await request.json()
    session_id: str = body["session_id"]
    scenario_id: str = body["scenario_id"]
    patient_input: str = body["patient_input"]
    turn_idx: int = int(body["turn"])
    source: str = body.get("source", "synthetic")

    if session_id not in _SESSIONS:
        _SESSIONS[session_id] = {"record": IntakeRecord(), "conversation": []}
    sess = _SESSIONS[session_id]
    record: IntakeRecord = sess["record"]
    conversation: list[dict] = sess["conversation"]

    conversation.append({"role": "user", "content": patient_input})

    # ── Response strategy ──────────────────────────────────────────────────────
    # 1. Remote Colab GPU (real Gemma 4) — when --remote-url is set
    # 2. Scripted mock (original 10 scenarios only)
    # 3. Smart keyword mock (synthetic gen_* scenarios, local fallback)
    parsed = None
    if _REMOTE_URL:
        parsed = _call_remote(conversation, scenario_id, turn_idx)
    if parsed is None:
        if scenario_id in _SCRIPTED_IDS:
            assistant = _get_assistant()
            raw = assistant._model.generate(conversation, scenario_id, turn_idx)
            parsed = _parse_json_response(raw)
        else:
            parsed = _smart_response(patient_input, turn_idx)

    _apply_extraction(record, parsed.get("extracted_field"), parsed.get("extracted_value"))
    if parsed.get("red_flag") and parsed.get("red_flag_reason"):
        reason = parsed["red_flag_reason"]
        if reason not in record.red_flags:
            record.red_flags.append(reason)

    assistant_msg = parsed.get("message", "")
    conversation.append({"role": "assistant", "content": assistant_msg})
    record.overall_confidence = float(parsed.get("confidence", 0.0))

    return JSONResponse({
        "message":         assistant_msg,
        "extracted_field": parsed.get("extracted_field"),
        "extracted_value": parsed.get("extracted_value"),
        "confidence":      float(parsed.get("confidence", 0.0)),
        "red_flag":        bool(parsed.get("red_flag", False)),
        "red_flag_reason": parsed.get("red_flag_reason"),
        "intake_complete": bool(parsed.get("intake_complete", False)),
        "record":          asdict(record),
    })


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    _SESSIONS.pop(session_id, None)
    return {"ok": True}


# ---------------------------------------------------------------------------
# eval_trimodal.py compatible REST endpoints
# (mock responses — structural validation only; real accuracy needs GPU)
# ---------------------------------------------------------------------------

class _GenRequest(BaseModel):
    conversation:   list
    max_new_tokens: int   = 400
    temperature:    float = 0.3

class _ImgRequest(BaseModel):
    image_b64:    str
    text_context: str = ""

class _AudRequest(BaseModel):
    audio_b64:    str
    text_context: str = ""


def _mock_text_response(conversation: list) -> dict:
    """Return eval-compatible JSON using the smart keyword mock."""
    last_user = next(
        (m["content"] for m in reversed(conversation) if m.get("role") == "user"),
        "",
    )
    smart     = _smart_response(last_user, turn_idx=0)
    is_urgent = bool(smart.get("red_flag"))
    return {
        "response": smart["message"],
        "extracted_info": {
            "chief_complaint":   smart.get("extracted_value"),
            "symptoms":          [],
            "duration":          None,
            "severity":          None,
            "medications":       [],
            "allergies":         [],
            "urgent":            is_urgent,
            "escalation_reason": smart.get("red_flag_reason"),
            "triage_level":      "red" if is_urgent else "green",
        },
        "intake_complete": smart.get("intake_complete", False),
    }


@app.get("/health")
def health():
    return {
        "status":     "ok",
        "model":      "mock",
        "modalities": ["text", "image", "audio"],
        "version":    "mock",
    }


@app.post("/generate")
def generate_compat(req: _GenRequest):
    return JSONResponse(content=_mock_text_response(req.conversation))


@app.post("/generate_image")
def generate_image_compat(req: _ImgRequest):
    return JSONResponse(content={
        "response": "Image received. This wound shows signs consistent with normal healing.",
        "visual_findings": {
            "image_type":          "wound",
            "description":         "Mock analysis — structural test only.",
            "severity_indicators": [],
            "differential":        ["normal healing", "superficial wound"],
        },
        "extracted_info": {
            "chief_complaint":   "wound assessment",
            "symptoms":          [],
            "urgent":            False,
            "escalation_reason": None,
            "triage_level":      "green",
        },
        "intake_complete":      False,
        "follow_up_questions":  ["How long ago was this wound sustained?"],
    })


@app.post("/triage_image")
def triage_image_compat(req: _ImgRequest):
    return generate_image_compat(req)


@app.post("/generate_audio")
def generate_audio_compat(req: _AudRequest):
    return JSONResponse(content={
        "response": "Audio received. I can hear respiratory sounds.",
        "audio_analysis": {
            "audio_type":    "breathing",
            "transcription": None,
            "clinical_observations": "Mock analysis — structural test only.",
            "respiratory_findings": {
                "cough_present":      False,
                "wheeze_present":     False,
                "stridor_present":    False,
                "abnormal_breathing": False,
            },
        },
        "extracted_info": {
            "chief_complaint":   "respiratory assessment",
            "symptoms":          [],
            "urgent":            False,
            "escalation_reason": None,
            "triage_level":      "green",
        },
        "intake_complete": False,
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="CareVoice demo server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--real-model", action="store_true")
    p.add_argument("--model-id", default="google/gemma-4-7b-it")
    p.add_argument("--remote-url", default=None,
                   help="Colab/remote inference URL e.g. https://abc.trycloudflare.com")
    args = p.parse_args()

    global _ASSISTANT, _REMOTE_URL
    if args.remote_url:
        _REMOTE_URL = args.remote_url.rstrip("/")
        print(f"Remote inference URL: {_REMOTE_URL}")
    if args.real_model:
        print(f"Loading real model: {args.model_id}")
        _ASSISTANT = IntakeAssistant.load(model_id=args.model_id)
    else:
        print("Using mock model (no GPU required)")
        _ASSISTANT = IntakeAssistant.mock()

    print(f"\n  CareVoice demo: http://{args.host}:{args.port}")
    print(f"  {ORIG_COUNT} scripted + {EXT_COUNT} synthetic = {TOTAL} total scenarios\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
