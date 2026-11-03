"""Creator Invoice Agent
Type a plain-English invoice request. An AI model turns it into a structured
invoice, PayPal's Invoicing API creates and sends it, and the same AI later
writes polite payment reminders that PayPal delivers.

Runs against the PayPal SANDBOX by default (no real money).
"""
import datetime
import json
import os
import re
import time

import requests
from flask import Flask, Response, jsonify, request

app = Flask(__name__)

PAYPAL_BASE = os.getenv("PAYPAL_BASE", "https://api-m.sandbox.paypal.com")
PAYPAL_CLIENT_ID = os.getenv("PAYPAL_CLIENT_ID", "")
PAYPAL_CLIENT_SECRET = os.getenv("PAYPAL_CLIENT_SECRET", "")
INVOICER_EMAIL = os.getenv("PAYPAL_BUSINESS_EMAIL", "")  # your sandbox business account
INVOICER_NAME = os.getenv("INVOICER_NAME", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

_token = {"value": None, "expires": 0.0}


class AppError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.message = message
        self.status = status


@app.errorhandler(AppError)
def handle_app_error(err):
    return jsonify({"error": err.message}), err.status


# ----------------------------------------------------------------- PayPal ---
def paypal_token():
    if not PAYPAL_CLIENT_ID or not PAYPAL_CLIENT_SECRET:
        raise AppError("PayPal keys are missing. Set PAYPAL_CLIENT_ID and PAYPAL_CLIENT_SECRET.", 500)
    if _token["value"] and time.time() < _token["expires"] - 60:
        return _token["value"]
    r = requests.post(
        f"{PAYPAL_BASE}/v1/oauth2/token",
        auth=(PAYPAL_CLIENT_ID, PAYPAL_CLIENT_SECRET),
        data={"grant_type": "client_credentials"},
        timeout=30,
    )
    if r.status_code != 200:
        raise AppError("PayPal rejected the Client ID/Secret. Check they are the Sandbox pair.", 502)
    data = r.json()
    _token["value"] = data["access_token"]
    _token["expires"] = time.time() + int(data.get("expires_in", 300))
    return _token["value"]


def paypal(method, path, **kwargs):
    headers = {"Authorization": f"Bearer {paypal_token()}", "Content-Type": "application/json"}
    r = requests.request(method, f"{PAYPAL_BASE}{path}", headers=headers, timeout=30, **kwargs)
    if r.status_code >= 400:
        detail = ""
        try:
            body = r.json()
            detail = body.get("message") or ""
            issues = body.get("details") or []
            if issues:
                detail += " " + "; ".join(i.get("description") or i.get("issue", "") for i in issues)
        except ValueError:
            detail = r.text[:200]
        raise AppError(f"PayPal error {r.status_code}: {detail.strip()}", 502)
    return r


# --------------------------------------------------------------------- AI ---
def gemini(prompt, as_json=False):
    if not GEMINI_API_KEY:
        raise AppError("AI key is missing. Set GEMINI_API_KEY.", 500)
    body = {"contents": [{"parts": [{"text": prompt}]}]}
    if as_json:
        body["generationConfig"] = {"responseMimeType": "application/json"}
    # Try the chosen model first, then backups. Retry when Google is busy (429/500/503).
    candidates = [GEMINI_MODEL, "gemini-flash-latest", "gemini-flash-lite-latest", "gemini-2.5-flash", "gemini-2.5-flash-lite"]
    models = [m for i, m in enumerate(candidates) if m and m not in candidates[:i]]
    last_status = None
    for model in models:
        for attempt in range(3):
            try:
                r = requests.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                    params={"key": GEMINI_API_KEY},
                    json=body,
                    timeout=60,
                )
            except requests.RequestException:
                last_status = "network"
                time.sleep(1.5)
                continue
            last_status = r.status_code
            if r.status_code == 200:
                break
            if r.status_code in (429, 500, 503):
                time.sleep(1.5 * (attempt + 1))
                continue
            break  # 400/403/404: retrying will not help; try the next model
        if last_status == 200:
            break
    if last_status != 200:
        if last_status in (400, 403):
            raise AppError("AI key was rejected. Check GEMINI_API_KEY.", 502)
        if last_status in (429, 500, 503, "network"):
            raise AppError("The AI service is busy right now. Wait a few seconds and try again.", 503)
        raise AppError(f"AI service error {last_status}. Check GEMINI_MODEL.", 502)
    try:
        text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, ValueError):
        raise AppError("AI returned an empty answer. Try again.", 502)
    if as_json:
        text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
        try:
            return json.loads(text)
        except ValueError:
            raise AppError("AI answer was not valid JSON. Try rephrasing.", 502)
    return text.strip()


PARSE_PROMPT = """You turn a freelancer's plain-English message into an invoice.
Today is {today}. Return ONLY JSON in this exact shape:
{{"client_name": string, "client_email": string or null, "currency": 3-letter code,
 "due_days": integer, "note": string or null,
 "items": [{{"name": string, "quantity": number, "unit_price": number}}]}}
Rules: default currency USD, default due_days 7, default quantity 1.
unit_price is the price of ONE unit. Never invent an email address; use null if none is given.
If the message is not a request to invoice someone, return {{"error": "short reason"}}.

Message: {message}"""


def clean_invoice(raw):
    if not isinstance(raw, dict):
        raise AppError("Could not read that as an invoice.")
    if raw.get("error"):
        raise AppError(str(raw["error"]))
    name = str(raw.get("client_name") or "").strip()
    if not name:
        raise AppError("Who is the invoice for? Add the client's name.")
    items = []
    for it in raw.get("items") or []:
        try:
            qty = float(it.get("quantity") or 1)
            price = float(it.get("unit_price"))
        except (TypeError, ValueError):
            continue
        label = str(it.get("name") or "").strip()
        if label and qty > 0 and price > 0:
            items.append({"name": label[:200], "quantity": qty, "unit_price": round(price, 2)})
    if not items:
        raise AppError("I could not find an item with a price. Say what the work was and the amount.")
    email = raw.get("client_email")
    email = email.strip() if isinstance(email, str) and "@" in email else None
    currency = str(raw.get("currency") or "USD").upper()[:3]
    try:
        due_days = max(0, min(int(raw.get("due_days") or 7), 365))
    except (TypeError, ValueError):
        due_days = 7
    note = raw.get("note")
    return {
        "client_name": name[:100],
        "client_email": email,
        "currency": currency,
        "due_days": due_days,
        "note": str(note)[:500] if note else None,
        "items": items,
    }


def fmt_qty(q):
    return str(int(q)) if float(q).is_integer() else str(q)


def build_paypal_invoice(inv):
    today = datetime.date.today()
    due = today + datetime.timedelta(days=inv["due_days"])
    recipient = {"billing_info": {"name": {"business_name": inv["client_name"]}}}
    if inv["client_email"]:
        recipient["billing_info"]["email_address"] = inv["client_email"]
    body = {
        "detail": {
            "invoice_date": today.isoformat(),
            "currency_code": inv["currency"],
            "payment_term": {"term_type": "DUE_ON_DATE_SPECIFIED", "due_date": due.isoformat()},
        },
        "primary_recipients": [recipient],
        "items": [
            {
                "name": it["name"],
                "quantity": fmt_qty(it["quantity"]),
                "unit_amount": {"currency_code": inv["currency"], "value": f"{it['unit_price']:.2f}"},
            }
            for it in inv["items"]
        ],
    }
    if inv["note"]:
        body["detail"]["note"] = inv["note"]
    invoicer = {}
    if INVOICER_EMAIL:
        invoicer["email_address"] = INVOICER_EMAIL
    if INVOICER_NAME:
        invoicer["name"] = {"business_name": INVOICER_NAME}
    if invoicer:
        body["invoicer"] = invoicer
    return body


# ----------------------------------------------------------------- routes ---
@app.post("/api/parse")
def api_parse():
    message = ((request.get_json(silent=True) or {}).get("message") or "").strip()
    if len(message) < 5:
        raise AppError("Type what you want to invoice, for example: Invoice Acme $300 for a thread.")
    raw = gemini(PARSE_PROMPT.format(today=datetime.date.today().isoformat(), message=message[:1000]), as_json=True)
    return jsonify({"invoice": clean_invoice(raw)})


@app.post("/api/send")
def api_send():
    inv = clean_invoice((request.get_json(silent=True) or {}).get("invoice"))
    if not inv["client_email"]:
        raise AppError("Add the client's email so PayPal knows where to send the invoice.")
    created = paypal("POST", "/v2/invoicing/invoices", json=build_paypal_invoice(inv)).json()
    invoice_id = created["href"].rstrip("/").split("/")[-1]
    sent = paypal("POST", f"/v2/invoicing/invoices/{invoice_id}/send", json={"send_to_recipient": True})
    link = ""
    try:
        link = sent.json().get("href", "")
    except ValueError:
        pass
    return jsonify({"id": invoice_id, "link": link})


def summarize(item):
    detail = item.get("detail", {})
    recipients = item.get("primary_recipients") or [{}]
    billing = recipients[0].get("billing_info", {})
    name = billing.get("name", {})
    client = name.get("business_name") or " ".join(
        p for p in (name.get("given_name"), name.get("surname")) if p
    ) or billing.get("email_address", "Unknown")
    amount = item.get("amount") or {}
    return {
        "id": item.get("id"),
        "number": detail.get("invoice_number", ""),
        "client": client,
        "amount": amount.get("value", ""),
        "currency": amount.get("currency_code") or detail.get("currency_code", ""),
        "status": item.get("status", ""),
        "due": (detail.get("payment_term") or {}).get("due_date", ""),
    }


@app.get("/api/invoices")
def api_invoices():
    data = paypal("GET", "/v2/invoicing/invoices", params={"page": 1, "page_size": 20, "total_required": "true"}).json()
    return jsonify({"invoices": [summarize(i) for i in data.get("items", [])]})


@app.post("/api/remind")
def api_remind():
    invoice_id = ((request.get_json(silent=True) or {}).get("id") or "").strip()
    if not re.fullmatch(r"INV2-[A-Z0-9-]+", invoice_id):
        raise AppError("That does not look like a PayPal invoice ID.")
    item = paypal("GET", f"/v2/invoicing/invoices/{invoice_id}").json()
    info = summarize(item)
    if info["status"] in ("PAID", "MARKED_AS_PAID", "CANCELLED", "REFUNDED", "DRAFT"):
        raise AppError(f"This invoice is {info['status']}; no reminder needed.")
    today = datetime.date.today().isoformat()
    note = gemini(
        "Write a short, warm, professional payment reminder (max 70 words) from a freelance Web3 content "
        f"creator to a client. Client: {info['client']}. Invoice {info['number']} for {info['amount']} "
        f"{info['currency']}, due {info['due'] or 'soon'}. Today is {today}. If the due date has passed, be "
        "politely firm; if not, be friendly. Plain text only, no subject line, no placeholders."
    )
    paypal(
        "POST",
        f"/v2/invoicing/invoices/{invoice_id}/remind",
        json={"subject": f"Friendly reminder: invoice {info['number']}", "note": note[:3900], "send_to_invoicer": True},
    )
    return jsonify({"message": note})


@app.get("/healthz")
def healthz():
    return jsonify({"ok": True, "paypal_keys": bool(PAYPAL_CLIENT_ID and PAYPAL_CLIENT_SECRET), "ai_key": bool(GEMINI_API_KEY)})


@app.get("/")
def index():
    return Response(INDEX_HTML, mimetype="text/html")


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Creator Invoice Agent</title>
<style>
:root{--bg:#eef2ef;--ink:#10231c;--mute:#52665e;--line:#cfd9d4;--teal:#0f766e;--paper:#fff;--warn:#b45309}
@media (prefers-color-scheme:dark){:root{--bg:#0e1714;--ink:#e6efeb;--mute:#93a8a0;--line:#26372f;--teal:#2dd4bf;--paper:#16231e;--warn:#fbbf24}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:640px;margin:0 auto;padding:20px 16px 56px}
h1{font:700 1.7rem/1.2 Georgia,serif;margin:8px 0 4px}
.lead{color:var(--mute);margin:0 0 20px}
textarea{width:100%;min-height:96px;padding:12px;border:1px solid var(--line);border-radius:10px;background:var(--paper);color:var(--ink);font:inherit;resize:vertical}
button{font:inherit;font-weight:600;border:0;border-radius:10px;padding:11px 16px;background:var(--teal);color:#fff;cursor:pointer}
@media (prefers-color-scheme:dark){button{color:#052e2a}}
button.ghost{background:transparent;color:var(--teal);border:1px solid var(--line);padding:6px 10px;font-size:.9rem}
button:disabled{opacity:.55;cursor:wait}
button:focus-visible,textarea:focus-visible,input:focus-visible{outline:3px solid var(--teal);outline-offset:2px}
.row{display:flex;gap:10px;align-items:center;margin:12px 0 0;flex-wrap:wrap}
.msg{margin:12px 0 0;min-height:1.4em}.err{color:var(--warn)}
.slip{position:relative;margin:20px 0;background:var(--paper);padding:20px 18px 22px;border-radius:6px;
 -webkit-mask:radial-gradient(circle 7px at 7px 100%,#0000 98%,#000) -7px 0/14px 100%;
 mask:radial-gradient(circle 7px at 7px 100%,#0000 98%,#000) -7px 0/14px 100%;
 box-shadow:0 0 0 1px var(--line)}
.slip h2{font:700 1.15rem Georgia,serif;margin:0}
.slip .to{color:var(--mute);margin:2px 0 14px}
.line{display:flex;justify-content:space-between;gap:12px;padding:7px 0;border-bottom:1px dashed var(--line)}
.total{display:flex;justify-content:space-between;font:700 1.25rem Georgia,serif;padding-top:12px}
.slip label{display:block;margin:14px 0 4px;color:var(--mute);font-size:.9rem}
.slip input{width:100%;padding:9px 10px;border:1px solid var(--line);border-radius:8px;background:transparent;color:var(--ink);font:inherit}
h3{font:700 1.1rem Georgia,serif;margin:32px 0 8px}
.inv{display:grid;grid-template-columns:1fr auto;gap:2px 12px;padding:12px 0;border-bottom:1px solid var(--line)}
.inv small{color:var(--mute)}
.st{font-weight:700;font-size:.85rem}.st.PAID,.st.MARKED_AS_PAID{color:var(--teal)}.st.SENT,.st.UNPAID,.st.PARTIALLY_PAID,.st.PAYMENT_PENDING{color:var(--warn)}
.reminder{grid-column:1/-1;margin-top:6px;padding:8px 10px;border-left:3px solid var(--teal);color:var(--mute);font-size:.92rem}
a{color:var(--teal)}
</style>
</head>
<body>
<main>
<h1>Creator Invoice Agent</h1>
<p class="lead">Describe the job in your own words. The agent builds the invoice, PayPal sends it, and the agent chases late payments for you.</p>

<label for="msg" class="lead">What do you want to invoice?</label>
<textarea id="msg" placeholder="Invoice Bybit Africa $300 for an X thread and $150 for a short video script, due in 7 days, billing@example.com"></textarea>
<div class="row"><button id="parse">Build invoice</button></div>
<div class="msg" id="status" role="status" aria-live="polite"></div>

<section id="slipWrap" hidden>
  <div class="slip" id="slip"></div>
  <div class="row"><button id="send">Send with PayPal</button></div>
</section>

<h3>Your invoices</h3>
<div class="row"><button class="ghost" id="refresh">Refresh from PayPal</button></div>
<div id="list" aria-live="polite"></div>
</main>
<script>
const $ = (id) => document.getElementById(id);
let current = null;

function say(text, isErr) { const s = $("status"); s.textContent = text || ""; s.className = "msg" + (isErr ? " err" : ""); }
function money(v, c) { try { return new Intl.NumberFormat(undefined, {style:"currency", currency:c}).format(v); } catch (e) { return v + " " + c; } }
function el(tag, cls, text) { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; }

async function api(path, body) {
  const opts = body ? {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)} : {};
  const r = await fetch(path, opts);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.error || "Something went wrong.");
  return data;
}

function drawSlip(inv) {
  const slip = $("slip"); slip.textContent = "";
  slip.append(el("h2", "", "Invoice draft"));
  slip.append(el("div", "to", "For " + inv.client_name + " · due in " + inv.due_days + " days"));
  let total = 0;
  inv.items.forEach((it) => {
    const amt = it.quantity * it.unit_price; total += amt;
    const row = el("div", "line");
    row.append(el("span", "", it.name + (it.quantity !== 1 ? " × " + it.quantity : "")), el("span", "", money(amt, inv.currency)));
    slip.append(row);
  });
  const t = el("div", "total"); t.append(el("span", "", "Total"), el("span", "", money(total, inv.currency))); slip.append(t);
  const lab = el("label", "", "Client email (PayPal sends the invoice here)"); lab.setAttribute("for", "email");
  const input = el("input"); input.id = "email"; input.type = "email"; input.value = inv.client_email || "";
  input.addEventListener("input", () => { current.client_email = input.value.trim() || null; });
  slip.append(lab, input);
  $("slipWrap").hidden = false;
}

$("parse").addEventListener("click", async () => {
  const b = $("parse"); b.disabled = true; say("Reading your message…");
  try { const d = await api("/api/parse", {message: $("msg").value}); current = d.invoice; drawSlip(current); say("Check the details, then send."); }
  catch (e) { $("slipWrap").hidden = true; say(e.message, true); }
  b.disabled = false;
});

$("send").addEventListener("click", async () => {
  const b = $("send"); b.disabled = true; say("Creating the invoice on PayPal…");
  try {
    const d = await api("/api/send", {invoice: current});
    say("Sent. Invoice " + d.id + " is with PayPal."); $("slipWrap").hidden = true; $("msg").value = ""; loadList();
  } catch (e) { say(e.message, true); }
  b.disabled = false;
});

async function loadList() {
  const box = $("list"); box.textContent = "Loading…";
  try {
    const d = await api("/api/invoices"); box.textContent = "";
    if (!d.invoices.length) { box.append(el("p", "lead", "No invoices yet. Build your first one above.")); return; }
    d.invoices.forEach((i) => {
      const row = el("div", "inv");
      row.append(el("strong", "", i.client), el("span", "st " + i.status, i.status.replace(/_/g, " ")));
      row.append(el("small", "", (i.number || i.id) + (i.due ? " · due " + i.due : "")), el("small", "", i.amount ? money(i.amount, i.currency) : ""));
      if (["SENT","UNPAID","PARTIALLY_PAID","PAYMENT_PENDING"].includes(i.status)) {
        const b = el("button", "ghost", "Chase with AI"); b.addEventListener("click", () => chase(i.id, b, row)); row.append(b);
      }
      box.append(row);
    });
  } catch (e) { box.textContent = ""; box.append(el("p", "err", e.message)); }
}

async function chase(id, btn, row) {
  btn.disabled = true; btn.textContent = "Writing…";
  try { const d = await api("/api/remind", {id}); row.append(el("div", "reminder", "Reminder sent by PayPal: " + d.message)); btn.remove(); }
  catch (e) { btn.disabled = false; btn.textContent = "Chase with AI"; say(e.message, true); }
}

$("refresh").addEventListener("click", loadList);
loadList();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
