"""Operator chat UI, served by the gateway itself at ``GET /chat``.

Why this exists: the point of HEARTH is that a conversation about the operator's bank
statements runs on-device. A browser tab is the natural way to have that conversation,
but every off-the-shelf chat front-end is a second origin, a CDN bundle and a font
request — three egress channels bolted onto a no-egress system. So the UI ships as one
inline HTML string served from the same origin as ``/v1/chat/completions``.

**Self-containment is the invariant, not a nicety.** The page loads no external script,
stylesheet, font, image or iframe; there is not a single absolute URL in it. That is why
no CORS middleware is added — same origin means the browser needs no cross-origin grant,
and adding one would open a hole for every other page the operator has open.
``tests/test_gateway_chat_ui.py`` asserts the served bytes contain no off-origin
reference, because "we didn't add a CDN link" is a configuration claim and the outcome
is what matters (CLAUDE.md §3).

**Auth: the page is unauthenticated and carries no credential; the operator pastes the
token once and the browser keeps it.** Two designs were on the table:

1. *Inject the token server-side into the HTML.* A browser navigating to ``/chat`` cannot
   send an ``Authorization`` header, so an injected-token page must itself be served
   unauthenticated — which turns ``GET /chat`` into an unauthenticated token-disclosure
   endpoint. Today the token's only home is a ``0600`` file, so another local user cannot
   read it; after injection, anything that can open a loopback socket can. That trades a
   filesystem permission for no permission at all, and it is the credential that gates
   every ``/v1/*`` route. Rejected.
2. *Serve static HTML, let the page hold the token.* ``/chat`` returns a document with no
   data and no secret in it. The operator pastes the contents of ``~/.hearth/token`` into
   a field once; it is stored in ``localStorage`` under the loopback origin and sent as a
   bearer header on each ``fetch``. The credential never appears in a served response, in
   a URL, or in server logs. Chosen.

Either way, ``/v1/*`` keeps its ``require_token`` dependency untouched: this module adds
a route and changes no existing one. An unauthenticated ``/chat`` is not a security
surprise — it is static markup, equivalent to ``/docs``, and it is documented here and in
``docs/API.md``. Operators who do not want it can note that it exposes no data: without a
valid token, the page can do nothing the caller could not already do with ``curl``.

The UI streams (SSE, ``[DONE]`` sentinel) because a 14B model at ~12 tok/s is unusable
if the page blocks until the last token. It surfaces the response's ``hearth`` telemetry
(``served_by``, ``backend``, ``model``) so the operator can see at a glance that the
conversation stayed on-device, and it renders ``finish_reason == "length"`` as a visible
warning — silent truncation is a bug class this repo has already fixed once in the
gateway (CLAUDE.md §3) and the UI must not reintroduce it.
"""

from __future__ import annotations

from typing import Final

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, RedirectResponse

# One string, no build step, no dependencies. Kept here rather than in a template file so
# the no-egress test can grep exactly the bytes the route serves.
CHAT_UI_HTML: Final[str] = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HEARTH — local chat</title>
<style>
  :root {
    color-scheme: light dark;
    --bg: #fbfaf8; --fg: #1b1b1d; --muted: #5c5c66; --line: #d8d5cf;
    --panel: #ffffff; --user: #eceaf3; --assistant: #ffffff;
    --ok: #17692f; --warn: #8a5300; --err: #a01b1b; --accent: #7a3b12;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #16161a; --fg: #e8e6e3; --muted: #a0a0ab; --line: #33333c;
      --panel: #1d1d23; --user: #262633; --assistant: #1d1d23;
      --ok: #6cc98a; --warn: #e0a955; --err: #f08b8b; --accent: #e0a06a;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--fg);
    font: 15px/1.55 ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto,
          "Helvetica Neue", Arial, sans-serif;
    display: flex; flex-direction: column; height: 100vh;
  }
  header {
    border-bottom: 1px solid var(--line); background: var(--panel);
    padding: 10px 14px; display: flex; flex-wrap: wrap; gap: 10px; align-items: center;
  }
  header h1 { font-size: 15px; margin: 0 8px 0 0; letter-spacing: .04em; }
  header h1 span { color: var(--accent); }
  label { font-size: 12px; color: var(--muted); display: inline-flex;
          gap: 5px; align-items: center; }
  input, select, textarea, button {
    font: inherit; color: inherit; background: var(--bg);
    border: 1px solid var(--line); border-radius: 6px; padding: 4px 7px;
  }
  input[type=number] { width: 6.5em; }
  #token { width: 15em; }
  #model { max-width: 22em; }
  button { cursor: pointer; background: var(--panel); }
  button:disabled { opacity: .5; cursor: default; }
  #log { flex: 1; overflow-y: auto; padding: 16px; display: flex;
         flex-direction: column; gap: 12px; }
  .turn { max-width: 52em; width: 100%; align-self: center; }
  .role { font-size: 11px; letter-spacing: .08em; text-transform: uppercase;
          color: var(--muted); margin-bottom: 3px; }
  .body { white-space: pre-wrap; word-wrap: break-word; border: 1px solid var(--line);
          border-radius: 8px; padding: 9px 11px; background: var(--assistant); }
  .turn.user .body { background: var(--user); }
  .meta { font-size: 12px; color: var(--muted); margin-top: 4px;
          display: flex; flex-wrap: wrap; gap: 4px 10px; }
  .meta .local { color: var(--ok); font-weight: 600; }
  .meta .remote { color: var(--warn); font-weight: 600; }
  .notice { margin-top: 5px; font-size: 13px; border-left: 3px solid var(--warn);
            padding: 3px 9px; color: var(--warn); }
  .error .body { border-color: var(--err); color: var(--err); }
  footer { border-top: 1px solid var(--line); background: var(--panel); padding: 10px 14px; }
  .composer { max-width: 52em; margin: 0 auto; display: flex; gap: 8px; }
  textarea { flex: 1; resize: vertical; min-height: 62px; background: var(--bg); }
  #status { max-width: 52em; margin: 6px auto 0; font-size: 12px; color: var(--muted); }
  #status.bad { color: var(--err); }
</style>
</head>
<body>
<header>
  <h1>HEARTH <span>local chat</span></h1>
  <label>token <input id="token" type="password" autocomplete="off"
    placeholder="paste ~/.hearth/token"></label>
  <button id="reload" type="button">Load models</button>
  <label>model <select id="model">
    <option value="auto">auto (router decides)</option>
  </select></label>
  <label>temp <input id="temp" type="number" min="0" max="2" step="0.1" value="0.7"></label>
  <label>max tokens
    <input id="maxtok" type="number" min="1" max="32768" step="1" value="512"></label>
  <button id="clear" type="button">Clear</button>
</header>

<div id="log"></div>

<footer>
  <div class="composer">
    <textarea id="input"
      placeholder="Message (Enter to send, Shift+Enter for a newline)"></textarea>
    <button id="send" type="button">Send</button>
  </div>
  <div id="status">Paste your token, then Load models.
    Nothing on this page leaves the machine.</div>
</footer>

<script>
(function () {
  "use strict";
  var TOKEN_KEY = "hearth.token";
  var el = function (id) { return document.getElementById(id); };
  var messages = [];
  var busy = false;

  function readStored() {
    try { return window.localStorage.getItem(TOKEN_KEY) || ""; } catch (e) { return ""; }
  }
  function writeStored(v) {
    try { window.localStorage.setItem(TOKEN_KEY, v); } catch (e) { /* private mode */ }
  }
  function token() { return el("token").value.trim(); }
  function headers() {
    var h = { "Content-Type": "application/json" };
    var t = token();
    if (t) { h["Authorization"] = "Bearer " + t; }
    return h;
  }
  function status(text, bad) {
    var s = el("status");
    s.textContent = text;
    s.className = bad ? "bad" : "";
  }

  // Turn any non-2xx into a sentence the operator can act on. The gateway answers with
  // an OpenAI-style envelope; fall back to the status line when it does not.
  function describe(res, text) {
    var msg = "";
    try {
      var body = JSON.parse(text);
      var d = body.detail && body.detail.error ? body.detail.error : null;
      var e = body.error || d;
      if (e && e.message) { msg = e.message; }
    } catch (err) { msg = (text || "").slice(0, 400); }
    if (res.status === 401) {
      return "401 — missing or invalid bearer token. Paste the contents of "
        + "~/.hearth/token into the token field above.";
    }
    if (res.status === 422) { return "422 — the gateway rejected the request: " + msg; }
    return res.status + " " + (res.statusText || "error") + (msg ? " — " + msg : "");
  }

  function turn(role, text, cls) {
    var wrap = document.createElement("div");
    wrap.className = "turn " + role + (cls ? " " + cls : "");
    var head = document.createElement("div");
    head.className = "role";
    head.textContent = role;
    var body = document.createElement("div");
    body.className = "body";
    body.textContent = text;
    wrap.appendChild(head);
    wrap.appendChild(body);
    el("log").appendChild(wrap);
    scroll();
    return { wrap: wrap, body: body };
  }
  function scroll() { var l = el("log"); l.scrollTop = l.scrollHeight; }

  // The whole point of a local-first tool: say what served the request, every time.
  function renderMeta(wrap, hearth, finish, maxTokens) {
    var meta = document.createElement("div");
    meta.className = "meta";
    if (hearth) {
      var where = document.createElement("span");
      where.className = hearth.served_by === "local" ? "local" : "remote";
      where.textContent = hearth.served_by === "local"
        ? "served on-device" : "served remotely";
      meta.appendChild(where);
      var model = document.createElement("span");
      model.textContent = "model: " + (hearth.model || "?");
      meta.appendChild(model);
      var backend = document.createElement("span");
      backend.textContent = "backend: " + (hearth.backend || "?");
      meta.appendChild(backend);
      if (hearth.adapter) {
        var ad = document.createElement("span");
        ad.textContent = "adapter: " + hearth.adapter;
        meta.appendChild(ad);
      }
    } else {
      var unknown = document.createElement("span");
      unknown.textContent = "no telemetry in the stream";
      meta.appendChild(unknown);
    }
    if (finish && finish !== "stop") {
      var fr = document.createElement("span");
      fr.textContent = "finish_reason: " + finish;
      meta.appendChild(fr);
    }
    wrap.appendChild(meta);
    // Truncation is never silent here: a reply cut off at the cap says so in the UI.
    if (finish === "length") {
      var note = document.createElement("div");
      note.className = "notice";
      note.textContent = "Truncated: the model hit max_tokens (" + maxTokens
        + "). Raise the limit and ask again — this reply is incomplete.";
      wrap.appendChild(note);
    }
    scroll();
  }

  function loadModels() {
    var sel = el("model");
    return fetch("/v1/models", { headers: headers() }).then(function (res) {
      return res.text().then(function (text) {
        if (!res.ok) { status("Could not list models: " + describe(res, text), true); return; }
        var data = (JSON.parse(text) || {}).data || [];
        sel.textContent = "";
        var auto = document.createElement("option");
        auto.value = "auto";
        auto.textContent = "auto (router decides)";
        sel.appendChild(auto);
        data.forEach(function (m) {
          var o = document.createElement("option");
          o.value = m.id;
          o.textContent = m.id + (m.backend ? "  [" + m.backend + "]" : "");
          sel.appendChild(o);
        });
        status("Loaded " + data.length + " model(s).", false);
      });
    }).catch(function (err) {
      status("Could not reach the gateway: " + err.message, true);
    });
  }

  function send() {
    if (busy) { return; }
    var text = el("input").value.trim();
    if (!text) { return; }
    var maxTokens = parseInt(el("maxtok").value, 10) || 512;
    var payload = {
      model: el("model").value,
      messages: messages.concat([{ role: "user", content: text }]),
      temperature: parseFloat(el("temp").value),
      max_tokens: maxTokens,
      stream: true
    };
    messages = payload.messages;
    turn("user", text);
    el("input").value = "";
    busy = true;
    el("send").disabled = true;
    status("Streaming…", false);
    var out = turn("assistant", "");
    var acc = "";
    var finish = null;
    var hearth = null;
    var errored = false;

    fetch("/v1/chat/completions", {
      method: "POST",
      headers: headers(),
      body: JSON.stringify(payload)
    }).then(function (res) {
      if (!res.ok || !res.body) {
        return res.text().then(function (t) { throw new Error(describe(res, t)); });
      }
      var reader = res.body.getReader();
      var decoder = new TextDecoder();
      var buf = "";
      var stop = false;

      function handle(payloadText) {
        if (payloadText === "[DONE]") { stop = true; return; }
        var ev;
        try { ev = JSON.parse(payloadText); } catch (err) { return; }
        if (ev.error) {
          // A mid-stream error event (budget, invalid JSON mode) must be visible, not
          // swallowed into a reply that merely looks short.
          errored = true;
          out.wrap.className += " error";
          acc += (acc ? "\\n\\n" : "") + "[" + (ev.error.type || "error") + "] "
            + (ev.error.message || "");
          out.body.textContent = acc;
          return;
        }
        if (ev.hearth) { hearth = ev.hearth; }
        var choice = (ev.choices || [])[0];
        if (!choice) { return; }
        if (choice.finish_reason) { finish = choice.finish_reason; }
        if (choice.delta && choice.delta.content) {
          acc += choice.delta.content;
          out.body.textContent = acc;
          scroll();
        }
      }

      function pump() {
        return reader.read().then(function (chunk) {
          if (chunk.done || stop) { return; }
          buf += decoder.decode(chunk.value, { stream: true });
          var sep = buf.indexOf("\\n\\n");
          while (sep >= 0) {
            var block = buf.slice(0, sep);
            buf = buf.slice(sep + 2);
            var lines = block.split("\\n");
            for (var i = 0; i < lines.length; i++) {
              if (lines[i].indexOf("data: ") === 0) { handle(lines[i].slice(6)); }
            }
            if (stop) { return; }
            sep = buf.indexOf("\\n\\n");
          }
          return pump();
        });
      }
      return pump();
    }).then(function () {
      if (acc && !errored) { messages.push({ role: "assistant", content: acc }); }
      renderMeta(out.wrap, hearth, finish, maxTokens);
      if (errored) {
        status("The gateway reported an error mid-stream; the reply above is incomplete.",
          true);
        return;
      }
      var cut = finish === "length" ? " Truncated at max_tokens." : "";
      status((hearth && hearth.served_by === "local"
        ? "Done — served on-device by " + hearth.model + "."
        : "Done.") + cut, finish === "length");
    }).catch(function (err) {
      out.wrap.className += " error";
      out.body.textContent = acc ? acc + "\\n\\n[stream failed] " + err.message : err.message;
      status(err.message, true);
    }).then(function () {
      busy = false;
      el("send").disabled = false;
      el("input").focus();
    });
  }

  el("token").value = readStored();
  el("token").addEventListener("change", function () { writeStored(token()); });
  el("reload").addEventListener("click", function () { writeStored(token()); loadModels(); });
  el("send").addEventListener("click", send);
  el("clear").addEventListener("click", function () {
    messages = [];
    el("log").textContent = "";
    status("Conversation cleared.", false);
  });
  el("input").addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
  });
  if (readStored()) { loadModels(); }
})();
</script>
</body>
</html>
"""


def chat_ui_html() -> str:
    """Return the served page. A function so tests assert on what the route emits."""
    return CHAT_UI_HTML


def register_chat_ui(app: FastAPI) -> None:
    """Mount ``GET /chat`` on ``app``.

    Deliberately *not* behind :func:`~hearth.gateway.auth.require_token`: a browser
    navigation cannot carry an ``Authorization`` header, and the document holds no data
    and no credential (see the module docstring). Every ``/v1/*`` route keeps its auth
    dependency, so an unauthenticated visitor gets markup and nothing else.
    """

    @app.get("/chat", response_class=HTMLResponse)
    def chat_ui() -> HTMLResponse:
        """Self-contained local chat UI (no external requests, no injected token)."""
        return HTMLResponse(content=chat_ui_html())

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        """Send the bare origin to the UI.

        Somebody who starts the daemon and opens http://127.0.0.1:8080 means the chat page;
        a 404 there reads as "the server is broken" rather than "you want /chat", and that is
        exactly what happened the first time this shipped. 307 rather than 301 so a browser
        never caches the redirect and a future root route is not shadowed by history.
        """
        return RedirectResponse(url="/chat", status_code=307)
