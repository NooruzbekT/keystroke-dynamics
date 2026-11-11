"use strict";

const FORBIDDEN = {
  Backspace: "Использован Backspace — исправления запрещены. Начните фразу заново.",
  Delete: "Использован Delete — исправления запрещены. Начните фразу заново.",
  Shift: "Нажат Shift — фраза вводится строчными буквами без модификаторов.",
  Control: "Нажат Ctrl — модификаторы запрещены.",
  Alt: "Нажат Alt — модификаторы запрещены.",
  AltGraph: "Нажат AltGr — модификаторы запрещены.",
  Meta: "Нажата клавиша Win/Cmd — модификаторы запрещены.",
  CapsLock: "Нажат CapsLock — фраза вводится строчными буквами.",
  ArrowLeft: "Курсор перемещён стрелкой — ввод должен идти подряд.",
  ArrowRight: "Курсор перемещён стрелкой — ввод должен идти подряд.",
  ArrowUp: "Курсор перемещён стрелкой — ввод должен идти подряд.",
  ArrowDown: "Курсор перемещён стрелкой — ввод должен идти подряд.",
  Home: "Нажат Home — ввод должен идти подряд.",
  End: "Нажат End — ввод должен идти подряд.",
};
const BLOCKED_INPUT_TYPES = /^(insertFromPaste|insertFromDrop|insertFromYank|insertReplacementText|insertTranspose|delete|historyUndo|historyRedo)/;

const $ = (id) => document.getElementById(id);
const input = $("typing");

let phrase = "typing rhythm";
let profile = null;
let mode = "enroll";
let busy = false;

const cap = { events: [], typed: 0, pressed: new Set(), stale: new Set() };

function announce(text) {
  $("sr-status").textContent = text;
}

function say(text, kind = "info") {
  const el = $("message");
  el.textContent = text;
  el.className = `message ${kind}`;
}

function renderPhrase() {
  const el = $("phrase");
  el.textContent = "";
  [...phrase].forEach((ch, i) => {
    const span = document.createElement("span");
    span.textContent = ch === " " ? " " : ch;
    if (ch === " ") span.classList.add("space");
    if (i < cap.typed) span.classList.add("done");
    else if (i === cap.typed) span.classList.add("next");
    el.appendChild(span);
  });
}

function resetCapture() {
  cap.stale = new Set([...cap.stale, ...cap.pressed]);
  cap.events = [];
  cap.typed = 0;
  cap.pressed.clear();
  input.value = "";
  renderPhrase();
}

function reject(text) {
  resetCapture();
  say(text, "error");
}

function started() {
  return cap.events.length > 0;
}

input.addEventListener("keydown", (e) => {
  if (busy) { e.preventDefault(); return; }
  if (e.key === "Enter") { e.preventDefault(); return; }
  // A new keydown means the stale press was released, possibly outside the field; repeats are rejected below.
  cap.stale.delete(e.code);
  // Tab and Esc keep their default behaviour so keyboard users can leave the field.
  if (e.key === "Tab" || e.key === "Escape") {
    if (started()) reject(e.key === "Tab" ? "Нажат Tab — попытка сброшена." : "Нажат Esc — попытка сброшена.");
    return;
  }
  const deny = (text) => {
    e.preventDefault();
    reject(text);
    // The rejected key is still held; its keyup must not break the next attempt.
    cap.stale.add(e.code);
  };
  if (e.isComposing || e.keyCode === 229) { deny("Включён метод ввода (IME). Переключите раскладку на английскую."); return; }
  if (e.repeat) { deny("Обнаружен автоповтор удерживаемой клавиши. Начните фразу заново."); return; }
  if (FORBIDDEN[e.key]) { deny(FORBIDDEN[e.key]); return; }
  if (e.ctrlKey || e.altKey || e.metaKey) { deny("Нажаты модификаторы — ввод прерван."); return; }
  if (e.getModifierState && e.getModifierState("CapsLock")) { deny("Включён CapsLock — выключите его и начните заново."); return; }
  if (e.key.length !== 1) { deny(`Недопустимая клавиша «${e.key}».`); return; }
  if (cap.typed >= phrase.length) { deny("Введён лишний символ после конца фразы."); return; }
  const expected = phrase[cap.typed];
  if (e.key !== expected) {
    deny(`Ожидался символ «${expected === " " ? "пробел" : expected}», введён «${e.key === " " ? "пробел" : e.key}». Проверьте фразу и раскладку.`);
    return;
  }
  if (cap.pressed.has(e.code)) { deny("Обнаружен автоповтор удерживаемой клавиши."); return; }
  if (!started()) say("");
  cap.events.push({ type: "down", key: e.key, code: e.code, t: performance.now() });
  cap.pressed.add(e.code);
  cap.typed += 1;
  renderPhrase();
});

input.addEventListener("keyup", (e) => {
  if (busy || e.key === "Enter") return;
  if (!cap.pressed.has(e.code)) {
    if (cap.stale.has(e.code)) { cap.stale.delete(e.code); return; }
    if (started()) reject("Отпускание клавиши без нажатия — попытка повреждена.");
    return;
  }
  cap.events.push({ type: "up", key: e.key, code: e.code, t: performance.now() });
  cap.pressed.delete(e.code);
  if (cap.typed === phrase.length && cap.pressed.size === 0) {
    if (input.value !== phrase) { reject("Текст в поле не совпадает с фразой."); return; }
    submit();
  }
});

input.addEventListener("input", () => {
  if (!phrase.startsWith(input.value)) reject("Текст перестал совпадать с фразой.");
});
input.addEventListener("beforeinput", (e) => {
  if (BLOCKED_INPUT_TYPES.test(e.inputType)) { e.preventDefault(); reject("Вставка и правка текста запрещены — наберите фразу вручную."); }
});
for (const type of ["paste", "drop", "cut"]) {
  input.addEventListener(type, (e) => { e.preventDefault(); reject("Вставка и правка текста запрещены — наберите фразу вручную."); });
}
input.addEventListener("compositionstart", () => reject("Включён метод ввода (IME). Переключите раскладку на английскую."));
input.addEventListener("blur", () => { if (started() && !busy) reject("Поле потеряло фокус — попытка сброшена."); });
input.addEventListener("mousedown", (e) => { if (started()) { e.preventDefault(); reject("Клик мышью во время ввода — попытка сброшена."); } });
input.addEventListener("select", () => { if (started() && input.selectionStart !== input.value.length) reject("Курсор перемещён — ввод должен идти подряд."); });

async function api(path, body) {
  const res = await fetch(path, {
    method: body === undefined ? "GET" : "POST",
    headers: body === undefined ? {} : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  let data = null;
  try { data = await res.json(); } catch { data = null; }
  if (!res.ok) {
    const err = new Error((data && data.message_ru) || "Ошибка обработки на сервере.");
    err.code = data && data.error_code;
    throw err;
  }
  return data;
}

async function submit() {
  const events = cap.events.slice();
  const maxSeconds = profile ? profile.max_attempt_seconds : 30;
  if (events[events.length - 1].t - events[0].t > maxSeconds * 1000) { reject(`Попытка длилась дольше ${maxSeconds} с — начните заново.`); return; }
  busy = true;
  input.disabled = true;
  try {
    if (mode === "enroll") {
      const r = await api("/api/enroll/attempt", { events });
      await refreshProfile();
      const nTrain = profile ? profile.n_train_needed : r.n_train;
      const nCalib = profile ? profile.n_calib_needed : r.n_calib;
      say(r.role === "train" ? `Ввод принят: обучение ${r.n_train}/${nTrain}.` : `Ввод принят: настройка порога ${r.n_calib}/${nCalib}.`);
    } else if (mode === "verify") {
      const r = await api("/api/verify", { events });
      showVerdict(r);
      say("");
      announce(`${verdictText(r.accepted)}. ${numbers(r)}`);
    } else {
      const condition = document.querySelector('input[name="condition"]:checked').value;
      const r = await api("/api/experiment/attempt", { events, condition });
      renderCounts(r.counts, r.per_condition);
      $("experiment-last").textContent = `${verdictText(r.accepted)}. ${numbers(r)}`;
      say("Попытка записана в журнал эксперимента.");
    }
  } catch (err) {
    say(err.message, "error");
    if (err.code === "no_profile") await refreshProfile();
  } finally {
    busy = false;
    input.disabled = false;
    resetCapture();
    input.focus();
  }
}

const fmt = (v, d = 3) => Number(v).toFixed(d);
const verdictText = (ok) => (ok ? "Похож на владельца профиля" : "Ритм отличается");
const numbers = (r) => `Оценка отклонения: ${fmt(r.score)}, порог: ${fmt(r.threshold)}. Длительность ввода: ${fmt(r.duration, 2)} с.`;

function keyLabel(token) {
  const k = token.slice(token.indexOf("_") + 1);
  return k === "space" ? "␣" : k;
}

function showVerdict(r) {
  const box = $("verdict");
  box.hidden = false;
  box.className = `verdict ${r.accepted ? "ok" : "bad"}`;
  $("verdict-text").textContent = verdictText(r.accepted);
  $("verdict-numbers").textContent = numbers(r);
  $("verify-empty").hidden = true;
  $("charts").hidden = false;

  const names = r.features.names;
  const pick = (idx, arr) => idx.map((i) => arr[i]);
  const dd = r.groups.DD, h = r.groups.H;
  const ddLabels = dd.map((i) => { const [, a, b] = names[i].split("."); return `${keyLabel(a)}→${keyLabel(b)}`; });
  const hLabels = h.map((i) => keyLabel(names[i].split(".")[1]));
  drawChart($("chart-dd"), ddLabels, pick(dd, r.features.values), pick(dd, r.profile_mean));
  drawChart($("chart-h"), hLabels, pick(h, r.features.values), pick(h, r.profile_mean));

  const table = $("values-table");
  table.replaceChildren();
  const head = document.createElement("thead");
  const headRow = document.createElement("tr");
  for (const title of ["Признак", "Текущий ввод, с", "Профиль, с"]) {
    const th = document.createElement("th");
    th.textContent = title;
    headRow.appendChild(th);
  }
  head.appendChild(headRow);
  table.appendChild(head);
  const body = document.createElement("tbody");
  names.forEach((n, i) => {
    const tr = document.createElement("tr");
    for (const v of [n, fmt(r.features.values[i]), fmt(r.profile_mean[i])]) {
      const td = document.createElement("td");
      td.textContent = v;
      tr.appendChild(td);
    }
    body.appendChild(tr);
  });
  table.appendChild(body);
}

const SVG_NS = "http://www.w3.org/2000/svg";
function el(name, attrs, text) {
  const node = document.createElementNS(SVG_NS, name);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (text !== undefined) node.textContent = text;
  return node;
}

function drawChart(svg, labels, current, mean) {
  const W = 760, H = 250, m = { l: 48, r: 16, t: 28, b: 36 };
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  [...svg.querySelectorAll(":scope > :not(desc)")].forEach((n) => n.remove());
  const all = current.concat(mean);
  const lo = Math.min(0, ...all), hi = Math.max(...all) * 1.1 || 1;
  const x = (i) => m.l + (labels.length === 1 ? 0 : (i * (W - m.l - m.r)) / (labels.length - 1));
  const y = (v) => m.t + ((hi - v) * (H - m.t - m.b)) / (hi - lo);

  for (let k = 0; k <= 4; k++) {
    const v = lo + ((hi - lo) * k) / 4;
    svg.appendChild(el("line", { class: "grid", x1: m.l, x2: W - m.r, y1: y(v), y2: y(v) }));
    svg.appendChild(el("text", { x: m.l - 6, y: y(v) + 4, "text-anchor": "end" }, fmt(v, 2)));
  }
  if (lo < 0) svg.appendChild(el("line", { class: "axis", x1: m.l, x2: W - m.r, y1: y(0), y2: y(0) }));
  labels.forEach((lab, i) => svg.appendChild(el("text", { x: x(i), y: H - m.b + 18, "text-anchor": "middle" }, lab)));

  const series = [
    { values: mean, line: "profile", dot: "profile-dot", name: "Средний ритм профиля" },
    { values: current, line: "current", dot: "current-dot", name: "Текущий ввод" },
  ];
  for (const s of series) {
    svg.appendChild(el("polyline", { class: s.line, points: s.values.map((v, i) => `${x(i)},${y(v)}`).join(" ") }));
    s.values.forEach((v, i) => {
      const c = el("circle", { class: s.dot, cx: x(i), cy: y(v), r: 4.5 });
      c.appendChild(el("title", {}, `${s.name}, ${labels[i]}: ${fmt(v)} с`));
      svg.appendChild(c);
    });
  }
  const legend = el("g", { class: "legend" });
  legend.appendChild(el("line", { class: "current", x1: m.l, x2: m.l + 24, y1: 12, y2: 12 }));
  legend.appendChild(el("text", { x: m.l + 30, y: 16 }, "Текущий ввод"));
  legend.appendChild(el("line", { class: "profile", x1: m.l + 150, x2: m.l + 174, y1: 12, y2: 12 }));
  legend.appendChild(el("text", { x: m.l + 180, y: 16 }, "Средний ритм профиля"));
  svg.appendChild(legend);
}

function renderCounts(counts, perCondition) {
  for (const [cond, n] of Object.entries(counts)) {
    const node = document.querySelector(`[data-count="${cond}"]`);
    if (node) node.textContent = `${n}/${perCondition}`;
  }
}

function setMode(next) {
  mode = next;
  for (const name of ["enroll", "verify", "experiment"]) {
    const selected = name === next;
    $(`tab-${name}`).setAttribute("aria-selected", String(selected));
    $(`panel-${name}`).hidden = !selected;
  }
  resetCapture();
  say("");
  input.disabled = next === "enroll" && profile && profile.status === "ready";
  if (next === "experiment") api("/api/experiment").then((r) => renderCounts(r.counts, r.per_condition)).catch(() => {});
}

async function refreshProfile() {
  try {
    profile = await api("/api/profile");
  } catch (err) {
    $("profile-status").textContent = err.message;
    return;
  }
  phrase = profile.phrase;
  $("phrase-text").textContent = `«${phrase}»`;
  renderPhrase();
  const ready = profile.status === "ready";
  $("tab-verify").disabled = !ready;
  $("tab-experiment").disabled = !ready;
  $("enroll-collect").hidden = ready;
  $("enroll-ready").hidden = !ready;

  if (ready) {
    $("profile-status").textContent = `Профиль готов: метод ${profile.method}, порог ${fmt(profile.threshold)}.`;
  } else {
    const nt = profile.n_train || 0, nc = profile.n_calib || 0;
    $("progress-train").max = profile.n_train_needed;
    $("progress-calib").max = profile.n_calib_needed;
    $("progress-train").value = nt;
    $("progress-calib").value = nc;
    $("count-train").textContent = `${nt}/${profile.n_train_needed}`;
    $("count-calib").textContent = `${nc}/${profile.n_calib_needed}`;
    $("build").disabled = !(nt >= profile.n_train_needed && nc >= profile.n_calib_needed);
    $("profile-status").textContent = profile.status === "none"
      ? "Профиль не создан."
      : `Идёт сбор: обучение ${nt}/${profile.n_train_needed}, настройка порога ${nc}/${profile.n_calib_needed}.`;
    if (mode !== "enroll") setMode("enroll");
  }
  input.disabled = mode === "enroll" && ready;
}

$("build").addEventListener("click", async () => {
  $("build").disabled = true;
  say("Обучение профиля…");
  try {
    await api("/api/profile/build", { confirm: true });
    await refreshProfile();
    setMode("verify");
    say("Профиль создан. Введите фразу для проверки.");
    input.focus();
  } catch (err) {
    say(err.message, "error");
    await refreshProfile();
  }
});

$("reset").addEventListener("click", async () => {
  if (!window.confirm("Сбросить профиль? Текущий профиль будет перенесён в архив, сбор начнётся заново.")) return;
  try {
    await api("/api/profile/reset", { confirm: true });
    $("verdict").hidden = true;
    $("charts").hidden = true;
    $("verify-empty").hidden = false;
    await refreshProfile();
    setMode("enroll");
    say("Профиль сброшен. Начните сбор заново.");
  } catch (err) {
    say(err.message, "error");
  }
});

$("again").addEventListener("click", () => {
  $("verdict").hidden = true;
  $("verify-empty").hidden = false;
  $("charts").hidden = true;
  say("");
  input.focus();
});

const TABS = ["enroll", "verify", "experiment"];
for (const name of TABS) {
  $(`tab-${name}`).addEventListener("click", () => setMode(name));
}
document.querySelector('[role="tablist"]').addEventListener("keydown", (e) => {
  if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
  const enabled = TABS.filter((n) => !$(`tab-${n}`).disabled);
  const current = enabled.indexOf(document.activeElement.id.replace("tab-", ""));
  if (current === -1) return;
  const next = enabled[(current + (e.key === "ArrowRight" ? 1 : enabled.length - 1)) % enabled.length];
  e.preventDefault();
  setMode(next);
  $(`tab-${next}`).focus();
});
$("panel-experiment").addEventListener("change", () => resetCapture());

refreshProfile();
