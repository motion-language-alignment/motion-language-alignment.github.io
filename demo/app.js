/* Live bidirectional retrieval in the browser.
   Text-to-Motion: the typed sentence is tokenized and encoded by the exported fp16 sentence tower, then
   ranked against the stored 256-d motion embeddings of every held-out test clip.
   Motion-to-Text: a clip's stored motion embedding is ranked against the stored sentence embeddings of
   the 5,670 complete descriptions, exactly as the paper evaluates it. */

import * as ort from "./vendor/ort/ort.wasm.bundle.min.mjs";
import { BertTokenizer } from "./vendor/transformers/transformers.min.js";

// The vendored runtime is the same build transformers.js depends on, so the two cannot drift apart.
ort.env.wasm.wasmPaths = new URL("./vendor/ort/", location.href).href;
ort.env.wasm.numThreads = 1;

const DEFAULT_MODEL_URL = "./model/text_tower_fp16.onnx";
const DIM = 256;
const TOP_K = 10;

// The encoder is 669 MB, which is past what a Pages repository may hold, so a deployment points this
// at wherever the file actually lives. A missing config keeps the local copy.
let modelUrl = DEFAULT_MODEL_URL;

const LEVEL_OF = {
    speed_state: "kinematic", longitudinal_motion: "kinematic", lateral_motion: "kinematic",
    lateral_stability: "dynamic", longitudinal_stability: "dynamic", ride_comfort: "semantic",
};
const CATEGORY_LABEL = {
    speed_state: "Speed", longitudinal_motion: "Longitudinal motion", lateral_motion: "Lateral motion",
    lateral_stability: "Lateral stability", longitudinal_stability: "Longitudinal stability",
    ride_comfort: "Ride comfort",
};
const PLOTS = [
    { title: "Speed", unit: "m/s", channels: [["vehicle_speed_mps", "speed", "--series-1"]] },
    { title: "Acceleration", unit: "m/s²", channels: [
        ["imu_acc_x_mps2", "x", "--series-1"], ["imu_acc_y_mps2", "y", "--series-2"],
        ["imu_acc_z_mps2", "z", "--series-3"]] },
    { title: "Angular rate", unit: "rad/s", channels: [
        ["imu_roll_rate_radps", "roll", "--series-1"], ["imu_pitch_rate_radps", "pitch", "--series-2"],
        ["imu_yaw_rate_radps", "yaw", "--series-3"]] },
    { title: "Steering angle", unit: "rad", channels: [["steering_angle_rad", "steering", "--series-4"]] },
    { title: "Wheel speed", unit: "m/s", channels: [
        ["wheel_speed_fl_mps", "FL", "--series-1"], ["wheel_speed_fr_mps", "FR", "--series-2"],
        ["wheel_speed_rl_mps", "RL", "--series-3"], ["wheel_speed_rr_mps", "RR", "--series-4"]] },
];

const token = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

const element = (id) => document.getElementById(id);
const status = (text) => { element("status").textContent = text; };
const pretty = (state) => (state ? state.replace(/_/g, " ") : "unknown");

let index = null;            // clip identity and ground-truth labels
let embeddings = null;       // Float32Array, n_clips * DIM
let declaredStates = null;   // text -> declared states, for the fixed evaluation queries
let tokenizer = null;
let session = null;
let lastResults = [];
let lastQueryStates = null;
let candidates = null;       // motion-to-text pool: 5,670 complete descriptions
let candidateEmbeddings = null;  // Float32Array, n_candidates * DIM
const M2T_TOP = 5;

/* ---------------------------------------------------------------- data */

async function loadIndex() {
    const config = await fetch("./config.json").then((r) => (r.ok ? r.json() : {})).catch(() => ({}));
    if (config.model_url) modelUrl = config.model_url;

    const [indexJson, embeddingBuffer, queriesJson] = await Promise.all([
        fetch("./data/index.json").then((r) => r.json()),
        fetch("./data/clip_embeddings.bin").then((r) => r.arrayBuffer()),
        fetch("./data/queries.json").then((r) => r.json()),
    ]);
    index = indexJson;
    const half = new Uint16Array(embeddingBuffer);
    if (half.length !== index.n_clips * DIM) throw new Error("embedding file does not match the index");
    embeddings = new Float32Array(half.length);
    for (let i = 0; i < half.length; i++) embeddings[i] = decodeHalf(half[i]);

    declaredStates = new Map(queriesJson.queries.map((q) => [normalize(q.text), q]));
    element("n-clips").textContent = index.n_clips.toLocaleString();
    element("pool-size").textContent = `${index.n_clips.toLocaleString()} clips × ${DIM} d`;

    const examples = element("examples");
    for (const query of pickExamples(queriesJson.queries)) {
        const button = document.createElement("button");
        button.textContent = query.text.replace(/^Find motion where (the )?/, "").replace(/\.$/, "");
        button.title = query.text;
        button.addEventListener("click", () => { element("query").value = query.text; search(); });
        examples.append(button);
    }
}

/* One short evaluation query per ontology level, so each example carries declared states. */
function pickExamples(queries) {
    const wanted = ["Q019", "Q022", "Q064", "Q082"];
    const chosen = wanted.map((id) => queries.find((q) => q.query_id === id)).filter(Boolean);
    return chosen.length ? chosen : queries.slice(0, 4);
}

function decodeHalf(bits) {
    const sign = (bits & 0x8000) ? -1 : 1;
    const exponent = (bits >> 10) & 0x1f;
    const fraction = bits & 0x3ff;
    if (exponent === 0) return sign * Math.pow(2, -14) * (fraction / 1024);
    if (exponent === 31) return fraction ? NaN : sign * Infinity;
    return sign * Math.pow(2, exponent - 15) * (1 + fraction / 1024);
}

const normalize = (text) => text.trim().replace(/\s+/g, " ").toLowerCase();

/* The motion-to-text pool is 5 MB and only needed once a clip is opened, so it is fetched then. */
async function loadCandidates() {
    if (candidates) return;
    const [pool, buffer] = await Promise.all([
        fetch("./data/m2t_candidates.json").then((r) => r.json()),
        fetch("./data/m2t_embeddings.bin").then((r) => r.arrayBuffer()),
    ]);
    const half = new Uint16Array(buffer);
    if (half.length !== pool.n_candidates * DIM) throw new Error("candidate embeddings do not match the pool");
    candidateEmbeddings = new Float32Array(half.length);
    for (let i = 0; i < half.length; i++) candidateEmbeddings[i] = decodeHalf(half[i]);
    candidates = pool;
}

/* Rank every complete description against the stored motion embedding of clip i: the paper's M2T. */
function describe(i) {
    const scores = new Float32Array(candidates.n_candidates);
    const base = i * DIM;
    for (let j = 0; j < candidates.n_candidates; j++) {
        let dot = 0;
        const row = j * DIM;
        for (let d = 0; d < DIM; d++) dot += embeddings[base + d] * candidateEmbeddings[row + d];
        scores[j] = dot;
    }
    const order = Array.from(scores.keys()).sort((a, b) => scores[b] - scores[a]).slice(0, M2T_TOP);
    return order.map((j, rank) => ({ rank: rank + 1, score: scores[j], ...candidates.candidates[j] }));
}

/* A fully labelled test clip at random, for readers who want motion-to-text without the encoder. */
async function sampleClip() {
    const complete = index.clips.filter((c) => index.categories.every((k) => c.labels[k].validity === "valid"));
    const clip = complete[Math.floor(Math.random() * complete.length)];
    for (const item of element("results").children) item.setAttribute("aria-selected", "false");
    const signals = await fetch(`./data/clips/${clip.i}.json`).then((r) => r.json());
    renderDetail({ rank: null, score: null, clip }, signals);
    status(`Sample clip ${clip.i}: one of the ${complete.length} test clips with all six labels valid, `
           + "the set the paper's motion-to-text evaluation uses.");
    await showDescriptions(clip);
}

async function showDescriptions(clip) {
    const box = element("m2t");
    if (!box) return;
    box.replaceChildren();
    const head = document.createElement("div");
    head.className = "plot-title";
    head.textContent = "Motion-to-Text · loading the 5,670 descriptions…";
    box.append(head);
    await loadCandidates();
    const started = performance.now();
    const ranked = describe(clip.i);
    const ms = performance.now() - started;
    head.textContent = `Motion-to-Text · top ${M2T_TOP} of ${candidates.n_candidates.toLocaleString()} `
                     + `complete descriptions, ranked in ${ms.toFixed(0)} ms`;
    const list = document.createElement("ol");
    list.className = "descriptions";
    for (const entry of ranked) {
        const item = document.createElement("li");
        const line = document.createElement("div");
        line.className = "rank-line";
        const text = document.createElement("span");
        text.className = "sentence";
        text.textContent = entry.text;
        const score = document.createElement("span");
        score.className = "score";
        score.textContent = `cos ${entry.score.toFixed(4)}`;
        line.append(text, score);
        item.append(line, verdictChips(entry.states, clip.labels));
        list.append(item);
    }
    box.append(list);
    const note = document.createElement("p");
    note.className = "note";
    const complete = index.categories.every((k) => clip.labels[k].validity === "valid");
    note.textContent = complete
        ? "Each chip compares a described state with the clip's ground truth. The top description's six states are "
          + "the predicted motion classes; this clip is fully labelled, so it counts in the paper's evaluation."
        : "Each chip compares a described state with the clip's ground truth where a label exists. Categories "
          + "without a valid label are left unverified; the paper's Recall@k uses fully labelled clips only.";
    box.append(note);
}

/* Six chips, one per category: agreement with the clip label, or unverified when the label is missing. */
function verdictChips(states, labels) {
    const box = document.createElement("div");
    box.className = "chips";
    for (const category of index.categories) {
        const label = labels[category];
        const chip = document.createElement("span");
        const valid = label.validity === "valid" && label.state;
        chip.className = `chip ${LEVEL_OF[category]} ` + (!valid ? "unverified" : label.state === states[category] ? "match" : "different");
        chip.textContent = pretty(states[category]);
        chip.title = `${CATEGORY_LABEL[category]}: ` + (!valid ? "clip label unknown" : label.state === states[category] ? "matches the clip" : `clip is ${pretty(label.state)}`);
        box.append(chip);
    }
    return box;
}

/* ---------------------------------------------------------------- model */

async function loadModel() {
    if (session) return;
    // The tokenizer files are fetched directly: the library's own resolver expects a model hub.
    if (!tokenizer) {
        const [definition, config] = await Promise.all([
            fetch("./model/tokenizer.json").then((r) => r.json()),
            fetch("./model/tokenizer_config.json").then((r) => r.json()),
        ]);
        if (config.tokenizer_class !== "BertTokenizer") {
            throw new Error(`unexpected tokenizer class: ${config.tokenizer_class}`);
        }
        tokenizer = new BertTokenizer(definition, config);
    }

    const weights = await downloadWeights();
    status("Starting the sentence encoder…");
    const started = performance.now();
    session = await ort.InferenceSession.create(weights, { executionProviders: ["wasm"] });
    console.log(`session created in ${Math.round(performance.now() - started)} ms`);
}

/* One pass into one preallocated buffer: a 669 MB model tolerates no extra copies. */
async function downloadWeights() {
    const bar = element("progress");
    bar.style.display = "block";
    const started = performance.now();
    const response = await fetch(modelUrl, { cache: "force-cache" });
    const total = Number(response.headers.get("content-length"));
    if (!total) throw new Error("the model response has no content-length");
    const bytes = new Uint8Array(total);
    const reader = response.body.getReader();
    let received = 0;
    for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        bytes.set(value, received);
        received += value.length;
        bar.firstElementChild.style.width = `${Math.min(100, 100 * received / total)}%`;
        status(`Downloading the sentence encoder: ${(received / 1e6).toFixed(0)} MB `
               + `of ${(total / 1e6).toFixed(0)} MB. This happens once.`);
    }
    if (received !== total) throw new Error(`model download stopped at ${received} of ${total} bytes`);
    bar.style.display = "none";
    console.log(`model downloaded in ${Math.round(performance.now() - started)} ms`);
    return bytes.buffer;
}

async function encode(text) {
    const encoded = await tokenizer(text, { padding: true, truncation: true, max_length: 512 });
    const length = encoded.input_ids.dims[1];
    const feeds = {
        input_ids: new ort.Tensor("int64", encoded.input_ids.data, [1, length]),
        attention_mask: new ort.Tensor("int64", encoded.attention_mask.data, [1, length]),
        token_type_ids: new ort.Tensor("int64",
            encoded.token_type_ids?.data ?? new BigInt64Array(length), [1, length]),
    };
    const output = await session.run(feeds);
    return output.embedding.data;
}

/* ---------------------------------------------------------------- search */

const STAGES = ["input", "encoder", "pool", "cosine", "output"];
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/* The walkthrough paces the stages; a plain search leaves them at their natural speed. */
const IMMEDIATE = { stage() {}, async hold() {}, async reveal() {} };

function markStage(current) {
    STAGES.forEach((name, position) => {
        const box = element(`flow-${name}`);
        box.classList.toggle("active", name === current);
        box.classList.toggle("done", current !== null && position < STAGES.indexOf(current));
        if (current === null) box.classList.remove("done");
    });
}

async function runQuery(text, pace = IMMEDIATE) {
    pace.stage("input");
    await pace.hold(900);

    pace.stage("encoder");
    element("query-vector").classList.remove("shown");
    await loadModel();
    status("Encoding the sentence…");
    const started = performance.now();
    const vector = await encode(text);
    const encoded = performance.now();
    element("query-vector").classList.add("shown");
    await pace.hold(900);

    pace.stage("pool");
    status(`Reading the stored embedding of all ${index.n_clips.toLocaleString()} test clips.`);
    await pace.hold(900);

    pace.stage("cosine");
    status("Scoring every clip against the sentence.");
    const scores = new Float32Array(index.n_clips);
    for (let i = 0; i < index.n_clips; i++) {
        let dot = 0;
        const base = i * DIM;
        for (let d = 0; d < DIM; d++) dot += embeddings[base + d] * vector[d];
        scores[i] = dot;
    }
    const order = Array.from(scores.keys()).sort((a, b) => scores[b] - scores[a]).slice(0, TOP_K);
    const ranked = performance.now();
    await pace.hold(900);

    pace.stage("output");
    lastQueryStates = declaredStates.get(normalize(text))?.states ?? null;
    lastResults = order.map((i, rank) => ({ rank: rank + 1, score: scores[i], clip: index.clips[i] }));
    renderResults();
    await pace.reveal();
    await select(0);
    await pace.hold(1200);
    markStage(null);
    status(`Encoded in ${Math.round(encoded - started)} ms, `
           + `ranked ${index.n_clips.toLocaleString()} clips in ${Math.round(ranked - encoded)} ms. `
           + (lastQueryStates
              ? "This sentence is one of the fixed evaluation queries, so its declared states are compared below."
              : "This sentence has no declared states, so the clip labels are shown without a comparison."));
}

async function search() {
    const text = element("query").value.trim();
    if (!text) { status("Type a sentence first."); return; }
    setBusy(true);
    try {
        await runQuery(text);
    } catch (error) {
        markStage(null);
        status(`Failed: ${error}`);
        throw error;
    } finally {
        setBusy(false);
    }
}

function setBusy(busy) {
    element("search").disabled = busy;
    element("walkthrough").disabled = busy;
    element("query").disabled = busy;
}

/* Runs the same query as the Search button, only slowly, so each stage can be read as it happens. */
async function walkthrough() {
    const text = element("query").value.trim() || element("query").placeholder;
    setBusy(true);
    element("results").replaceChildren();
    element("detail").replaceChildren();
    element("query-vector").classList.remove("shown");
    element("query").value = "";
    try {
        status("Writing the query…");
        for (const character of text) {
            element("query").value += character;
            await sleep(28);
        }
        await sleep(500);
        await runQuery(text, {
            stage: markStage,
            hold: (ms) => sleep(ms),
            async reveal() {
                const items = [...element("results").children];
                items.forEach((item) => { item.style.visibility = "hidden"; });
                for (const item of items) {
                    item.style.visibility = "";
                    await sleep(140);
                }
            },
        });
    } catch (error) {
        markStage(null);
        status(`Failed: ${error}`);
        throw error;
    } finally {
        setBusy(false);
    }
}

function renderResults() {
    const list = element("results");
    list.replaceChildren();
    lastResults.forEach((result, position) => {
        const item = document.createElement("li");
        item.setAttribute("aria-selected", position === 0 ? "true" : "false");
        item.addEventListener("click", () => select(position));

        const line = document.createElement("div");
        line.className = "rank-line";
        const rank = document.createElement("span");
        rank.className = "rank";
        rank.textContent = `Rank ${result.rank}`;
        const score = document.createElement("span");
        score.className = "score";
        score.textContent = `cos ${result.score.toFixed(4)}`;
        line.append(rank, score);

        const where = document.createElement("div");
        where.className = "where";
        where.textContent = `${result.clip.dataset} · ${result.clip.unit} · `
                          + `${result.clip.start_s.toFixed(1)}–${result.clip.end_s.toFixed(1)} s`;

        item.append(line, where, stateChips(result.clip.labels));
        list.append(item);
    });
}

/* Only the valid labels become chips; the rest are summarised, since the table below names them all. */
function stateChips(labels) {
    const box = document.createElement("div");
    box.className = "chips";
    let withheld = 0;
    for (const category of index.categories) {
        const label = labels[category];
        if (label.validity !== "valid" || !label.state) { withheld += 1; continue; }
        const chip = document.createElement("span");
        chip.className = `chip ${LEVEL_OF[category]}`;
        chip.textContent = pretty(label.state);
        chip.title = CATEGORY_LABEL[category];
        box.append(chip);
    }
    if (withheld) {
        const chip = document.createElement("span");
        chip.className = "chip unknown";
        chip.textContent = `${withheld} without a label`;
        box.append(chip);
    }
    return box;
}

/* ---------------------------------------------------------------- detail */

async function select(position) {
    for (const [i, item] of [...element("results").children].entries()) {
        item.setAttribute("aria-selected", i === position ? "true" : "false");
    }
    const result = lastResults[position];
    const clip = await fetch(`./data/clips/${result.clip.i}.json`).then((r) => r.json());
    renderDetail(result, clip);
}

function renderDetail(result, clip) {
    const detail = element("detail");
    detail.replaceChildren();

    const head = document.createElement("div");
    head.className = "detail-head";
    const title = document.createElement("h3");
    title.textContent = result.rank ? `Rank ${result.rank} · cosine ${result.score.toFixed(4)}` : "Sample clip";
    const where = document.createElement("div");
    where.className = "where";
    where.textContent = `${clip.dataset} · ${clip.unit} · `
                      + `${clip.clip_start_s.toFixed(1)}–${clip.clip_end_s.toFixed(1)} s`;
    head.append(title, where);

    const grid = document.createElement("div");
    grid.className = "detail-grid";

    const left = document.createElement("div");
    left.className = "panel";
    const bevTitle = document.createElement("div");
    bevTitle.className = "plot-title";
    bevTitle.textContent = "Trajectory, bird's-eye (right / forward, m; start heading up)";
    const bev = document.createElement("canvas");
    left.append(bevTitle, bev);
    left.append(stateTable(result.clip.labels));

    const right = document.createElement("div");
    right.className = "plots";
    const canvases = PLOTS.map((plot) => {
        const panel = document.createElement("div");
        panel.className = "panel";
        const title = document.createElement("div");
        title.className = "plot-title";
        title.textContent = `${plot.title} (${plot.unit})`;
        const canvas = document.createElement("canvas");
        panel.append(title, canvas);
        if (plot.channels.length > 1) {
            const legend = document.createElement("div");
            legend.className = "legend";
            for (const [channel, name, colour] of plot.channels) {
                const recorded = clip.c[channel]?.some((v) => v !== null && Number.isFinite(v));
                const entry = document.createElement("span");
                const swatch = document.createElement("i");
                swatch.style.background = recorded ? `var(${colour})` : "transparent";
                swatch.style.borderTop = recorded ? "" : "1px dashed currentColor";
                entry.append(swatch, recorded ? name : `${name} (not recorded)`);
                legend.append(entry);
            }
            panel.append(legend);
        }
        right.append(panel);
        return canvas;
    });

    grid.append(left, right);
    const m2t = document.createElement("div");
    m2t.id = "m2t";
    m2t.className = "panel";
    const button = document.createElement("button");
    button.className = "secondary";
    button.textContent = "Describe this clip (Motion-to-Text)";
    button.addEventListener("click", () => showDescriptions(result.clip));
    m2t.append(button);
    detail.append(head, grid, m2t);

    play(clip, bev, canvases);
}

/* Playback: the marker travels the trajectory in real time while a cursor crosses the plots. */
let playing = 0;

function play(clip, bev, canvases) {
    cancelAnimationFrame(playing);
    const draw = (at) => {
        drawTrajectory(bev, clip, at);
        PLOTS.forEach((plot, i) => drawSignal(canvases[i], clip, plot, at));
    };
    const still = matchMedia("(prefers-reduced-motion: reduce)").matches || clip.t.length < 2;
    if (still) { draw(null); return; }
    const t0 = clip.t[0], t1 = clip.t[clip.t.length - 1];
    const started = performance.now();
    const frame = (now) => {
        if (!bev.isConnected) return;
        const elapsed = ((now - started) / 1000) % (t1 - t0);
        draw(t0 + elapsed);
        playing = requestAnimationFrame(frame);
    };
    playing = requestAnimationFrame(frame);
}

/* Linear interpolation of the trajectory at time `at`; null means the clip start. */
function poseAt(points, times, at, inClip) {
    if (at === null) {
        const start = points.findIndex((_, i) => inClip(i));
        return start >= 0 ? points[start] : null;
    }
    let j = times.findIndex((time) => time > at);
    if (j <= 0) return points[j < 0 ? points.length - 1 : 0];
    const a = points[j - 1], b = points[j];
    const w = (at - times[j - 1]) / (times[j] - times[j - 1]);
    const dh = Math.atan2(Math.sin(b.h - a.h), Math.cos(b.h - a.h));
    return { r: a.r + (b.r - a.r) * w, f: a.f + (b.f - a.f) * w, h: a.h + dh * w };
}

function stateTable(labels) {
    const box = document.createElement("div");
    box.className = "states";
    const table = document.createElement("table");
    const head = document.createElement("tr");
    for (const column of ["Category", "Clip label", lastQueryStates ? "Requested" : ""]) {
        if (!column) continue;
        const cell = document.createElement("th");
        cell.textContent = column;
        head.append(cell);
    }
    table.append(head);

    for (const category of index.categories) {
        const label = labels[category];
        const row = document.createElement("tr");
        const name = document.createElement("td");
        name.textContent = CATEGORY_LABEL[category];
        const value = document.createElement("td");
        value.textContent = label.validity === "valid" ? pretty(label.state)
            : (label.validity === "not_applicable" ? "not applicable" : "unknown");
        row.append(name, value);

        if (lastQueryStates) {
            const requested = lastQueryStates[category];
            const cell = document.createElement("td");
            if (!requested) {
                cell.textContent = "—";
                cell.className = "verdict not-applicable";
            } else if (label.validity !== "valid") {
                cell.textContent = `${pretty(requested)}: unverified`;
                cell.className = "verdict unverified";
            } else if (label.state === requested) {
                cell.textContent = `${pretty(requested)}: match`;
                cell.className = "verdict match";
            } else {
                cell.textContent = `${pretty(requested)}: different`;
                cell.className = "verdict different";
            }
            row.append(cell);
        }
        table.append(row);
    }
    box.append(table);
    return box;
}

/* ---------------------------------------------------------------- drawing */

function prepare(canvas, height) {
    const ratio = window.devicePixelRatio || 1;
    const width = canvas.clientWidth || 520;
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    canvas.style.height = `${height}px`;
    const context = canvas.getContext("2d");
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, width, height);
    return { context, width, height };
}

function drawSignal(canvas, clip, plot, at = null) {
    const { context, width, height } = prepare(canvas, 84);
    const left = 44, right = 8, top = 8, bottom = 18;
    const series = plot.channels.map(([channel]) => clip.c[channel]);
    const values = series.flat().filter((v) => v !== null && Number.isFinite(v));
    if (!values.length) {
        context.fillStyle = token("--ink-mute");
        context.font = '11px "IBM Plex Sans", sans-serif';
        context.fillText("this recording has no such channel", left, height / 2);
        return;
    }
    let low = Math.min(...values), high = Math.max(...values);
    if (high - low < 1e-6) { low -= 0.5; high += 0.5; }
    const pad = (high - low) * 0.12;
    low -= pad; high += pad;

    const times = clip.t;
    const t0 = times[0], t1 = times[times.length - 1];
    const sx = (t) => left + (t - t0) / (t1 - t0) * (width - left - right);
    const sy = (v) => top + (high - v) / (high - low) * (height - top - bottom);

    context.fillStyle = token("--plot-window");
    context.fillRect(sx(0), top, sx(clip.clip_end_s - clip.clip_start_s) - sx(0), height - top - bottom);

    context.strokeStyle = token("--plot-grid");
    context.lineWidth = 1;
    context.beginPath();
    context.moveTo(left, sy(high)); context.lineTo(width - right, sy(high));
    context.moveTo(left, sy(low)); context.lineTo(width - right, sy(low));
    context.stroke();

    context.fillStyle = token("--ink-mute");
    context.font = '10px "IBM Plex Mono", monospace';
    context.fillText(high.toFixed(2), 2, sy(high) + 3);
    context.fillText(low.toFixed(2), 2, sy(low) + 3);
    context.fillText(`${t0.toFixed(0)} s`, left, height - 5);
    context.fillText(`${t1.toFixed(0)} s`, width - right - 22, height - 5);

    plot.channels.forEach(([, , colour], i) => {
        context.strokeStyle = token(colour);
        context.lineWidth = 1.2;
        context.beginPath();
        let drawing = false;
        series[i].forEach((value, j) => {
            if (value === null || !Number.isFinite(value)) { drawing = false; return; }
            const x = sx(times[j]), y = sy(value);
            if (drawing) context.lineTo(x, y); else context.moveTo(x, y);
            drawing = true;
        });
        context.stroke();
    });

    if (at !== null && values.length) {
        context.strokeStyle = token("--ink");
        context.lineWidth = 1;
        context.beginPath();
        context.moveTo(sx(at), top); context.lineTo(sx(at), height - bottom);
        context.stroke();
    }
}

function drawTrajectory(canvas, clip, at = null) {
    const { context, width, height } = prepare(canvas, 300);
    const heading0 = clip.h[0];
    const cos = Math.cos(-heading0 + Math.PI / 2), sin = Math.sin(-heading0 + Math.PI / 2);
    const points = clip.x.map((x, i) => {
        const y = clip.y[i];
        return { r: x * cos - y * sin, f: x * sin + y * cos, h: clip.h[i] - heading0 };
    });
    const inClip = (i) => clip.t[i] >= 0 && clip.t[i] <= clip.clip_end_s - clip.clip_start_s;

    const rs = points.map((p) => p.r), fs = points.map((p) => p.f);
    const span = Math.max(Math.max(...rs) - Math.min(...rs), Math.max(...fs) - Math.min(...fs), 12) * 1.15;
    const cr = (Math.max(...rs) + Math.min(...rs)) / 2, cf = (Math.max(...fs) + Math.min(...fs)) / 2;
    const scale = Math.min(width, height) / span;
    const sx = (r) => width / 2 + (r - cr) * scale;
    const sy = (f) => height / 2 - (f - cf) * scale;

    context.lineWidth = 1.5;
    for (const [only, colour] of [[false, token("--trace-context")], [true, token("--trace-clip")]]) {
        context.strokeStyle = colour;
        context.lineWidth = only ? 2.5 : 1.5;
        context.beginPath();
        let drawing = false;
        points.forEach((p, i) => {
            if (inClip(i) !== only) { drawing = false; return; }
            const x = sx(p.r), y = sy(p.f);
            if (drawing) context.lineTo(x, y); else context.moveTo(x, y);
            drawing = true;
        });
        context.stroke();
    }

    // At motorway speed the context spans a few hundred metres, where a to-scale car is sub-pixel.
    // The marker therefore has a floor in pixels and the caption says when it is no longer to scale.
    const p = poseAt(points, clip.t, at, inClip);
    let toScale = true;
    if (p) {
        const halfWidth = Math.max(0.9 * scale, 4);
        const halfLength = Math.max(2.3 * scale, 10);
        toScale = 0.9 * scale >= 4;
        context.save();
        context.translate(sx(p.r), sy(p.f));
        context.rotate(-p.h);
        context.strokeStyle = token("--ink");
        context.fillStyle = token("--paper");
        context.lineWidth = 1.2;
        context.beginPath();
        context.rect(-halfWidth, -halfLength, halfWidth * 2, halfLength * 2);
        context.fill();
        context.stroke();
        context.beginPath();
        context.moveTo(0, -halfLength);
        context.lineTo(0, -halfLength - Math.max(halfLength * 0.6, 8));
        context.stroke();
        context.restore();
    }

    context.fillStyle = token("--ink-mute");
    context.font = '10px "IBM Plex Mono", monospace';
    context.fillText(`${span.toFixed(0)} m across`, 2, height - 4);
    const caption = toScale ? "vehicle to scale" : "vehicle marker enlarged";
    context.fillText(caption, width - context.measureText(caption).width - 2, height - 4);
}

/* ---------------------------------------------------------------- start */

element("search").addEventListener("click", search);
element("walkthrough").addEventListener("click", walkthrough);
element("query").addEventListener("keydown", (event) => { if (event.key === "Enter") search(); });
element("sample").addEventListener("click", () => sampleClip().catch((error) => status(`Failed: ${error}`)));

loadIndex().then(() => {
    status("Clip index loaded. Type a sentence and press Search; the encoder is downloaded on the "
           + "first search and cached afterwards. Motion-to-Text needs no download: open a sample clip.");
}).catch((error) => status(`Failed to load the clip index: ${error}`));
