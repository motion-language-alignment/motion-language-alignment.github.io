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

    declaredStates = new Map([...queriesJson.queries, HIGH_SPEED_OVERSTEER].map((q) => [normalize(q.text), q]));
    element("n-clips").textContent = index.n_clips.toLocaleString();
    element("pool-size").textContent = `${index.n_clips.toLocaleString()} clips × ${DIM} d`;

    const examples = element("examples");
    for (const query of pickExamples(queriesJson.queries)) {
        const button = document.createElement("button");
        button.textContent = query.label ?? query.text.replace(/^Find motion where (the )?/, "").replace(/\.$/, "");
        button.title = query.text;
        button.addEventListener("click", () => { element("query").value = query.text; search(); });
        examples.append(button);
    }
}

/* The headline example names two states; the evaluation set has no such two-state query, so it is declared here. */
const HIGH_SPEED_OVERSTEER = {
    query_id: null, label: "high speed oversteer",
    text: "The vehicle moves at a high speed while exhibiting oversteer.",
    states: { speed_state: "high", longitudinal_motion: null, lateral_motion: null,
              lateral_stability: "oversteer", longitudinal_stability: null, ride_comfort: null },
};

/* One example per ontology category after the two-state headline, each carrying declared states. */
const EXAMPLE_IDS = ["Q019", "Q022", "Q052", "Q070", "Q082"];
const EXAMPLE_LABELS = { Q019: "extreme speed", Q022: "hard braking", Q052: "tight left turn",
                         Q070: "traction slip", Q082: "uncomfortable ride" };
function pickExamples(queries) {
    const chosen = EXAMPLE_IDS.map((id) => queries.find((q) => q.query_id === id)).filter(Boolean)
                              .map((q) => ({ ...q, label: EXAMPLE_LABELS[q.query_id] }));
    return [HIGH_SPEED_OVERSTEER, ...(chosen.length ? chosen : queries.slice(0, 5))];
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
    status(`Clip ${clip.i} opened without any typed sentence. Its stored motion embedding (from IMU + CAN) is ranked `
           + "against 5,670 descriptions; the best-matching one is the predicted description.");
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
    head.textContent = `Motion-to-Text · this clip's motion embedding ranked against `
                     + `${candidates.n_candidates.toLocaleString()} complete descriptions in ${ms.toFixed(0)} ms`;
    box.append(predictedBlock(ranked[0], clip.labels));
    const listTitle = document.createElement("div");
    listTitle.className = "plot-title descriptions-title";
    listTitle.textContent = `Next closest descriptions (ranks 2–${M2T_TOP})`;
    box.append(listTitle);
    const list = document.createElement("ol");
    list.className = "descriptions";
    list.style.counterReset = "description 1";   // the CSS counter numbers these from 2
    for (const entry of ranked.slice(1)) {
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
    note.textContent = "Chips compare each described state with the ground truth: green agrees, red differs, dashed "
        + (complete ? "unverified. This clip is fully labelled, so it counts in the paper's Motion-to-Text evaluation."
                    : "unverified. Categories without a valid label are unverified; the paper's Recall@k uses fully labelled clips only.");
    box.append(note);
}

/* The top description is the prediction: its sentence, then its six states next to the clip's ground truth. */
function predictedBlock(top, labels) {
    const block = document.createElement("div");
    block.className = "m2t-result";
    const cap = document.createElement("div");
    cap.className = "plot-title";
    cap.textContent = `Predicted description · cos ${top.score.toFixed(4)}`;
    const lead = document.createElement("p");
    lead.className = "lead";
    lead.textContent = top.text;
    block.append(cap, lead, comparisonTable(top.states, labels));
    const checked = index.categories.filter((k) => labels[k].validity === "valid" && labels[k].state);
    const matches = checked.filter((k) => labels[k].state === top.states[k]).length;
    const summary = document.createElement("p");
    summary.className = "summary";
    summary.innerHTML = `<b>${matches} of ${checked.length}</b> predicted states match the ground-truth labels`
                      + (checked.length < index.categories.length ? `; ${index.categories.length - checked.length} unverified.` : ".");
    block.append(summary);
    return block;
}

function comparisonTable(states, labels) {
    const box = document.createElement("div");
    box.className = "states";
    const table = document.createElement("table");
    const head = document.createElement("tr");
    for (const column of ["Category", "Predicted", "GT label"]) {
        const cell = document.createElement("th");
        cell.textContent = column;
        head.append(cell);
    }
    table.append(head);
    for (const category of index.categories) {
        const label = labels[category], valid = label.validity === "valid" && label.state;
        const row = document.createElement("tr");
        const name = document.createElement("td");
        name.textContent = CATEGORY_LABEL[category];
        const predicted = document.createElement("td");
        predicted.textContent = pretty(states[category]);
        predicted.className = "verdict " + (!valid ? "unverified" : label.state === states[category] ? "match" : "different");
        const truth = document.createElement("td");
        truth.textContent = valid ? pretty(label.state) : (label.validity === "not_applicable" ? "not applicable" : "unknown");
        row.append(name, predicted, truth);
        table.append(row);
    }
    box.append(table);
    return box;
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

/* A plain search shows each stage; IMMEDIATE skips the show for programmatic use. */
const SHOWN = { stage: markStage, hold: sleep, reveal: staggerResults, fly };
const IMMEDIATE = { stage() {}, async hold() {}, async reveal() {}, async fly() {} };

/* A copy of the sentence travels from one element to another and lands (or shrinks into the target). */
async function fly(text, from, to, { shrink = false, ms = 700 } = {}) {
    const a = from.getBoundingClientRect(), b = to.getBoundingClientRect();
    const ghost = document.createElement("div");
    ghost.className = "fly";
    ghost.textContent = text;
    const width = Math.min(a.width, 720);
    ghost.style.left = `${a.left}px`; ghost.style.top = `${a.top}px`; ghost.style.width = `${width}px`;
    document.body.append(ghost);
    const dx = b.left - a.left, dy = b.top - a.top;
    const end = shrink
        ? { transform: `translate(${dx + b.width / 2 - width / 2}px, ${dy + b.height / 2 - a.height / 2}px) scale(.08)`, opacity: 0 }
        : { transform: `translate(${dx}px, ${dy}px) scale(${Math.min(1, b.width / width)})`, opacity: 1 };
    const flight = ghost.animate(
        [{ transform: "translate(0, 0) scale(1)", opacity: 1, transformOrigin: "left top" },
         { ...end, transformOrigin: "left top" }],
        { duration: ms, easing: "cubic-bezier(0.16, 1, 0.3, 1)", fill: "forwards" });
    await flight.finished;
    ghost.remove();
}

function markStage(current) {
    const at = current === null ? -1 : STAGES.indexOf(current);
    STAGES.forEach((name, position) => {
        const box = element(`flow-${name}`);
        box.classList.toggle("active", name === current);
        box.classList.toggle("done", current !== null && position < at);
        if (current === null) box.classList.remove("done");
    });
    // arrows light up once the stage they lead to has been reached
    document.querySelectorAll(".flow .arrow").forEach((arrow, k) => arrow.classList.toggle("lit", k < at));
}

async function staggerResults() {
    const items = [...element("results").children];
    items.forEach((item, i) => { item.classList.add("enter"); item.style.setProperty("--delay", `${i * 70}ms`); });
    await sleep(items.length * 70 + 300);
}

async function runQuery(text, pace = SHOWN) {
    const ticket = element("query-ticket");
    ticket.textContent = text;
    element("query-vector").classList.remove("shown");
    ticket.style.visibility = "hidden";
    pace.stage("input");
    await pace.fly(text, element("query"), ticket);
    ticket.style.visibility = "";
    await pace.hold(600);

    pace.stage("encoder");
    await loadModel();
    status("Encoding the sentence…");
    const flight = pace.fly(text, ticket, element("flow-encoder").querySelector(".net"), { shrink: true, ms: 900 });
    const started = performance.now();
    const vector = await encode(text);
    const encoded = performance.now();
    await flight;
    await pace.hold(1500);
    element("query-vector").classList.add("shown");
    await pace.hold(500);

    pace.stage("pool");
    status(`Reading the stored embedding of all ${index.n_clips.toLocaleString()} test clips.`);
    await pace.hold(1200);

    pace.stage("cosine");
    status("Scoring every clip against the sentence.");
    const scoring = performance.now();
    const scores = new Float32Array(index.n_clips);
    for (let i = 0; i < index.n_clips; i++) {
        let dot = 0;
        const base = i * DIM;
        for (let d = 0; d < DIM; d++) dot += embeddings[base + d] * vector[d];
        scores[i] = dot;
    }
    const order = Array.from(scores.keys()).sort((a, b) => scores[b] - scores[a]).slice(0, TOP_K);
    const ranked = performance.now();
    await pace.hold(2600);

    pace.stage("output");
    lastQueryStates = declaredStates.get(normalize(text))?.states ?? null;
    lastResults = order.map((i, rank) => ({ rank: rank + 1, score: scores[i], clip: index.clips[i] }));
    renderResults();
    await pace.reveal();
    await select(0);
    await pace.hold(1200);
    markStage(null);
    status(`Encoded in ${Math.round(encoded - started)} ms, `
           + `ranked ${index.n_clips.toLocaleString()} clips in ${Math.round(ranked - scoring)} ms. `
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
    element("query").disabled = busy;
    document.querySelectorAll("#examples button").forEach((button) => { button.disabled = busy; });
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
    for (const column of ["Category", "GT label", lastQueryStates ? "Requested" : ""]) {
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
element("query").addEventListener("keydown", (event) => { if (event.key === "Enter") search(); });
element("sample").addEventListener("click", () => sampleClip().catch((error) => status(`Failed: ${error}`)));

loadIndex().then(() => {
    status("Clip index loaded. Type a sentence and press Search; the encoder is downloaded on the "
           + "first search and cached afterwards. Motion-to-Text needs no download: describe a random clip.");
}).catch((error) => status(`Failed to load the clip index: ${error}`));
