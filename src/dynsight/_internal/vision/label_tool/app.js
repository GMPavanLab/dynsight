/* dynsight label tool - frontend */

"use strict";

/* ---------- constants ---------- */

const PALETTE = [
    "#f43f5e", "#f97316", "#eab308", "#22c55e", "#06b6d4",
    "#3b82f6", "#8b5cf6", "#ec4899", "#14b8a6", "#a3e635",
];
const MIN_BOX_SIZE = 3; // px, in image space
const HANDLE_SIZE = 7; // px, in screen space
const HANDLE_HIT = 6; // px tolerance
const MIN_SCALE = 0.05;
const MAX_SCALE = 32;

/* ---------- state ---------- */

const state = {
    workspace: "",
    images: [], // [{name, width, height}]
    annotations: {}, // name -> [{label, x, y, w, h}]
    labels: [], // [{name, color}]
    activeLabel: null,
    current: -1,
    selection: -1,
    view: { scale: 1, x: 0, y: 0 },
    fitted: true, // refit on container resize until the user zooms/pans
};

let drag = null; // {mode, ...} while a pointer drag is active
let hover = { box: -1, handle: -1 };
let pointer = { x: 0, y: 0, inside: false };
let spaceDown = false;
let syncTimer = null;
let dirty = false; // changes not yet saved to a session file
let sessionPath = null; // last file the session was saved to / loaded from
let quitAfterSave = false;

const imageCache = new Map(); // name -> HTMLImageElement
const imageVersion = new Map(); // name -> int, bumped on re-upload

function imageUrl(name) {
    const version = imageVersion.get(name);
    const suffix = version ? `?v=${version}` : "";
    return `/images/${encodeURIComponent(name)}${suffix}`;
}

/* ---------- dom ---------- */

const $ = (id) => document.getElementById(id);
const canvas = $("canvas");
const ctx = canvas.getContext("2d");
const stage = $("stage");

/* ---------- helpers ---------- */

function currentImage() {
    return state.images[state.current] || null;
}

function currentBoxes() {
    const img = currentImage();
    if (!img) return [];
    if (!state.annotations[img.name]) state.annotations[img.name] = [];
    return state.annotations[img.name];
}

function labelColor(name) {
    const label = state.labels.find((l) => l.name === name);
    return label ? label.color : "#9ca3af";
}

function clamp(value, lo, hi) {
    return Math.max(lo, Math.min(hi, value));
}

function screenToImage(sx, sy) {
    return {
        x: (sx - state.view.x) / state.view.scale,
        y: (sy - state.view.y) / state.view.scale,
    };
}

/* ---------- api ---------- */

async function api(path, options = {}) {
    const response = await fetch(path, options);
    let payload = {};
    try {
        payload = await response.json();
    } catch {
        /* non-json error */
    }
    if (!response.ok) {
        throw new Error(payload.error || `Request failed (${response.status})`);
    }
    return payload;
}

function setSaveStatus() {
    const el = $("saveStatus");
    if (dirty) {
        el.textContent = "● Unsaved session";
        el.className = "busy";
    } else if (sessionPath) {
        el.textContent = `Saved ✓ (${sessionPath})`;
        el.className = "ok";
    } else {
        el.textContent = "";
        el.className = "";
    }
}

// The session is never written to disk automatically: edits are only
// mirrored to the server's memory so a page reload does not lose work
// while the server is running. Disk writes happen exclusively through
// the "Save session" dialog, to a user-chosen path.
function markChanged() {
    dirty = true;
    setSaveStatus();
    clearTimeout(syncTimer);
    syncTimer = setTimeout(syncSession, 300);
}

async function syncSession() {
    clearTimeout(syncTimer);
    try {
        await api("/api/sync", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: sessionBody(),
        });
    } catch {
        /* retried on the next change */
    }
}

function sessionBody() {
    return JSON.stringify({
        labels: state.labels,
        annotations: state.annotations,
    });
}

// Mirror unsaved work to the server's memory when the page is closed
// or reloaded (no disk write). The server keeps running: it is stopped
// only via the Quit button or Ctrl+C.
window.addEventListener("pagehide", () => {
    navigator.sendBeacon(
        "/api/sync",
        new Blob([sessionBody()], { type: "application/json" }),
    );
});

// Warn before leaving the page with an unsaved session.
window.addEventListener("beforeunload", (e) => {
    if (dirty) e.preventDefault();
});

/* ---------- toasts ---------- */

function toast(message, cls = "", detail = "", timeout = 6000) {
    const el = document.createElement("div");
    el.className = `toast ${cls}`;
    el.textContent = message;
    if (detail) {
        const line = document.createElement("span");
        line.className = "mono";
        line.textContent = detail;
        el.appendChild(line);
    }
    el.onclick = () => el.remove();
    $("toasts").appendChild(el);
    setTimeout(() => el.remove(), timeout);
}

/* ---------- progress ---------- */

let progressPoll = null;

function showProgress(label) {
    $("progressLabel").textContent = label;
    $("progressPct").textContent = "";
    $("progressFill").classList.add("indeterminate");
    $("progress").classList.remove("hidden");
}

function setProgress(done, total) {
    const fill = $("progressFill");
    if (total > 0) {
        const pct = Math.min(100, Math.round((done / total) * 100));
        fill.classList.remove("indeterminate");
        fill.style.width = `${pct}%`;
        $("progressPct").textContent = `${pct}%`;
    } else {
        fill.classList.add("indeterminate");
        $("progressPct").textContent = done > 0 ? String(done) : "";
    }
}

function hideProgress() {
    stopProgressPoll();
    $("progress").classList.add("hidden");
    $("progressFill").style.width = "0%";
    $("progressFill").classList.remove("indeterminate");
}

// Long server-side operations (export, synthesize, frame extraction)
// report their progress through /api/progress, polled while the main
// request is in flight.
function startProgressPoll(fallbackLabel) {
    stopProgressPoll();
    progressPoll = setInterval(async () => {
        try {
            const p = await api("/api/progress");
            if (p.active) {
                $("progressLabel").textContent =
                    (p.label || fallbackLabel) + "…";
                setProgress(p.done || 0, p.total || 0);
            }
        } catch {
            /* server busy or gone; keep the bar as-is */
        }
    }, 250);
}

function stopProgressPoll() {
    if (progressPoll) {
        clearInterval(progressPoll);
        progressPoll = null;
    }
}

/* ---------- sidebar: labels ---------- */

function renderLabels() {
    const list = $("labelList");
    list.innerHTML = "";
    state.labels.forEach((label, idx) => {
        const li = document.createElement("li");
        if (label.name === state.activeLabel) li.classList.add("active");

        const id = document.createElement("span");
        id.className = "class-id";
        id.textContent = String(idx);

        const dot = document.createElement("span");
        dot.className = "color-dot";
        dot.style.backgroundColor = label.color;

        const name = document.createElement("span");
        name.className = "item-name";
        name.textContent = label.name;

        const count = document.createElement("span");
        count.className = "item-badge";
        count.textContent = String(countBoxes(label.name));

        const del = document.createElement("button");
        del.className = "del-btn";
        del.textContent = "×";
        del.title = "Delete label and its boxes";
        del.onclick = (e) => {
            e.stopPropagation();
            deleteLabel(label.name);
        };

        li.append(id, dot, name, count, del);
        li.onclick = () => {
            state.activeLabel = label.name;
            const boxes = currentBoxes();
            if (state.selection >= 0 && boxes[state.selection]) {
                boxes[state.selection].label = label.name;
                markChanged();
            }
            renderLabels();
            render();
        };
        list.appendChild(li);
    });
    $("labelHint").classList.toggle("hidden", state.labels.length > 0);
}

function countBoxes(labelName) {
    let total = 0;
    for (const boxes of Object.values(state.annotations)) {
        total += boxes.filter((b) => b.label === labelName).length;
    }
    return total;
}

function addLabel(name) {
    if (!name || state.labels.some((l) => l.name === name)) return;
    const color = PALETTE[state.labels.length % PALETTE.length];
    state.labels.push({ name, color });
    state.activeLabel = name;
    markChanged();
    renderLabels();
    render();
}

function deleteLabel(name) {
    const used = countBoxes(name);
    if (
        used > 0 &&
        !confirm(`Delete label "${name}" and its ${used} box(es)?`)
    ) {
        return;
    }
    state.labels = state.labels.filter((l) => l.name !== name);
    for (const key of Object.keys(state.annotations)) {
        state.annotations[key] = state.annotations[key].filter(
            (b) => b.label !== name,
        );
    }
    if (state.activeLabel === name) state.activeLabel = null;
    state.selection = -1;
    markChanged();
    renderLabels();
    renderImages();
    render();
}

$("labelForm").onsubmit = (e) => {
    e.preventDefault();
    addLabel($("labelInput").value.trim());
    $("labelInput").value = "";
};

/* ---------- sidebar: images ---------- */

function renderImages() {
    const list = $("imageList");
    list.innerHTML = "";
    state.images.forEach((info, idx) => {
        const li = document.createElement("li");
        if (idx === state.current) li.classList.add("active");

        const thumb = document.createElement("img");
        thumb.className = "thumb";
        thumb.loading = "lazy";
        thumb.src = imageUrl(info.name);

        const name = document.createElement("span");
        name.className = "item-name";
        name.textContent = info.name;
        name.title = info.name;

        const count = document.createElement("span");
        count.className = "item-badge";
        const n = (state.annotations[info.name] || []).length;
        count.textContent = n > 0 ? String(n) : "";

        const del = document.createElement("button");
        del.className = "del-btn";
        del.textContent = "×";
        del.title = "Remove image";
        del.onclick = (e) => {
            e.stopPropagation();
            deleteImage(info.name);
        };

        li.append(thumb, name, count, del);
        li.onclick = () => selectImage(idx);
        list.appendChild(li);
    });
    $("imageCounter").textContent = state.images.length
        ? `${state.current + 1} / ${state.images.length}`
        : "0 / 0";
    $("emptyState").classList.toggle("hidden", state.images.length > 0);
}

async function deleteImage(name) {
    if (!confirm(`Remove "${name}" and its annotations?`)) return;
    try {
        await api(`/api/images?name=${encodeURIComponent(name)}`, {
            method: "DELETE",
        });
    } catch (err) {
        toast(`Could not delete: ${err.message}`, "error");
        return;
    }
    const idx = state.images.findIndex((i) => i.name === name);
    state.images = state.images.filter((i) => i.name !== name);
    delete state.annotations[name];
    imageCache.delete(name);
    if (state.current >= state.images.length) {
        state.current = state.images.length - 1;
    } else if (idx <= state.current) {
        state.current = Math.max(0, state.current - (idx < state.current));
    }
    state.selection = -1;
    markChanged();
    selectImage(state.current, true);
    renderLabels();
}

function selectImage(idx, force = false) {
    if (idx === state.current && !force) return;
    state.current = clamp(idx, -1, state.images.length - 1);
    state.selection = -1;
    const info = currentImage();
    if (info && !imageCache.has(info.name)) {
        const img = new Image();
        img.onload = () => {
            if (currentImage() === info) fitView();
        };
        img.src = imageUrl(info.name);
        imageCache.set(info.name, img);
    }
    fitView();
    renderImages();
}

$("prevBtn").onclick = () => {
    if (state.current > 0) selectImage(state.current - 1);
};
$("nextBtn").onclick = () => {
    if (state.current < state.images.length - 1) {
        selectImage(state.current + 1);
    }
};

/* ---------- uploads ---------- */

async function uploadImages(files) {
    const list = Array.from(files);
    if (!list.length) return;
    let done = 0;
    showProgress(`Uploading images (0/${list.length})…`);
    setProgress(0, list.length);
    try {
        for (const [idx, file] of list.entries()) {
            try {
                const info = await api(
                    `/api/images?name=${encodeURIComponent(file.name)}`,
                    { method: "POST", body: file },
                );
                const existing = state.images.findIndex(
                    (i) => i.name === info.name,
                );
                if (existing >= 0) {
                    state.images[existing] = info;
                    imageCache.delete(info.name);
                    imageVersion.set(
                        info.name,
                        (imageVersion.get(info.name) || 0) + 1,
                    );
                } else {
                    state.images.push(info);
                }
                done += 1;
            } catch (err) {
                toast(`"${file.name}": ${err.message}`, "error");
            }
            $("progressLabel").textContent =
                `Uploading images (${idx + 1}/${list.length})…`;
            setProgress(idx + 1, list.length);
        }
    } finally {
        hideProgress();
    }
    if (done > 0) {
        toast(`Added ${done} image(s).`, "ok");
        if (state.current < 0) selectImage(0);
        renderImages();
    }
}

let pendingVideo = null;

function askVideoStride(file) {
    pendingVideo = file;
    $("videoFileName").textContent = file.name;
    $("videoDialog").showModal();
}

// Upload with XMLHttpRequest to get byte-level upload progress, then
// poll /api/progress while the server extracts frames.
function uploadVideo(url, file) {
    return new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.open("POST", url);
        xhr.upload.onprogress = (e) => {
            if (e.lengthComputable) {
                $("progressLabel").textContent = "Uploading video…";
                setProgress(e.loaded, e.total);
            }
        };
        xhr.upload.onload = () => {
            showProgress("Extracting frames…");
            startProgressPoll("Extracting frames");
        };
        xhr.onload = () => {
            let payload = {};
            try {
                payload = JSON.parse(xhr.responseText);
            } catch {
                /* non-json */
            }
            if (xhr.status >= 200 && xhr.status < 300) resolve(payload);
            else {
                reject(
                    new Error(
                        payload.error || `Request failed (${xhr.status})`,
                    ),
                );
            }
        };
        xhr.onerror = () => reject(new Error("Network error"));
        xhr.send(file);
    });
}

$("videoForm").onsubmit = async (e) => {
    e.preventDefault();
    const stride = $("videoForm").elements.stride.value || "1";
    const file = pendingVideo;
    pendingVideo = null;
    $("videoDialog").close();
    if (!file) return;
    showProgress("Uploading video…");
    try {
        const result = await uploadVideo(
            `/api/video?name=${encodeURIComponent(file.name)}` +
                `&stride=${encodeURIComponent(stride)}`,
            file,
        );
        for (const info of result.frames) {
            if (!state.images.some((i) => i.name === info.name)) {
                state.images.push(info);
            }
        }
        toast(`Added ${result.frames.length} frame(s).`, "ok");
        if (state.current < 0) selectImage(0);
        renderImages();
    } catch (err) {
        toast(`Video import failed: ${err.message}`, "error");
    } finally {
        hideProgress();
    }
};

$("addImagesBtn").onclick = () => $("imageFiles").click();
$("addVideoBtn").onclick = () => $("videoFile").click();
$("imageFiles").onchange = (e) => {
    uploadImages(e.target.files);
    e.target.value = "";
};
$("videoFile").onchange = (e) => {
    if (e.target.files[0]) askVideoStride(e.target.files[0]);
    e.target.value = "";
};

/* drag & drop */

let dragDepth = 0;
stage.addEventListener("dragenter", (e) => {
    e.preventDefault();
    dragDepth += 1;
    $("dropHint").classList.remove("hidden");
});
stage.addEventListener("dragleave", () => {
    dragDepth = Math.max(0, dragDepth - 1);
    if (dragDepth === 0) $("dropHint").classList.add("hidden");
});
stage.addEventListener("dragover", (e) => e.preventDefault());
stage.addEventListener("drop", (e) => {
    e.preventDefault();
    dragDepth = 0;
    $("dropHint").classList.add("hidden");
    const files = Array.from(e.dataTransfer.files);
    const videos = files.filter((f) => f.type.startsWith("video/"));
    const images = files.filter((f) => f.type.startsWith("image/"));
    if (images.length) uploadImages(images);
    if (videos.length) askVideoStride(videos[0]);
});

/* ---------- canvas: view ---------- */

function resizeCanvas() {
    const dpr = window.devicePixelRatio || 1;
    const rect = stage.getBoundingClientRect();
    canvas.width = Math.round(rect.width * dpr);
    canvas.height = Math.round(rect.height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (state.fitted) fitView();
    else render();
}

function fitView() {
    const info = currentImage();
    if (!info) {
        render();
        return;
    }
    const rect = stage.getBoundingClientRect();
    const pad = 24;
    const scale = Math.min(
        (rect.width - pad * 2) / info.width,
        (rect.height - pad * 2) / info.height,
        1,
    );
    state.view.scale = Math.max(scale, MIN_SCALE);
    state.view.x = (rect.width - info.width * state.view.scale) / 2;
    state.view.y = (rect.height - info.height * state.view.scale) / 2;
    state.fitted = true;
    updateZoomText();
    render();
}

function setZoom(newScale, cx, cy) {
    const scale = clamp(newScale, MIN_SCALE, MAX_SCALE);
    const before = screenToImage(cx, cy);
    state.view.scale = scale;
    state.view.x = cx - before.x * scale;
    state.view.y = cy - before.y * scale;
    state.fitted = false;
    updateZoomText();
    render();
}

function updateZoomText() {
    $("zoomText").textContent = `${Math.round(state.view.scale * 100)}%`;
}

$("fitBtn").onclick = fitView;
$("zoomInBtn").onclick = () => {
    const rect = stage.getBoundingClientRect();
    setZoom(state.view.scale * 1.25, rect.width / 2, rect.height / 2);
};
$("zoomOutBtn").onclick = () => {
    const rect = stage.getBoundingClientRect();
    setZoom(state.view.scale / 1.25, rect.width / 2, rect.height / 2);
};

canvas.addEventListener(
    "wheel",
    (e) => {
        e.preventDefault();
        const factor = Math.exp(-e.deltaY * 0.0015);
        setZoom(state.view.scale * factor, e.offsetX, e.offsetY);
    },
    { passive: false },
);

/* ---------- canvas: hit testing ---------- */

function handlePositions(box) {
    const s = state.view.scale;
    const x = box.x * s + state.view.x;
    const y = box.y * s + state.view.y;
    const w = box.w * s;
    const h = box.h * s;
    return [
        { x, y, cursor: "nwse-resize", dx: -1, dy: -1 },
        { x: x + w / 2, y, cursor: "ns-resize", dx: 0, dy: -1 },
        { x: x + w, y, cursor: "nesw-resize", dx: 1, dy: -1 },
        { x: x + w, y: y + h / 2, cursor: "ew-resize", dx: 1, dy: 0 },
        { x: x + w, y: y + h, cursor: "nwse-resize", dx: 1, dy: 1 },
        { x: x + w / 2, y: y + h, cursor: "ns-resize", dx: 0, dy: 1 },
        { x, y: y + h, cursor: "nesw-resize", dx: -1, dy: 1 },
        { x, y: y + h / 2, cursor: "ew-resize", dx: -1, dy: 0 },
    ];
}

function hitTest(sx, sy) {
    const boxes = currentBoxes();
    // Handles of the selected box take priority.
    if (state.selection >= 0 && boxes[state.selection]) {
        const handles = handlePositions(boxes[state.selection]);
        for (let h = 0; h < handles.length; h++) {
            if (
                Math.abs(sx - handles[h].x) <= HANDLE_HIT &&
                Math.abs(sy - handles[h].y) <= HANDLE_HIT
            ) {
                return { box: state.selection, handle: h };
            }
        }
    }
    const pt = screenToImage(sx, sy);
    let best = -1;
    let bestArea = Infinity;
    boxes.forEach((box, idx) => {
        const inside =
            pt.x >= box.x &&
            pt.x <= box.x + box.w &&
            pt.y >= box.y &&
            pt.y <= box.y + box.h;
        const area = box.w * box.h;
        if (inside && area < bestArea) {
            best = idx;
            bestArea = area;
        }
    });
    return { box: best, handle: -1 };
}

/* ---------- canvas: interactions ---------- */

canvas.addEventListener("pointerdown", (e) => {
    if (!currentImage() && e.button === 0) return;
    canvas.setPointerCapture(e.pointerId);
    const info = currentImage();

    if (e.button === 1 || (e.button === 0 && (spaceDown || !info))) {
        drag = {
            mode: "pan",
            startX: e.offsetX,
            startY: e.offsetY,
            viewX: state.view.x,
            viewY: state.view.y,
        };
        return;
    }
    if (e.button !== 0) return;

    const hit = hitTest(e.offsetX, e.offsetY);
    const boxes = currentBoxes();

    if (hit.handle >= 0) {
        const box = boxes[hit.box];
        drag = {
            mode: "resize",
            index: hit.box,
            handle: hit.handle,
            orig: { ...box },
            moved: false,
        };
        return;
    }
    if (hit.box >= 0) {
        state.selection = hit.box;
        const pt = screenToImage(e.offsetX, e.offsetY);
        const box = boxes[hit.box];
        drag = {
            mode: "move",
            index: hit.box,
            offsetX: pt.x - box.x,
            offsetY: pt.y - box.y,
            moved: false,
        };
        render();
        return;
    }

    state.selection = -1;
    if (state.activeLabel) {
        const pt = screenToImage(e.offsetX, e.offsetY);
        const x = clamp(pt.x, 0, info.width);
        const y = clamp(pt.y, 0, info.height);
        drag = { mode: "draw", startX: x, startY: y, rect: null };
    } else {
        drag = {
            mode: "pan",
            startX: e.offsetX,
            startY: e.offsetY,
            viewX: state.view.x,
            viewY: state.view.y,
        };
    }
    render();
});

canvas.addEventListener("pointermove", (e) => {
    pointer = { x: e.offsetX, y: e.offsetY, inside: true };
    const info = currentImage();

    if (!drag) {
        hover = info ? hitTest(e.offsetX, e.offsetY) : { box: -1, handle: -1 };
        updateCursor();
        render();
        return;
    }

    if (drag.mode === "pan") {
        state.view.x = drag.viewX + (e.offsetX - drag.startX);
        state.view.y = drag.viewY + (e.offsetY - drag.startY);
        state.fitted = false;
        render();
        return;
    }

    const pt = screenToImage(e.offsetX, e.offsetY);
    const boxes = currentBoxes();

    if (drag.mode === "draw") {
        const x = clamp(pt.x, 0, info.width);
        const y = clamp(pt.y, 0, info.height);
        drag.rect = {
            x: Math.min(drag.startX, x),
            y: Math.min(drag.startY, y),
            w: Math.abs(x - drag.startX),
            h: Math.abs(y - drag.startY),
        };
    } else if (drag.mode === "move") {
        const box = boxes[drag.index];
        box.x = clamp(pt.x - drag.offsetX, 0, info.width - box.w);
        box.y = clamp(pt.y - drag.offsetY, 0, info.height - box.h);
        drag.moved = true;
    } else if (drag.mode === "resize") {
        resizeBox(boxes[drag.index], drag, pt, info);
        drag.moved = true;
    }
    render();
});

function resizeBox(box, dragState, pt, info) {
    const orig = dragState.orig;
    const handle = handlePositions(orig)[dragState.handle];
    let x1 = orig.x;
    let y1 = orig.y;
    let x2 = orig.x + orig.w;
    let y2 = orig.y + orig.h;
    const px = clamp(pt.x, 0, info.width);
    const py = clamp(pt.y, 0, info.height);
    if (handle.dx < 0) x1 = px;
    if (handle.dx > 0) x2 = px;
    if (handle.dy < 0) y1 = py;
    if (handle.dy > 0) y2 = py;
    box.x = Math.min(x1, x2);
    box.y = Math.min(y1, y2);
    box.w = Math.abs(x2 - x1);
    box.h = Math.abs(y2 - y1);
}

canvas.addEventListener("pointerup", (e) => {
    if (!drag) return;
    const info = currentImage();

    if (drag.mode === "draw" && drag.rect && info) {
        if (drag.rect.w >= MIN_BOX_SIZE && drag.rect.h >= MIN_BOX_SIZE) {
            const boxes = currentBoxes();
            boxes.push({ label: state.activeLabel, ...drag.rect });
            state.selection = boxes.length - 1;
            markChanged();
            renderLabels();
            renderImages();
        }
    } else if (
        (drag.mode === "move" || drag.mode === "resize") &&
        drag.moved
    ) {
        markChanged();
    }
    drag = null;
    hover = info ? hitTest(e.offsetX, e.offsetY) : { box: -1, handle: -1 };
    updateCursor();
    render();
});

canvas.addEventListener("pointerleave", () => {
    pointer.inside = false;
    hover = { box: -1, handle: -1 };
    render();
});

canvas.addEventListener("contextmenu", (e) => {
    e.preventDefault();
    if (!currentImage()) return;
    const hit = hitTest(e.offsetX, e.offsetY);
    if (hit.box >= 0) deleteBox(hit.box);
});

function deleteBox(index) {
    const boxes = currentBoxes();
    boxes.splice(index, 1);
    if (state.selection === index) state.selection = -1;
    else if (state.selection > index) state.selection -= 1;
    markChanged();
    renderLabels();
    renderImages();
    render();
}

function updateCursor() {
    if (spaceDown || (drag && drag.mode === "pan")) {
        canvas.style.cursor = "grab";
    } else if (hover.handle >= 0) {
        const boxes = currentBoxes();
        canvas.style.cursor = handlePositions(boxes[hover.box])[
            hover.handle
        ].cursor;
    } else if (hover.box >= 0) {
        canvas.style.cursor = "move";
    } else if (state.activeLabel && currentImage()) {
        canvas.style.cursor = "crosshair";
    } else {
        canvas.style.cursor = "default";
    }
}

/* ---------- keyboard ---------- */

document.addEventListener("keydown", (e) => {
    const tag = document.activeElement && document.activeElement.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA") return;
    if (document.querySelector("dialog[open]")) return;

    if (e.code === "Space") {
        spaceDown = true;
        updateCursor();
        e.preventDefault();
    } else if (e.key === "Escape") {
        if (drag && drag.mode === "draw") drag = null;
        state.selection = -1;
        render();
    } else if (e.key === "Delete" || e.key === "Backspace") {
        if (state.selection >= 0) {
            deleteBox(state.selection);
            e.preventDefault();
        }
    } else if (e.key === "ArrowLeft") {
        $("prevBtn").click();
    } else if (e.key === "ArrowRight") {
        $("nextBtn").click();
    }
});

document.addEventListener("keyup", (e) => {
    if (e.code === "Space") {
        spaceDown = false;
        updateCursor();
    }
});

/* ---------- rendering ---------- */

function render() {
    const rect = stage.getBoundingClientRect();
    ctx.clearRect(0, 0, rect.width, rect.height);
    const info = currentImage();
    if (!info) return;

    const img = imageCache.get(info.name);
    const { scale, x: ox, y: oy } = state.view;

    if (img && img.complete && img.naturalWidth) {
        ctx.imageSmoothingEnabled = scale < 4;
        ctx.drawImage(img, ox, oy, info.width * scale, info.height * scale);
    }
    ctx.strokeStyle = "rgba(255,255,255,0.25)";
    ctx.lineWidth = 1;
    ctx.strokeRect(ox, oy, info.width * scale, info.height * scale);

    const boxes = currentBoxes();
    boxes.forEach((box, idx) => {
        drawBox(box, idx === state.selection, idx === hover.box);
    });

    if (drag && drag.mode === "draw" && drag.rect) {
        drawBox({ label: state.activeLabel, ...drag.rect }, false, false);
    }

    // Crosshair guides while drawing is possible.
    const drawing = drag && drag.mode === "draw";
    const idle = !drag && hover.box < 0 && state.activeLabel && !spaceDown;
    if (pointer.inside && (drawing || idle)) {
        ctx.save();
        ctx.strokeStyle = "rgba(230,233,239,0.35)";
        ctx.lineWidth = 1;
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        ctx.moveTo(pointer.x, 0);
        ctx.lineTo(pointer.x, rect.height);
        ctx.moveTo(0, pointer.y);
        ctx.lineTo(rect.width, pointer.y);
        ctx.stroke();
        ctx.restore();
    }
}

function drawBox(box, selected, hovered) {
    const { scale, x: ox, y: oy } = state.view;
    const x = box.x * scale + ox;
    const y = box.y * scale + oy;
    const w = box.w * scale;
    const h = box.h * scale;
    const color = labelColor(box.label);

    ctx.save();
    ctx.fillStyle = hexToRgba(color, selected || hovered ? 0.22 : 0.12);
    ctx.fillRect(x, y, w, h);
    ctx.strokeStyle = color;
    ctx.lineWidth = selected ? 2.5 : 2;
    ctx.setLineDash(selected ? [] : [6, 4]);
    ctx.strokeRect(x, y, w, h);
    ctx.setLineDash([]);

    // Label tag.
    const text = box.label || "";
    ctx.font = "11px " + getComputedStyle(document.body).fontFamily;
    const tw = ctx.measureText(text).width + 10;
    const ty = y - 17 < 0 ? y : y - 17;
    ctx.fillStyle = color;
    ctx.fillRect(x, ty, tw, 17);
    ctx.fillStyle = "#fff";
    ctx.fillText(text, x + 5, ty + 12);

    if (selected) {
        for (const handle of handlePositions(box)) {
            ctx.fillStyle = "#fff";
            ctx.strokeStyle = color;
            ctx.lineWidth = 1.5;
            ctx.fillRect(
                handle.x - HANDLE_SIZE / 2,
                handle.y - HANDLE_SIZE / 2,
                HANDLE_SIZE,
                HANDLE_SIZE,
            );
            ctx.strokeRect(
                handle.x - HANDLE_SIZE / 2,
                handle.y - HANDLE_SIZE / 2,
                HANDLE_SIZE,
                HANDLE_SIZE,
            );
        }
    }
    ctx.restore();
}

function hexToRgba(hex, alpha) {
    const value = parseInt(hex.slice(1), 16);
    const r = (value >> 16) & 255;
    const g = (value >> 8) & 255;
    const b = value & 255;
    return `rgba(${r},${g},${b},${alpha})`;
}

/* ---------- dialogs ---------- */

for (const dialog of document.querySelectorAll("dialog")) {
    const closeBtn = dialog.querySelector("[data-close]");
    if (closeBtn) closeBtn.onclick = () => dialog.close();
}

$("exportBtn").onclick = () => $("exportDialog").showModal();
$("synthBtn").onclick = () => $("synthDialog").showModal();

$("exportForm").onsubmit = async (e) => {
    e.preventDefault();
    const form = e.target.elements;
    const submitBtn = e.target.querySelector("button[type=submit]");
    submitBtn.disabled = true;
    await syncSession();
    showProgress("Exporting dataset…");
    startProgressPoll("Exporting dataset");
    try {
        const result = await api("/api/export", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                name: form.name.value.trim() || "yolo_dataset",
                train_split: Number(form.train.value) / 100,
                shuffle: form.shuffle.checked,
                seed: form.seed.value === "" ? null : Number(form.seed.value),
                output_dir: form.output.value.trim() || null,
            }),
        });
        $("exportDialog").close();
        toast(
            `Dataset exported (${result.num_train} train / ` +
                `${result.num_val} val).`,
            "ok",
            result.yaml,
            12000,
        );
    } catch (err) {
        toast(`Export failed: ${err.message}`, "error");
    } finally {
        hideProgress();
        submitBtn.disabled = false;
    }
};

$("synthForm").onsubmit = async (e) => {
    e.preventDefault();
    const form = e.target.elements;
    const submitBtn = e.target.querySelector("button[type=submit]");
    submitBtn.disabled = true;
    await syncSession();
    showProgress("Synthesizing dataset…");
    startProgressPoll("Synthesizing dataset");
    try {
        const result = await api("/api/synthesize", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                name: form.name.value.trim() || "synt_dataset",
                num_images: Number(form.num.value),
                width: Number(form.width.value),
                height: Number(form.height.value),
                per_image: Number(form.per.value),
                train_split: Number(form.train.value) / 100,
                scale_min: Number(form.smin.value),
                scale_max: Number(form.smax.value),
                background: form.background.value,
                output_dir: form.output.value.trim() || null,
            }),
        });
        $("synthDialog").close();
        toast(
            `Synthetic dataset created (${result.num_train} train / ` +
                `${result.num_val} val).`,
            "ok",
            result.yaml,
            12000,
        );
    } catch (err) {
        toast(`Synthesis failed: ${err.message}`, "error");
    } finally {
        hideProgress();
        submitBtn.disabled = false;
    }
};

/* ---------- session save / load ---------- */

$("saveSessionBtn").onclick = () => {
    quitAfterSave = false;
    openSaveDialog();
};

function openSaveDialog() {
    const input = $("saveForm").elements.path;
    if (sessionPath && !input.value) input.value = sessionPath;
    $("saveDialog").showModal();
}

$("saveForm").onsubmit = async (e) => {
    e.preventDefault();
    const path = e.target.elements.path.value.trim();
    if (!path) return;
    try {
        const result = await api("/api/session", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                path,
                labels: state.labels,
                annotations: state.annotations,
            }),
        });
        sessionPath = result.path;
        dirty = false;
        setSaveStatus();
        $("saveDialog").close();
        toast("Session saved.", "ok", result.path);
        if (quitAfterSave) {
            quitAfterSave = false;
            await shutdownServer();
        }
    } catch (err) {
        toast(`Save failed: ${err.message}`, "error");
    }
};

$("loadSessionBtn").onclick = () => {
    const input = $("loadForm").elements.path;
    if (sessionPath && !input.value) input.value = sessionPath;
    $("loadDialog").showModal();
};

$("loadForm").onsubmit = async (e) => {
    e.preventDefault();
    const path = e.target.elements.path.value.trim();
    if (!path) return;
    if (
        dirty &&
        !confirm("Loading a session discards unsaved changes. Continue?")
    ) {
        return;
    }
    try {
        const session = await api("/api/session/load", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ path }),
        });
        state.labels = session.labels || [];
        state.annotations = session.annotations || {};
        state.activeLabel = null;
        state.selection = -1;
        sessionPath = path;
        dirty = false;
        setSaveStatus();
        $("loadDialog").close();
        renderLabels();
        renderImages();
        render();
        toast("Session loaded.", "ok", path);
    } catch (err) {
        toast(`Load failed: ${err.message}`, "error");
    }
};

/* ---------- quit ---------- */

async function shutdownServer() {
    dirty = false; // suppress the beforeunload warning
    try {
        await api("/api/shutdown", { method: "POST" });
    } catch {
        /* server is going down */
    }
    document.body.innerHTML =
        '<div id="emptyState"><p><strong>Server stopped.</strong></p>' +
        "<p>You can close this tab.</p></div>";
}

$("quitBtn").onclick = () => {
    if (dirty) $("quitDialog").showModal();
    else shutdownServer();
};

$("quitDiscardBtn").onclick = () => {
    $("quitDialog").close();
    shutdownServer();
};

$("quitSaveBtn").onclick = () => {
    $("quitDialog").close();
    quitAfterSave = true;
    openSaveDialog();
};

/* ---------- init ---------- */

async function init() {
    resizeCanvas();
    try {
        const data = await api("/api/state");
        state.workspace = data.workspace;
        state.images = data.images;
        state.labels = data.labels;
        state.annotations = data.annotations;
        sessionPath = data.session_path;
        dirty = Boolean(data.dirty);
        $("workspacePath").textContent = data.workspace;
        setSaveStatus();
        renderLabels();
        renderImages();
        if (state.images.length) selectImage(0, true);
    } catch (err) {
        toast(`Could not load session: ${err.message}`, "error");
    }
}

window.addEventListener("resize", resizeCanvas);
new ResizeObserver(resizeCanvas).observe(stage);
init();
