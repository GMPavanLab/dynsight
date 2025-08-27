// script.js

const imageInput = document.getElementById("imageInput");
const imageDisplay = document.getElementById("imageDisplay");
const imageContainer = document.getElementById("imageContainer");
const imageWrapper = document.getElementById("imageWrapper");
const labelList = document.getElementById("labelList");
const addLabelBtn = document.getElementById("addLabelBtn");
const newLabelInput = document.getElementById("newLabelInput");
const clearLastBtn = document.getElementById("clearLast");
const clearAllBtn = document.getElementById("clearAll");
const exportBtn = document.getElementById("exportYolo");
const exportAllBtn = document.getElementById("exportAll");
const synthBtn = document.getElementById("synthesize");
const nextImageBtn = document.getElementById("nextImage");
const prevImageBtn = document.getElementById("prevImage");
const verticalLine = document.getElementById("verticalLine");
const horizontalLine = document.getElementById("horizontalLine");
const zoomSlider = document.getElementById("zoomSlider");

let zoomLevel = 1;
let baseZoom = 1;
let naturalWidth = 0;
let naturalHeight = 0;

const overlay = document.getElementById("overlay");

verticalLine.style.display = "none";
horizontalLine.style.display = "none";

imageContainer.onmouseenter = () => {
    verticalLine.style.display = "block";
    horizontalLine.style.display = "block";
};

imageContainer.onmouseleave = () => {
    verticalLine.style.display = "none";
    horizontalLine.style.display = "none";
};

let currentLabel = null;
const labelColors = {};
let isDrawing = false;
let startX,
    startY,
    box = null;

let images = [];
let currentIndex = 0;
const annotations = {}; // imageName -> [boxData]

function getRandomColor() {
    const hue = Math.floor(Math.random() * 360);
    return `hsl(${hue}, 90%, 50%)`;
}

function setActiveLabel(item) {
    document
        .querySelectorAll(".label-item")
        .forEach((i) => i.classList.remove("active"));
    item.classList.add("active");
    currentLabel = item.textContent;
}

function createLabelItem(text) {
    const item = document.createElement("div");
    item.className = "label-item";
    item.textContent = text;
    labelColors[text] = labelColors[text] || getRandomColor();
    item.style.backgroundColor = labelColors[text];
    item.style.color = "#fff";
    item.addEventListener("click", () => setActiveLabel(item));
    labelList.appendChild(item);
}

addLabelBtn.onclick = () => {
    const label = newLabelInput.value.trim();
    if (label && !labelColors[label]) {
        createLabelItem(label);
        newLabelInput.value = "";
    }
};

imageInput.onchange = (e) => {
    images = Array.from(e.target.files);
    currentIndex = 0;
    loadImage(currentIndex);
};

function loadImage(index) {
    if (!images[index]) return;
    const url = URL.createObjectURL(images[index]);
    imageDisplay.onload = () => {
        const iw = imageDisplay.naturalWidth;
        const ih = imageDisplay.naturalHeight;
        imageDisplay.style.width = `${iw}px`;
        imageDisplay.style.height = `${ih}px`;
        naturalWidth = iw;
        naturalHeight = ih;
        baseZoom = Math.min(
            imageContainer.clientWidth / iw,
            imageContainer.clientHeight / ih,
            1,
        );
        zoomLevel = baseZoom;
        zoomSlider.value = 100;
        updateTransform();
        const name = images[index].name;
        if (!annotations[name]) annotations[name] = [];
        clearBoxes();
        annotations[name].forEach(addBoxFromData);
    };
    imageDisplay.src = url;
}

zoomSlider.oninput = (e) => {
    zoomLevel = (e.target.value / 100) * baseZoom;
    updateTransform();
    clearBoxes();
    annotations[images[currentIndex].name].forEach(addBoxFromData);
};

function clearBoxes() {
    overlay.innerHTML = "";
}

function updateTransform() {
    const w = naturalWidth * zoomLevel;
    const h = naturalHeight * zoomLevel;
    imageDisplay.style.width = `${w}px`;
    imageDisplay.style.height = `${h}px`;
    imageWrapper.style.width = `${w}px`;
    imageWrapper.style.height = `${h}px`;
    overlay.style.width = `${w}px`;
    overlay.style.height = `${h}px`;
}

function addBoxFromData(data) {
    const box = document.createElement("div");
    box.className = "bounding-box";
    box.style.left = `${data.left * zoomLevel}px`;
    box.style.top = `${data.top * zoomLevel}px`;
    box.style.width = `${data.width * zoomLevel}px`;
    box.style.height = `${data.height * zoomLevel}px`;
    box.style.border = `2px dashed ${labelColors[data.label]}`;
    box.style.backgroundColor = labelColors[data.label]
        .replace("hsl", "hsla")
        .replace(")", ", 0.1)");

    const tag = document.createElement("div");
    tag.className = "label-tag";
    tag.textContent = data.label;
    tag.style.backgroundColor = labelColors[data.label];
    box.appendChild(tag);

    overlay.appendChild(box);
}

imageContainer.onmousedown = (e) => {
    if (e.button !== 0) {
        return;
    }

    if (!currentLabel || !images[currentIndex]) return;

    const rect = imageDisplay.getBoundingClientRect();
    startX = (e.clientX - rect.left) / zoomLevel;
    startY = (e.clientY - rect.top) / zoomLevel;

    // Ignore clicks started outside the image boundaries
    if (
        startX < 0 ||
        startY < 0 ||
        startX > naturalWidth ||
        startY > naturalHeight
    ) {
        isDrawing = false;
        return;
    }

    box = document.createElement("div");
    box.className = "bounding-box";
    box.style.left = `${startX * zoomLevel}px`;
    box.style.top = `${startY * zoomLevel}px`;
    box.style.border = `2px dashed ${labelColors[currentLabel]}`;
    box.style.backgroundColor = labelColors[currentLabel]
        .replace("hsl", "hsla")
        .replace(")", ", 0.1)");

    const tag = document.createElement("div");
    tag.className = "label-tag";
    tag.textContent = currentLabel;
    tag.style.backgroundColor = labelColors[currentLabel];
    box.appendChild(tag);

    overlay.appendChild(box);
    isDrawing = true;
};

imageContainer.onmousemove = (e) => {
    const imgRect = imageDisplay.getBoundingClientRect();
    const containerRect = imageContainer.getBoundingClientRect();

    const currX = (e.clientX - imgRect.left) / zoomLevel;
    const currY = (e.clientY - imgRect.top) / zoomLevel;
    const clampedX = Math.max(0, Math.min(naturalWidth, currX));
    const clampedY = Math.max(0, Math.min(naturalHeight, currY));

    verticalLine.style.left = `${
        e.clientX - containerRect.left + imageContainer.scrollLeft
    }px`;
    horizontalLine.style.top = `${
        e.clientY - containerRect.top + imageContainer.scrollTop
    }px`;

    if (!isDrawing || !box) return;

    box.style.left = `${Math.min(clampedX, startX) * zoomLevel}px`;
    box.style.top = `${Math.min(clampedY, startY) * zoomLevel}px`;
    box.style.width = `${Math.abs(clampedX - startX) * zoomLevel}px`;
    box.style.height = `${Math.abs(clampedY - startY) * zoomLevel}px`;
};

imageContainer.onmouseup = (e) => {
    if (!isDrawing || !box) return;

    const imgRect = imageDisplay.getBoundingClientRect();
    const endX = (e.clientX - imgRect.left) / zoomLevel;
    const endY = (e.clientY - imgRect.top) / zoomLevel;
    const clampedX = Math.max(0, Math.min(naturalWidth, endX));
    const clampedY = Math.max(0, Math.min(naturalHeight, endY));

    const left = Math.min(startX, clampedX);
    const top = Math.min(startY, clampedY);
    const width = Math.abs(clampedX - startX);
    const height = Math.abs(clampedY - startY);

    annotations[images[currentIndex].name].push({
        label: currentLabel,
        left,
        top,
        width,
        height,
    });

    box = null;
    isDrawing = false;
};

clearLastBtn.onclick = () => {
    const ann = annotations[images[currentIndex].name];
    if (ann.length > 0) {
        ann.pop();
        clearBoxes();
        ann.forEach(addBoxFromData);
    }
};

clearAllBtn.onclick = () => {
    annotations[images[currentIndex].name] = [];
    clearBoxes();
};

prevImageBtn.onclick = () => {
    if (currentIndex > 0) {
        currentIndex--;
        loadImage(currentIndex);
    }
};

nextImageBtn.onclick = () => {
    if (currentIndex < images.length - 1) {
        currentIndex++;
        loadImage(currentIndex);
    }
};

exportBtn.onclick = () => {
    const img = images[currentIndex];
    if (!img) return;
    const iw = imageDisplay.naturalWidth;
    const ih = imageDisplay.naturalHeight;
    const annots = annotations[img.name] || [];
    const labelMap = {};
    let nextId = 0;
    let txt = "";
    annots.forEach((obj) => {
        if (!(obj.label in labelMap)) labelMap[obj.label] = nextId++;
        const cx = (obj.left + obj.width / 2) / iw;
        const cy = (obj.top + obj.height / 2) / ih;
        const w = obj.width / iw;
        const h = obj.height / ih;
        txt +=
            labelMap[obj.label] +
            " " +
            cx.toFixed(6) +
            " " +
            cy.toFixed(6) +
            " " +
            w.toFixed(6) +
            " " +
            h.toFixed(6) +
            "\n";
    });
    const blob = new Blob([txt], { type: "text/plain" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = img.name.replace(/\.[^/.]+$/, "") + ".txt";
    a.click();
    URL.revokeObjectURL(a.href);
};

exportAllBtn.onclick = async () => {
    if (images.length === 0) {
        alert("No images uploaded.");
        return;
    }
    let trainPercent = parseFloat(
        prompt("Percentage of images for training?", "80"),
    );
    if (
        Number.isNaN(trainPercent) ||
        trainPercent <= 0 ||
        trainPercent >= 100
    ) {
        trainPercent = 80;
    }
    const numTrain = Math.floor(images.length * (trainPercent / 100));
    const zip = new JSZip();
    const imgTrain = zip.folder("images/train");
    const imgVal = zip.folder("images/val");
    const lblTrain = zip.folder("labels/train");
    const lblVal = zip.folder("labels/val");
    const labelMap = {};
    let nextId = 0;
    for (let i = 0; i < images.length; i++) {
        const image = images[i];
        const name = image.name;
        const imgData = await image.arrayBuffer();
        const imgFolder = i < numTrain ? imgTrain : imgVal;
        const lblFolder = i < numTrain ? lblTrain : lblVal;
        imgFolder.file(name, imgData);

        const img = new Image();
        const url = URL.createObjectURL(image);
        img.src = url;
        await new Promise((resolve) => (img.onload = resolve));
        const iw = img.naturalWidth;
        const ih = img.naturalHeight;
        const annots = annotations[name] || [];
        let txt = "";
        annots.forEach((obj) => {
            if (!(obj.label in labelMap)) labelMap[obj.label] = nextId++;
            const cx = (obj.left + obj.width / 2) / iw;
            const cy = (obj.top + obj.height / 2) / ih;
            const w = obj.width / iw;
            const h = obj.height / ih;
            txt +=
                labelMap[obj.label] +
                " " +
                cx.toFixed(6) +
                " " +
                cy.toFixed(6) +
                " " +
                w.toFixed(6) +
                " " +
                h.toFixed(6) +
                "\n";
        });
        const labelFileName = name.replace(/\.[^/.]+$/, "") + ".txt";
        lblFolder.file(labelFileName, txt);
    }

    const names = Object.keys(labelMap);
    const yaml = `path: .
    train: images/train
    val: images/val
    nc: ${names.length}
    names: [${names.map((n) => `'${n}'`).join(", ")}]
    `;
    zip.file("dataset.yaml", yaml);

    const content = await zip.generateAsync({ type: "blob" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(content);
    a.download = "yolo_dataset.zip";
    a.click();
    URL.revokeObjectURL(a.href);
};

synthBtn.onclick = async () => {
    if (images.length === 0) {
        alert("No images uploaded.");
        return;
    }

    const numImages = parseInt(
        prompt("Number of synthetic images to generate?", "10"),
        10,
    );
    const width = parseInt(prompt("Image width?", "640"), 10);
    const height = parseInt(prompt("Image height?", "640"), 10);

    if (
        !numImages ||
        Number.isNaN(numImages) ||
        !width ||
        Number.isNaN(width) ||
        !height ||
        Number.isNaN(height)
    ) {
        alert("Invalid parameters.");
        return;
    }

    const crops = [];
    const labelMap = {};
    let nextId = 0;
    for (const file of images) {
        const ann = annotations[file.name] || [];
        for (const c of ann) {
            crops.push({ file, ...c });
            if (!(c.label in labelMap)) labelMap[c.label] = nextId++;
        }
    }

    if (crops.length === 0) {
        alert("No label found.");
        return;
    }

    async function loadImage(file) {
        return await new Promise((resolve) => {
            const img = new Image();
            img.src = URL.createObjectURL(file);
            img.onload = () => {
                URL.revokeObjectURL(img.src);
                resolve(img);
            };
        });
    }

    function overlaps(x, y, w, h, boxes) {
        return boxes.some((b) => {
            return !(
                x + w <= b.x ||
                x >= b.x + b.w ||
                y + h <= b.y ||
                y >= b.y + b.h
            );
        });
    }

    async function createCollage() {
        const canvas = document.createElement("canvas");
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext("2d");
        ctx.fillStyle = "white";
        ctx.fillRect(0, 0, width, height);

        const placed = [];
        const numObj = Math.min(5, crops.length);
        for (let i = 0; i < numObj; i++) {
            const crop = crops[Math.floor(Math.random() * crops.length)];
            const img = await loadImage(crop.file);
            const c = document.createElement("canvas");
            c.width = crop.width;
            c.height = crop.height;
            c.getContext("2d").drawImage(
                img,
                crop.left,
                crop.top,
                crop.width,
                crop.height,
                0,
                0,
                crop.width,
                crop.height,
            );

            const scale =
                (0.15 + 0.15 * Math.random()) *
                (Math.min(width, height) / Math.max(crop.width, crop.height));
            const w = crop.width * scale;
            const h = crop.height * scale;

            let x, y;
            let tries = 0;
            do {
                x = Math.random() * (width - w);
                y = Math.random() * (height - h);
                tries += 1;
            } while (tries < 50 && overlaps(x, y, w, h, placed));

            if (tries === 50) continue;

            ctx.drawImage(c, 0, 0, crop.width, crop.height, x, y, w, h);
            placed.push({ label: crop.label, x, y, w, h });
        }

        let txt = "";
        placed.forEach((p) => {
            const cls = labelMap[p.label];
            const cx = (p.x + p.w / 2) / width;
            const cy = (p.y + p.h / 2) / height;
            const ww = p.w / width;
            const hh = p.h / height;
            txt += `${cls} ${cx.toFixed(6)} ${cy.toFixed(6)} ${ww.toFixed(
                6,
            )} ${hh.toFixed(6)}\n`;
        });

        const blob = await new Promise((resolve) =>
            canvas.toBlob((b) => resolve(b), "image/jpeg"),
        );
        return { blob, txt };
    }

    const zip = new JSZip();
    const imgTrain = zip.folder("images/train");
    const imgVal = zip.folder("images/val");
    const lblTrain = zip.folder("labels/train");
    const lblVal = zip.folder("labels/val");

    const numTrain = Math.floor(numImages * 0.8);

    for (let i = 0; i < numImages; i++) {
        const { blob, txt } = await createCollage();
        const imgName = `synt_${i}.jpg`;
        const txtName = `synt_${i}.txt`;
        if (i < numTrain) {
            imgTrain.file(imgName, blob);
            lblTrain.file(txtName, txt);
        } else {
            imgVal.file(imgName, blob);
            lblVal.file(txtName, txt);
        }
    }

    const names = Object.keys(labelMap);
    const yaml = `path: .
    train: images/train
    val: images/val
    nc: ${names.length}
    names: [${names.map((n) => `'${n}'`).join(", ")}]
    `;
    zip.file("dataset.yaml", yaml);

    const content = await zip.generateAsync({ type: "blob" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(content);
    a.download = "synt_dataset.zip";
    a.click();
    URL.revokeObjectURL(a.href);
};

let navigatingAway = false;
document.addEventListener("click", (e) => {
    const link = e.target.closest("a");
    if (link && link.href) {
        navigatingAway = true;
    }
});

window.addEventListener("pagehide", () => {
    if (!navigatingAway) {
        navigator.sendBeacon("/shutdown");
    }
});
