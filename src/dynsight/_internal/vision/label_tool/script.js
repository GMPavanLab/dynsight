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
const nextImageBtn = document.getElementById("nextImage");
const prevImageBtn = document.getElementById("prevImage");
const verticalLine = document.getElementById("verticalLine");
const horizontalLine = document.getElementById("horizontalLine");
const zoomSlider = document.getElementById("zoomSlider");

let zoomLevel = 1;
let panX = 0;
let isPanning = false;
let panStart = 0;

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
  if (isPanning) {
    isPanning = false;
    imageContainer.style.cursor = "crosshair";
  }
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
    imageWrapper.style.width = `${iw}px`;
    imageWrapper.style.height = `${ih}px`;
    overlay.style.width = `${iw}px`;
    overlay.style.height = `${ih}px`;
    zoomLevel = 1;
    panX = 0;
    zoomSlider.value = 100;
    const name = images[index].name;
    if (!annotations[name]) annotations[name] = [];
    updateTransform();
    clearBoxes();
    annotations[name].forEach(addBoxFromData);
  };
  imageDisplay.src = url;
}

zoomSlider.oninput = (e) => {
  zoomLevel = e.target.value / 100;
  updateTransform();
};

function clearBoxes() {
  overlay.innerHTML = "";
}

function updateTransform() {
  imageWrapper.style.transform = `translateX(${panX}px) scale(${zoomLevel})`;
}

function addBoxFromData(data) {
  const box = document.createElement("div");
  box.className = "bounding-box";
  box.style.left = `${data.left}px`;
  box.style.top = `${data.top}px`;
  box.style.width = `${data.width}px`;
  box.style.height = `${data.height}px`;
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
    isPanning = true;
    panStart = e.clientX;
    imageContainer.style.cursor = "grabbing";
    e.preventDefault();
    return;
  }

  if (!currentLabel || !images[currentIndex]) return;

  const rect = imageDisplay.getBoundingClientRect();
  startX = (e.clientX - rect.left) / zoomLevel;
  startY = (e.clientY - rect.top) / zoomLevel;

  box = document.createElement("div");
  box.className = "bounding-box";
  box.style.left = `${startX}px`;
  box.style.top = `${startY}px`;
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

  verticalLine.style.left = `${e.clientX - containerRect.left}px`;
  horizontalLine.style.top = `${e.clientY - containerRect.top}px`;

  if (isPanning) {
    const dx = e.clientX - panStart;
    panStart = e.clientX;
    panX += dx;
    updateTransform();
    return;
  }

  if (!isDrawing || !box) return;

  box.style.left = `${Math.min(currX, startX)}px`;
  box.style.top = `${Math.min(currY, startY)}px`;
  box.style.width = `${Math.abs(currX - startX)}px`;
  box.style.height = `${Math.abs(currY - startY)}px`;
};

imageContainer.onmouseup = (e) => {
  if (isPanning) {
    isPanning = false;
    imageContainer.style.cursor = "crosshair";
    return;
  }

  if (!isDrawing || !box) return;

  const imgRect = imageDisplay.getBoundingClientRect();
  const endX = (e.clientX - imgRect.left) / zoomLevel;
  const endY = (e.clientY - imgRect.top) / zoomLevel;

  const left = Math.min(startX, endX);
  const top = Math.min(startY, endY);
  const width = Math.abs(endX - startX);
  const height = Math.abs(endY - startY);

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
    txt += `${labelMap[obj.label]} ${cx.toFixed(6)} ${cy.toFixed(6)} ${w.toFixed(6)} ${h.toFixed(6)}\n`;
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
    alert("Nessuna immagine caricata.");
    return;
  }
  const zip = new JSZip();
  const imgFolder = zip.folder("images");
  const lblFolder = zip.folder("labels");
  const labelMap = {};
  let nextId = 0;
  for (const image of images) {
    const name = image.name;
    const imgData = await image.arrayBuffer();
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
      txt += `${labelMap[obj.label]} ${cx.toFixed(6)} ${cy.toFixed(6)} ${w.toFixed(6)} ${h.toFixed(6)}\n`;
    });
    const labelFileName = name.replace(/\.[^/.]+$/, "") + ".txt";
    lblFolder.file(labelFileName, txt);
  }
  const content = await zip.generateAsync({ type: "blob" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(content);
  a.download = "yolo_dataset.zip";
  a.click();
  URL.revokeObjectURL(a.href);
};

window.addEventListener("unload", () => {
  navigator.sendBeacon("/shutdown");
});
