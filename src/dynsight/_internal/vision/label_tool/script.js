// script.js

const imageInput = document.getElementById("imageInput");
const imageDisplay = document.getElementById("imageDisplay");
const imageContainer = document.getElementById("imageContainer");
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
  imageDisplay.src = url;
  clearBoxes();
  const name = images[index].name;
  if (annotations[name]) annotations[name].forEach(addBoxFromData);
  else annotations[name] = [];
}

function clearBoxes() {
  document.querySelectorAll(".bounding-box").forEach((el) => el.remove());
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
  imageContainer.appendChild(box);
}

imageContainer.onmousedown = (e) => {
  if (!currentLabel || !images[currentIndex]) return;
  const rect = imageContainer.getBoundingClientRect();
  startX = e.clientX - rect.left;
  startY = e.clientY - rect.top;
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
  imageContainer.appendChild(box);
  isDrawing = true;
};

imageContainer.onmousemove = (e) => {
  const rect = imageContainer.getBoundingClientRect();
  const currX = e.clientX - rect.left;
  const currY = e.clientY - rect.top;

  verticalLine.style.left = `${currX}px`;
  horizontalLine.style.top = `${currY}px`;

  if (!isDrawing || !box) return;
  box.style.left = `${Math.min(currX, startX)}px`;
  box.style.top = `${Math.min(currY, startY)}px`;
  box.style.width = `${Math.abs(currX - startX)}px`;
  box.style.height = `${Math.abs(currY - startY)}px`;
};

imageContainer.onmouseup = () => {
  if (!isDrawing || !box) return;
  const rect = box.getBoundingClientRect();
  const imgRect = imageDisplay.getBoundingClientRect();
  annotations[images[currentIndex].name].push({
    label: currentLabel,
    left: rect.left - imgRect.left,
    top: rect.top - imgRect.top,
    width: rect.width,
    height: rect.height,
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
  const imgRect = imageDisplay.getBoundingClientRect();
  const iw = imageDisplay.naturalWidth;
  const ih = imageDisplay.naturalHeight;
  const annots = annotations[img.name] || [];
  const labelMap = {};
  let nextId = 0;
  let txt = "";
  annots.forEach((obj) => {
    if (!(obj.label in labelMap)) labelMap[obj.label] = nextId++;
    const x = (obj.left / imgRect.width) * iw;
    const y = (obj.top / imgRect.height) * ih;
    const w = (obj.width / imgRect.width) * iw;
    const h = (obj.height / imgRect.height) * ih;
    const cx = (x + w / 2) / iw;
    const cy = (y + h / 2) / ih;
    txt += `${labelMap[obj.label]} ${cx.toFixed(6)} ${cy.toFixed(6)} ${(w / iw).toFixed(6)} ${(h / ih).toFixed(6)}\n`;
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
    const rect = imageDisplay.getBoundingClientRect();
    const annots = annotations[name] || [];
    let txt = "";
    annots.forEach((obj) => {
      if (!(obj.label in labelMap)) labelMap[obj.label] = nextId++;
      const x = (obj.left / rect.width) * iw;
      const y = (obj.top / rect.height) * ih;
      const w = (obj.width / rect.width) * iw;
      const h = (obj.height / rect.height) * ih;
      const cx = (x + w / 2) / iw;
      const cy = (y + h / 2) / ih;
      txt += `${labelMap[obj.label]} ${cx.toFixed(6)} ${cy.toFixed(6)} ${(w / iw).toFixed(6)} ${(h / ih).toFixed(6)}\n`;
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
