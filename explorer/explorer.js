const el = id => document.getElementById(id);
const canvas = el("map");
const ctx = canvas.getContext("2d");
const slider = el("breadth");
let dataset, result, target, request, geometry;
const colors = {
  stored: "#83958a", examined: "#78bcad", found: "#c3f17e",
  exact: "#f2f5db", query: "#ff997b", ink: "#172622",
};

function setStatus(state, label) {
  el("status").dataset.state = state;
  el("status").lastElementChild.textContent = label;
}

async function readJSON(url, signal) {
  const response = await fetch(url, {signal});
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

function circle(x, y, radius) {
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, 2 * Math.PI);
}

function line(x0, y0, x1, y1) {
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x1, y1);
  ctx.stroke();
}

function draw() {
  if (!dataset) return;
  const {width, height} = canvas.getBoundingClientRect();
  if (!width || !height) return;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  ctx.setTransform(canvas.width / width, 0, 0, canvas.height / height, 0, 0);

  // The same number of pixels per unit on both axes keeps L2 circles round.
  const [maxX, maxY] = dataset.domain;
  const scale = Math.min((width - 50) / maxX, (height - 42) / maxY);
  const left = 30 + (width - 50 - maxX * scale) / 2;
  const bottom = height - 25 - (height - 42 - maxY * scale) / 2;
  geometry = {left, bottom, scale};
  const px = x => left + scale * x;
  const py = y => bottom - scale * y;

  ctx.font = '10px "IBM Plex Mono", ui-monospace, monospace';
  ctx.lineWidth = 1;
  for (let x = 0; x <= maxX; x += 10) {
    ctx.strokeStyle = x % 20 ? "#304138" : "#3a4b41";
    line(px(x), py(0), px(x), py(maxY));
    if (!(x % 20)) {
      ctx.textAlign = "center";
      ctx.fillStyle = "#a4b6a8";
      ctx.fillText(x, px(x), py(0) + 17);
    }
  }
  for (let y = 0; y <= maxY; y += 10) {
    ctx.strokeStyle = y % 20 ? "#304138" : "#3a4b41";
    line(px(0), py(y), px(maxX), py(y));
    if (!(y % 20)) {
      ctx.textAlign = "right";
      ctx.fillStyle = "#a4b6a8";
      ctx.fillText(y, px(0) - 7, py(y) + 3);
    }
  }

  ctx.save();
  // Clip the radius and edges at the map frame, keeping the axes readable.
  ctx.beginPath();
  ctx.rect(px(0) - 4, py(maxY) - 4, maxX * scale + 8, maxY * scale + 8);
  ctx.clip();
  const positions = dataset.points.map(([x, y]) => [px(x), py(y)]);
  const dotRadius = Math.max(1.7, Math.min(3, scale * .48));
  if (result) {
    circle(px(result.query[0]), py(result.query[1]), result.radius * scale);
    ctx.fillStyle = "rgba(194, 239, 127, .05)";
    ctx.fill();
    ctx.strokeStyle = "rgba(214, 229, 175, .55)";
    ctx.setLineDash([3, 4]);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.strokeStyle = "rgba(127, 192, 160, .39)";
    for (const [source, destination] of result.graph.discovery_edges) {
      line(...positions[source], ...positions[destination]);
    }
  }
  function points(ids, radius, color, outline = false) {
    ctx.fillStyle = ctx.strokeStyle = color;
    for (const id of ids) {
      circle(...positions[id], radius);
      if (outline) ctx.stroke();
      else ctx.fill();
    }
  }
  points(positions.keys(), dotRadius, colors.stored);
  if (result) {
    points(result.graph.examined, dotRadius + .5, colors.examined);
    points(result.graph.ids, dotRadius + 1, colors.found);
    ctx.lineWidth = 1.25;
    points(result.exact, dotRadius + 3, colors.exact, true);

    const [qx, qy] = [px(result.query[0]), py(result.query[1])];
    ctx.save();
    ctx.translate(qx, qy);
    ctx.rotate(Math.PI / 4);
    ctx.fillStyle = colors.query;
    ctx.strokeStyle = colors.ink;
    ctx.lineWidth = 2;
    ctx.fillRect(-4.5, -4.5, 9, 9);
    ctx.strokeRect(-4.5, -4.5, 9, 9);
    ctx.restore();
  }
  ctx.restore();
}

function showResults(data) {
  result = data;
  const truth = new Set(data.exact);
  const recovered = data.graph.ids.filter(id => truth.has(id)).length;
  el("recovered").textContent = recovered;
  el("examined").textContent = data.graph.examined.length;
  el("coverage").style.width = `${100 * data.graph.examined.length / dataset.points.length}%`;
  [...el("recall-meter").children].forEach((segment, i) => segment.classList.toggle("on", i < recovered));
  el("position").textContent = `x ${data.query[0].toFixed(1)}   y ${data.query[1].toFixed(1)}`;
  el("fallback").hidden = !data.graph.exact_fallback;
  draw();
}

function sliderValue() {
  const value = Number(slider.value);
  const [min, max] = dataset.efRange;
  slider.style.setProperty("--fill", `${100 * (value - min) / (max - min)}%`);
  el("ef-value").textContent = value;
  return value;
}

async function search() {
  const ef = sliderValue();
  request?.abort();
  const current = new AbortController();
  request = current;
  setStatus("busy", "Searching");
  el("error").hidden = true;
  try {
    const params = new URLSearchParams({x: target[0], y: target[1], ef});
    const data = await readJSON(`/api/search?${params}`, current.signal);
    if (request !== current) return;
    showResults(data);
    setStatus("ready", "Engine connected");
  } catch (error) {
    if (request !== current || error.name === "AbortError") return;
    result = null;
    draw();
    el("position").textContent = "x —   y —";
    el("recovered").textContent = el("examined").textContent = "—";
    el("coverage").style.width = "0";
    for (const segment of el("recall-meter").children) segment.classList.remove("on");
    el("fallback").hidden = true;
    setStatus("error", "Search unavailable");
    el("error").textContent = "Couldn’t reach the engine. Check the local server, then click the map to retry.";
    el("error").hidden = false;
  }
}

canvas.addEventListener("click", event => {
  if (!geometry) return;
  const box = canvas.getBoundingClientRect();
  const {left, bottom, scale} = geometry;
  const x = (event.clientX - box.left - left) / scale;
  const y = (bottom - (event.clientY - box.top)) / scale;
  const [maxX, maxY] = dataset.domain;
  if (x < 0 || x > maxX || y < 0 || y > maxY) return;
  target = [x, y];
  search();
});

canvas.addEventListener("keydown", event => {
  const steps = {ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowDown: [0, -1], ArrowUp: [0, 1]};
  const move = steps[event.key];
  if (!target || !move) return;
  event.preventDefault();
  target = target.map((x, i) => Math.min(dataset.domain[i], Math.max(0, x + move[i])));
  search();
});
slider.addEventListener("input", search);
new ResizeObserver(draw).observe(canvas);

async function initialize() {
  dataset = await readJSON("/api/dataset");
  target = dataset.initialQuery;
  [slider.min, slider.max] = dataset.efRange;
  slider.value = dataset.initialEf;
  slider.disabled = false;
  el("k").textContent = dataset.k;
  el("total").textContent = dataset.points.length;
  el("recall-meter").replaceChildren(...Array.from({length: dataset.k}, () => document.createElement("i")));
  el("settings").textContent = `seed ${dataset.seed}  ·  M ${dataset.M}  ·  efConstruction ${dataset.efConstruction}`;
  await search();
}
initialize().catch(() => {
  setStatus("error", "Engine offline");
  el("error").textContent = "The dataset could not be loaded. Start the local server and refresh this page.";
  el("error").hidden = false;
});
