const fileInput = document.getElementById("bundle-file");
const loadDemoBtn = document.getElementById("load-demo");
const statusEl = document.getElementById("status");

const nodeCountEl = document.getElementById("node-count");
const edgeCountEl = document.getElementById("edge-count");
const kindCountEl = document.getElementById("kind-count");
const promptCountEl = document.getElementById("prompt-count");
const mapStatsEl = document.getElementById("map-stats");
const selectionBadgeEl = document.getElementById("selection-badge");
const nodeInspectEl = document.getElementById("node-inspect");

const resetCameraBtn = document.getElementById("reset-camera");
const clearSelectionBtn = document.getElementById("clear-selection");
const findBtn = document.getElementById("find-btn");
const searchInput = document.getElementById("node-search");
const complexitySelect = document.getElementById("complexity-select");
const kindTogglesEl = document.getElementById("kind-toggles");
const decisionSelectEl = document.getElementById("decision-select");

const whySummaryEl = document.getElementById("why-summary");
const supportListEl = document.getElementById("support-list");
const opposeListEl = document.getElementById("oppose-list");
const pathsListEl = document.getElementById("paths-list");
const tokenLinkTitleEl = document.getElementById("token-link-title");
const tokenLinksEl = document.getElementById("token-links");

const prevDecisionBtn = document.getElementById("prev-decision-btn");
const nextDecisionBtn = document.getElementById("next-decision-btn");
const decisionBadgeEl = document.getElementById("decision-badge");
const flowPlayBtn = document.getElementById("flow-play-btn");
const flowStepBtn = document.getElementById("flow-step-btn");
const flowResetBtn = document.getElementById("flow-reset-btn");
const flowSpeedSelect = document.getElementById("flow-speed-select");
const flowProgressEl = document.getElementById("flow-progress");
const flowCaptionEl = document.getElementById("flow-step-caption");
const flowStepsEl = document.getElementById("flow-steps");
const equationTextEl = document.getElementById("equation-text");
const marginChartEl = document.getElementById("margin-chart");
const residPlayBtn = document.getElementById("resid-play-btn");
const residStepBtn = document.getElementById("resid-step-btn");
const residResetBtn = document.getElementById("resid-reset-btn");
const residProgressEl = document.getElementById("resid-progress");
const residCaptionEl = document.getElementById("resid-caption");
const residLayerBarsEl = document.getElementById("resid-layer-bars");
const sankeyCanvasEl = document.getElementById("sankey-canvas");
const sankeyCaptionEl = document.getElementById("sankey-caption");
const counterSummaryEl = document.getElementById("counterfactual-summary");
const counterComponentsEl = document.getElementById("counterfactual-components");
const counterClearBtn = document.getElementById("counter-clear-btn");

const viewEl = document.getElementById("view");

window.addEventListener("error", (event) => {
  const msg = event?.error?.message || event?.message || "unknown JS error";
  statusEl.textContent = `Runtime error: ${msg}`;
});

let state = {
  graph: { nodes: [], edges: [] },
  tracker: { num_prompts: 0, prompts: [] },
  summary: null,
  meta: null,
};
let hasExternalFileLoad = false;

const viewer = {
  canvas: document.createElement("canvas"),
  ctx: null,
  width: 0,
  height: 0,
  dpr: 1,
  camera: { yaw: -0.55, pitch: 0.35, distance: 78, panX: 0, panY: 0 },
  defaultCamera: { yaw: -0.55, pitch: 0.35, distance: 78, panX: 0, panY: 0 },
  nodes: [],
  edges: [],
  nodeById: new Map(),
  degreeById: new Map(),
  topHeadByLayer: new Map(),
  kinds: [],
  kindVisibility: {},
  selectedId: null,
  projected: [],
  projectedById: new Map(),
  drag: null,
};
viewer.ctx = viewer.canvas.getContext("2d");
viewEl.appendChild(viewer.canvas);

let promptEntries = [];
let currentPromptIndex = -1;
let currentLayerTrace = { base: 0, rows: [] };
let promptBaseEmphasis = new Set();
let promptBasePath = new Set();
let flowActiveNodeIds = new Set();
let flowActivePathIds = new Set();
let emphasizedNodeIds = new Set();
let pathHighlightIds = new Set();

const flowState = {
  steps: [],
  index: 0,
  playing: false,
  speed: 1.0,
  stepMs: 1200,
  accumulatorMs: 0,
};
let flowPulse = 0;
const residualState = {
  rows: [],
  index: 0,
  playing: false,
  stepMs: 1200,
  accumulatorMs: 0,
};
let residualPulse = 0;
let residualActiveNodeIds = new Set();
let residualActivePathIds = new Set();
let counterfactualSelection = new Set();

const PROMPT_HIGHLIGHT_COMPONENTS = 5;
const PROMPT_HIGHLIGHT_PATHS = 3;

const COMPLEXITY_KINDS = {
  simple: new Set(["embed", "unembed", "resid", "ln1", "attn", "ln2", "mlp"]),
  medium: new Set(["embed", "unembed", "resid", "ln1", "attn", "ln2", "mlp", "head", "mlp_fc_in", "mlp_act", "mlp_fc_out"]),
  full: null,
};

const DEFAULT_KIND_COLORS = {
  embed: "#6BCBEB",
  unembed: "#F2A65A",
  resid: "#F4D35E",
  attn: "#4BA3C3",
  ln1: "#7CCBA2",
  ln2: "#7CCBA2",
  mlp: "#8D8DE8",
  head: "#88C7FF",
  head_q: "#5EA8E0",
  head_k: "#6BB7E4",
  head_v: "#7EC5EA",
  head_o: "#8FD4F0",
  mlp_fc_in: "#A2A2F0",
  mlp_act: "#B4B4F3",
  mlp_fc_out: "#C2C2F7",
  neuron: "#CFA3D8",
};

function safeNum(value, fallback = null) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function safeInt(value, fallback = 0) {
  const n = parseInt(value, 10);
  return Number.isFinite(n) ? n : fallback;
}

function fmt(value, digits = 4) {
  const n = safeNum(value, null);
  return n === null ? "n/a" : n.toFixed(digits);
}

function esc(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function clipText(text, maxChars = 220) {
  const s = String(text || "");
  if (s.length <= maxChars) return s;
  return `${s.slice(0, maxChars - 3)}...`;
}

function confidenceBand(margin) {
  const m = Math.abs(safeNum(margin, 0) || 0);
  if (m < 0.2) return "weak";
  if (m < 1.0) return "moderate";
  return "strong";
}

function agreement(score, drop) {
  if (drop === null || drop === undefined) return { label: "N/A", cls: "na" };
  const s = safeNum(score, 0) || 0;
  const d = safeNum(drop, 0) || 0;
  if (Math.abs(d) < 1e-8) return { label: "Neutral", cls: "na" };
  if (s * d > 0) return { label: "Agree", cls: "good" };
  return { label: "Disagree", cls: "bad" };
}

function hexToRgb(hex) {
  const raw = String(hex || "").replace("#", "");
  if (!/^[0-9a-fA-F]{6}$/.test(raw)) {
    return { r: 170, g: 180, b: 194 };
  }
  return {
    r: parseInt(raw.slice(0, 2), 16),
    g: parseInt(raw.slice(2, 4), 16),
    b: parseInt(raw.slice(4, 6), 16),
  };
}

function rgba(hex, alpha) {
  const rgb = hexToRgb(hex);
  return `rgba(${rgb.r},${rgb.g},${rgb.b},${alpha})`;
}

function syncHighlightFromState() {
  emphasizedNodeIds = new Set([
    ...promptBaseEmphasis,
    ...flowActiveNodeIds,
    ...residualActiveNodeIds,
  ]);
  pathHighlightIds = new Set([
    ...promptBasePath,
    ...flowActivePathIds,
    ...residualActivePathIds,
  ]);
}

function currentComplexity() {
  return complexitySelect.value || "full";
}

function kindAllowedByComplexity(kind) {
  const c = currentComplexity();
  if (c === "full") return true;
  const allowed = COMPLEXITY_KINDS[c];
  return allowed ? allowed.has(kind) : true;
}

function isNodeVisible(node) {
  return !!viewer.kindVisibility[node.kind] && kindAllowedByComplexity(node.kind);
}

function isEdgeVisible(edge) {
  return isNodeVisible(edge.src) && isNodeVisible(edge.dst);
}

function resizeViewer() {
  const rect = viewEl.getBoundingClientRect();
  const width = Math.max(320, Math.floor(rect.width || 900));
  const height = Math.max(220, Math.floor(rect.height || 600));
  const dpr = window.devicePixelRatio || 1;

  viewer.width = width;
  viewer.height = height;
  viewer.dpr = dpr;

  viewer.canvas.width = Math.floor(width * dpr);
  viewer.canvas.height = Math.floor(height * dpr);
  viewer.canvas.style.width = `${width}px`;
  viewer.canvas.style.height = `${height}px`;
  viewer.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function resetCamera() {
  viewer.camera = { ...viewer.defaultCamera };
}

function projectPoint(x, y, z) {
  const yaw = viewer.camera.yaw;
  const pitch = viewer.camera.pitch;

  const cy = Math.cos(yaw);
  const sy = Math.sin(yaw);
  const x1 = cy * x + sy * z;
  const z1 = -sy * x + cy * z;

  const cp = Math.cos(pitch);
  const sp = Math.sin(pitch);
  const y2 = cp * y - sp * z1;
  const z2 = sp * y + cp * z1;

  const zz = z2 + viewer.camera.distance;
  if (zz <= 0.3) return null;

  const focal = 480;
  const k = focal / zz;
  return {
    sx: viewer.width * 0.5 + viewer.camera.panX + x1 * k,
    sy: viewer.height * 0.5 + viewer.camera.panY - y2 * k,
    depth: zz,
    scale: k,
  };
}

function render() {
  const ctx = viewer.ctx;
  ctx.clearRect(0, 0, viewer.width, viewer.height);
  ctx.fillStyle = "rgba(10,17,29,0.96)";
  ctx.fillRect(0, 0, viewer.width, viewer.height);
  const flowBeat = flowState.playing
    ? (0.5 + 0.5 * Math.sin(flowPulse))
    : (0.68 + 0.12 * Math.sin(flowPulse * 0.45));
  const residualBeat = residualState.playing
    ? (0.5 + 0.5 * Math.sin(residualPulse))
    : (0.64 + 0.1 * Math.sin(residualPulse * 0.42));

  const visibleNodes = viewer.nodes.filter(isNodeVisible);
  if (!visibleNodes.length) {
    ctx.fillStyle = "rgba(217,228,243,0.92)";
    ctx.font = "13px IBM Plex Sans, Segoe UI, sans-serif";
    ctx.fillText("Load viewer_payload.json to render the model graph.", 12, 24);
    return;
  }

  viewer.projected = [];
  viewer.projectedById.clear();

  for (const node of visibleNodes) {
    const p = projectPoint(node.wx, node.wy, node.wz);
    if (!p) continue;
    // Smaller than before so full graph is visible by default.
    const r = Math.max(0.65, node.baseRadius * p.scale * 0.065);
    const entry = { node, sx: p.sx, sy: p.sy, depth: p.depth, r };
    viewer.projected.push(entry);
    viewer.projectedById.set(node.id, entry);
  }

  for (const edge of viewer.edges) {
    if (!isEdgeVisible(edge)) continue;
    const p1 = viewer.projectedById.get(edge.source);
    const p2 = viewer.projectedById.get(edge.target);
    if (!p1 || !p2) continue;

    const selectedEdge = viewer.selectedId && (edge.source === viewer.selectedId || edge.target === viewer.selectedId);
    const pathEdge = pathHighlightIds.has(edge.source) && pathHighlightIds.has(edge.target);
    const emphEdge = emphasizedNodeIds.has(edge.source) && emphasizedNodeIds.has(edge.target);
    const flowEdge = flowActivePathIds.has(edge.source) && flowActivePathIds.has(edge.target);
    const flowLinked = flowActiveNodeIds.has(edge.source) || flowActiveNodeIds.has(edge.target);
    const residualEdge = residualActivePathIds.has(edge.source) && residualActivePathIds.has(edge.target);
    const residualLinked = residualActiveNodeIds.has(edge.source) || residualActiveNodeIds.has(edge.target);

    let alpha = 0.16;
    let width = 1;
    let stroke = `rgba(136,166,198,${alpha})`;

    if (edge.kind.startsWith("residual")) alpha = 0.24;
    if (emphEdge) {
      alpha = Math.max(alpha, 0.3);
      width = 1.15;
    }
    if (pathEdge) {
      alpha = Math.max(alpha, 0.55);
      width = 1.35;
    }
    if (selectedEdge) {
      alpha = 0.95;
      width = 1.9;
    }
    if (residualLinked) {
      alpha = Math.max(alpha, 0.48 + 0.2 * residualBeat);
      width = Math.max(width, 1.7 + 0.4 * residualBeat);
    }
    if (residualEdge) {
      alpha = Math.max(alpha, 0.72 + 0.18 * residualBeat);
      width = Math.max(width, 2.0 + 0.55 * residualBeat);
    }
    if (flowLinked) {
      alpha = Math.max(alpha, 0.45 + 0.2 * flowBeat);
      width = Math.max(width, 1.6 + 0.35 * flowBeat);
    }
    if (flowEdge) {
      alpha = Math.max(alpha, 0.78 + 0.18 * flowBeat);
      width = Math.max(width, 2.1 + 0.55 * flowBeat);
      stroke = `rgba(127,231,255,${alpha})`;
    } else if (residualEdge) {
      stroke = `rgba(255,213,127,${alpha})`;
    } else {
      stroke = `rgba(136,166,198,${alpha})`;
    }

    ctx.strokeStyle = stroke;
    ctx.lineWidth = width;
    ctx.beginPath();
    ctx.moveTo(p1.sx, p1.sy);
    ctx.lineTo(p2.sx, p2.sy);
    ctx.stroke();
    if (flowEdge) {
      ctx.strokeStyle = `rgba(127,231,255,${0.22 + 0.22 * flowBeat})`;
      ctx.lineWidth = width + 2.2 + 1.4 * flowBeat;
      ctx.beginPath();
      ctx.moveTo(p1.sx, p1.sy);
      ctx.lineTo(p2.sx, p2.sy);
      ctx.stroke();
    } else if (residualEdge) {
      ctx.strokeStyle = `rgba(255,213,127,${0.2 + 0.22 * residualBeat})`;
      ctx.lineWidth = width + 1.9 + 1.1 * residualBeat;
      ctx.beginPath();
      ctx.moveTo(p1.sx, p1.sy);
      ctx.lineTo(p2.sx, p2.sy);
      ctx.stroke();
    }
  }

  const drawNodes = [...viewer.projected].sort((a, b) => b.depth - a.depth);
  for (const p of drawNodes) {
    const n = p.node;
    const selected = n.id === viewer.selectedId;
    const pathHit = pathHighlightIds.has(n.id);
    const emphHit = emphasizedNodeIds.has(n.id);
    const flowNode = flowActiveNodeIds.has(n.id);
    const flowPathNode = flowActivePathIds.has(n.id);
    const residualNode = residualActiveNodeIds.has(n.id);
    const residualPathNode = residualActivePathIds.has(n.id);

    let alpha = 0.74;
    let radius = p.r;
    if (emphHit) {
      alpha = 0.9;
      radius *= 1.15;
    }
    if (pathHit) {
      alpha = 0.95;
      radius *= 1.23;
    }
    if (selected) {
      alpha = 0.99;
      radius *= 1.6;
    } else if (viewer.selectedId) {
      alpha *= 0.75;
    }

    if (flowNode || flowPathNode || residualNode || residualPathNode) {
      const haloR = radius * (2.05 + 0.35 * flowBeat);
      ctx.beginPath();
      ctx.arc(p.sx, p.sy, haloR, 0, Math.PI * 2);
      const isFlow = flowNode || flowPathNode;
      const beat = isFlow ? flowBeat : residualBeat;
      ctx.fillStyle = isFlow
        ? (flowPathNode
            ? `rgba(127,231,255,${0.16 + 0.18 * beat})`
            : `rgba(245,195,106,${0.12 + 0.15 * beat})`)
        : (residualPathNode
            ? `rgba(255,213,127,${0.18 + 0.2 * beat})`
            : `rgba(255,190,95,${0.12 + 0.16 * beat})`);
      ctx.fill();
    }

    ctx.beginPath();
    ctx.arc(p.sx, p.sy, radius, 0, Math.PI * 2);
    ctx.fillStyle = rgba(n.color, alpha);
    ctx.fill();

    if (flowNode || flowPathNode || residualNode || residualPathNode) {
      const isFlow = flowNode || flowPathNode;
      const beat = isFlow ? flowBeat : residualBeat;
      ctx.beginPath();
      ctx.arc(p.sx, p.sy, radius * (1.2 + 0.1 * beat), 0, Math.PI * 2);
      ctx.strokeStyle = isFlow
        ? (flowPathNode
            ? `rgba(127,231,255,${0.8 + 0.15 * beat})`
            : `rgba(255,213,127,${0.72 + 0.12 * beat})`)
        : (residualPathNode
            ? `rgba(255,213,127,${0.88 + 0.1 * beat})`
            : `rgba(255,193,94,${0.76 + 0.1 * beat})`);
      ctx.lineWidth = 1.35 + 0.85 * beat;
      ctx.stroke();
    }

    if (selected) {
      ctx.strokeStyle = "rgba(255,226,156,0.99)";
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }

  const labelIds = ["embed", "unembed", viewer.selectedId];
  ctx.fillStyle = "rgba(237,245,252,0.95)";
  ctx.font = "11px IBM Plex Mono, SFMono-Regular, Menlo, monospace";
  for (const id of labelIds) {
    if (!id) continue;
    const p = viewer.projectedById.get(id);
    if (!p) continue;
    ctx.fillText(id, p.sx + 5, p.sy - 5);
  }
}

let lastFrameTs = null;
function animate(ts) {
  if (lastFrameTs === null) lastFrameTs = ts;
  const dt = Math.max(0, Math.min(80, ts - lastFrameTs));
  lastFrameTs = ts;
  flowPulse += dt * (flowState.playing ? 0.012 * Math.max(0.6, flowState.speed) : 0.0045);
  residualPulse += dt * (residualState.playing ? 0.014 * Math.max(0.6, flowState.speed) : 0.0052);
  updateFlowPlayback(dt);
  updateResidualPlayback(dt);
  render();
  requestAnimationFrame(animate);
}

function pointerPos(event) {
  const rect = viewer.canvas.getBoundingClientRect();
  return { x: event.clientX - rect.left, y: event.clientY - rect.top };
}

function pickNodeAt(x, y) {
  let best = null;
  let bestDist = Infinity;
  for (const p of viewer.projected) {
    const dx = p.sx - x;
    const dy = p.sy - y;
    const d2 = dx * dx + dy * dy;
    const hit = p.r + 4;
    if (d2 <= hit * hit && d2 < bestDist) {
      bestDist = d2;
      best = p;
    }
  }
  return best ? best.node : null;
}

function focusNode(node) {
  const p = viewer.projectedById.get(node.id);
  if (!p) return;
  viewer.camera.panX += (viewer.width * 0.5 - p.sx) * 0.55;
  viewer.camera.panY += (viewer.height * 0.5 - p.sy) * 0.55;
}

function updateSelectionBadge() {
  selectionBadgeEl.textContent = viewer.selectedId || "No selection";
}

function updateNodeInspect() {
  if (!viewer.selectedId) {
    nodeInspectEl.textContent = "Click a node to inspect.";
    return;
  }
  const n = viewer.nodeById.get(viewer.selectedId);
  if (!n) {
    nodeInspectEl.textContent = "Click a node to inspect.";
    return;
  }

  const degree = viewer.degreeById.get(n.id) || 0;
  nodeInspectEl.textContent =
    `id: ${n.id}\n` +
    `label: ${n.label}\n` +
    `kind: ${n.kind}\n` +
    `layer: ${n.layer >= 0 ? `L${n.layer}` : "n/a"}\n` +
    `degree: ${degree}\n` +
    `strength: ${fmt(n.strength)}\n` +
    `xyz: (${fmt(n.x, 3)}, ${fmt(n.y, 3)}, ${fmt(n.z, 3)})`;
}

function selectNode(node, shouldFocus = false) {
  viewer.selectedId = node ? node.id : null;
  if (node && shouldFocus) focusNode(node);
  updateSelectionBadge();
  updateNodeInspect();
  renderTokenLinks();
  refreshMapStats();
}

function clearSelection() {
  selectNode(null, false);
}

function initCanvasInteractions() {
  viewer.canvas.addEventListener("contextmenu", (e) => e.preventDefault());

  viewer.canvas.addEventListener("pointerdown", (event) => {
    const p = pointerPos(event);
    viewer.drag = {
      pointerId: event.pointerId,
      startX: p.x,
      startY: p.y,
      lastX: p.x,
      lastY: p.y,
      moved: false,
      mode: event.button === 2 || event.shiftKey ? "pan" : "rotate",
    };
    viewer.canvas.classList.add("dragging");
    viewer.canvas.setPointerCapture(event.pointerId);
  });

  viewer.canvas.addEventListener("pointermove", (event) => {
    if (!viewer.drag || event.pointerId !== viewer.drag.pointerId) return;
    const p = pointerPos(event);
    const dx = p.x - viewer.drag.lastX;
    const dy = p.y - viewer.drag.lastY;
    viewer.drag.lastX = p.x;
    viewer.drag.lastY = p.y;

    if (Math.abs(p.x - viewer.drag.startX) > 2 || Math.abs(p.y - viewer.drag.startY) > 2) {
      viewer.drag.moved = true;
    }

    if (viewer.drag.mode === "pan") {
      viewer.camera.panX += dx;
      viewer.camera.panY += dy;
    } else {
      viewer.camera.yaw += dx * 0.006;
      viewer.camera.pitch += dy * 0.006;
      viewer.camera.pitch = Math.max(-1.45, Math.min(1.45, viewer.camera.pitch));
    }
  });

  function endDrag(event) {
    if (!viewer.drag || event.pointerId !== viewer.drag.pointerId) return;
    const p = pointerPos(event);
    const moved = viewer.drag.moved;
    viewer.drag = null;
    viewer.canvas.classList.remove("dragging");

    if (!moved) {
      const node = pickNodeAt(p.x, p.y);
      selectNode(node, false);
    }
  }

  viewer.canvas.addEventListener("pointerup", endDrag);
  viewer.canvas.addEventListener("pointercancel", endDrag);

  viewer.canvas.addEventListener(
    "wheel",
    (event) => {
      event.preventDefault();
      viewer.camera.distance *= Math.exp(event.deltaY * 0.0012);
      viewer.camera.distance = Math.max(10, Math.min(220, viewer.camera.distance));
    },
    { passive: false }
  );
}

function createKindToggle(kind, color) {
  const row = document.createElement("label");
  row.className = "check";

  const cb = document.createElement("input");
  cb.type = "checkbox";
  cb.checked = true;
  cb.addEventListener("change", () => {
    viewer.kindVisibility[kind] = cb.checked;
    if (viewer.selectedId) {
      const node = viewer.nodeById.get(viewer.selectedId);
      if (!node || !isNodeVisible(node)) {
        selectNode(null, false);
      }
    }
  });

  const swatch = document.createElement("span");
  swatch.className = "swatch";
  swatch.style.background = color;

  const txt = document.createElement("span");
  txt.textContent = kind;

  row.appendChild(cb);
  row.appendChild(swatch);
  row.appendChild(txt);
  return row;
}

function applyGraph(graph) {
  const rawNodes = Array.isArray(graph?.nodes) ? graph.nodes : [];
  const rawEdges = Array.isArray(graph?.edges) ? graph.edges : [];

  viewer.nodes = [];
  viewer.edges = [];
  viewer.nodeById = new Map();
  viewer.degreeById = new Map();
  viewer.topHeadByLayer = new Map();
  viewer.kinds = [];
  viewer.kindVisibility = {};
  kindTogglesEl.innerHTML = "";

  for (let i = 0; i < rawNodes.length; i += 1) {
    const n = rawNodes[i] || {};
    const id = String(n.id || `node_${i}`);
    const layer = safeInt(n.layer, -1);
    const node = {
      id,
      label: String(n.label || id),
      kind: String(n.kind || "unknown"),
      layer,
      x: safeNum(n.x, layer >= 0 ? layer : 0),
      y: safeNum(n.y, 0),
      z: safeNum(n.z, 0),
      size: Math.max(1, safeNum(n.size, 10) || 10),
      strength: safeNum(n.strength, null),
      color: String(n.color || DEFAULT_KIND_COLORS[String(n.kind || "")] || "#aab5c2"),
    };
    viewer.nodes.push(node);
    viewer.nodeById.set(id, node);
  }

  if (!viewer.nodes.length) {
    nodeCountEl.textContent = "0";
    edgeCountEl.textContent = "0";
    kindCountEl.textContent = "0";
    mapStatsEl.textContent = "No graph loaded.";
    return;
  }

  let minX = Infinity;
  let minY = Infinity;
  let minZ = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  let maxZ = -Infinity;
  let minSize = Infinity;
  let maxSize = -Infinity;

  for (const n of viewer.nodes) {
    minX = Math.min(minX, n.x);
    minY = Math.min(minY, n.y);
    minZ = Math.min(minZ, n.z);
    maxX = Math.max(maxX, n.x);
    maxY = Math.max(maxY, n.y);
    maxZ = Math.max(maxZ, n.z);
    minSize = Math.min(minSize, n.size);
    maxSize = Math.max(maxSize, n.size);
  }

  const cx = (minX + maxX) * 0.5;
  const cy = (minY + maxY) * 0.5;
  const cz = (minZ + maxZ) * 0.5;
  const span = Math.max(1e-5, maxX - minX, maxY - minY, maxZ - minZ);
  const scale = 22 / span;
  const sizeRange = Math.max(1e-6, maxSize - minSize);

  for (const n of viewer.nodes) {
    n.wx = (n.x - cx) * scale;
    n.wy = (n.y - cy) * scale;
    n.wz = (n.z - cz) * scale;
    const t = Math.max(0, Math.min(1, (n.size - minSize) / sizeRange));
    // Smaller legacy-like footprint for dense graph.
    n.baseRadius = 0.85 + 2.9 * Math.sqrt(t);
  }

  // Auto-fit camera to graph bounds on load so it's neither over-zoomed nor tiny.
  let maxWorldRadius = 0;
  for (const n of viewer.nodes) {
    const r = Math.hypot(n.wx, n.wy, n.wz) + n.baseRadius * 0.7;
    if (r > maxWorldRadius) maxWorldRadius = r;
  }
  const focal = 480;
  const minView = Math.max(260, Math.min(viewer.width || 900, viewer.height || 600));
  const targetScreenRadius = minView * 0.36;
  const fitDistance = Math.max(
    22,
    Math.min(74, (focal * Math.max(6, maxWorldRadius)) / Math.max(140, targetScreenRadius))
  );
  viewer.defaultCamera.yaw = -0.55;
  viewer.defaultCamera.pitch = 0.35;
  viewer.defaultCamera.distance = fitDistance;
  viewer.defaultCamera.panX = 0;
  viewer.defaultCamera.panY = 0;
  viewer.camera = { ...viewer.defaultCamera };

  for (const e of rawEdges) {
    const source = String(e?.source || "");
    const target = String(e?.target || "");
    const src = viewer.nodeById.get(source);
    const dst = viewer.nodeById.get(target);
    if (!src || !dst) continue;
    viewer.edges.push({
      source,
      target,
      src,
      dst,
      kind: String(e.kind || "unknown"),
      weight: safeNum(e.weight, 0),
    });
    viewer.degreeById.set(source, (viewer.degreeById.get(source) || 0) + 1);
    viewer.degreeById.set(target, (viewer.degreeById.get(target) || 0) + 1);
  }

  for (const n of viewer.nodes) {
    if (n.kind !== "head") continue;
    if (n.layer < 0) continue;
    const strength = safeNum(n.strength, 0) || 0;
    const cur = viewer.topHeadByLayer.get(n.layer);
    if (!cur || strength > cur.strength) {
      viewer.topHeadByLayer.set(n.layer, { id: n.id, strength });
    }
  }

  const kinds = [...new Set(viewer.nodes.map((n) => n.kind))].sort();
  viewer.kinds = kinds;
  for (const kind of kinds) {
    const color = viewer.nodes.find((n) => n.kind === kind)?.color || "#8899aa";
    viewer.kindVisibility[kind] = true;
    kindTogglesEl.appendChild(createKindToggle(kind, color));
  }

  nodeCountEl.textContent = String(viewer.nodes.length);
  edgeCountEl.textContent = String(viewer.edges.length);
  kindCountEl.textContent = String(kinds.length);

  if (viewer.selectedId && !viewer.nodeById.has(viewer.selectedId)) {
    viewer.selectedId = null;
  }
  updateSelectionBadge();
  updateNodeInspect();
  refreshMapStats();
}

function refreshMapStats() {
  const visibleNodes = viewer.nodes.filter(isNodeVisible);
  const visibleEdges = viewer.edges.filter(isEdgeVisible);
  mapStatsEl.textContent = `visible nodes=${visibleNodes.length} | visible edges=${visibleEdges.length}`;
}

function parseComponentLabel(label) {
  const s = String(label || "");
  if (s === "embed" || s === "pos_embed") return { kind: s, layer: -1 };
  const m = s.match(/^(\d+)_(attn_out|mlp_out)$/);
  if (!m) return { kind: "unknown", layer: -1 };
  return { kind: m[2], layer: safeInt(m[1], -1) };
}

function componentNodeCandidates(label, kindHint = "", layerHint = -1) {
  const parsed = parseComponentLabel(label);
  const kind = parsed.kind === "unknown" ? String(kindHint || "") : parsed.kind;
  const layer = parsed.layer >= 0 ? parsed.layer : safeInt(layerHint, -1);
  const out = [];

  if (kind === "embed" || kind === "pos_embed" || label === "embed" || label === "pos_embed") {
    out.push("embed");
  } else if ((kind === "attn_out" || kind === "attn") && layer >= 0) {
    out.push(`L${layer}_attn`);
    out.push(`L${layer}_resid`);
    const head = viewer.topHeadByLayer.get(layer);
    if (head && head.id) out.push(head.id);
  } else if ((kind === "mlp_out" || kind === "mlp") && layer >= 0) {
    out.push(`L${layer}_mlp`);
    out.push(`L${layer}_resid`);
  }

  const unique = [];
  const seen = new Set();
  for (const id of out) {
    if (!viewer.nodeById.has(id) || seen.has(id)) continue;
    seen.add(id);
    unique.push(id);
  }
  return unique;
}

function nodeIdsForPath(path) {
  const ids = componentNodeCandidates(path.component_label || "", path.component_kind || "", path.layer);
  const layer = safeInt(path.layer, -1);
  if (layer >= 0) {
    const head = viewer.topHeadByLayer.get(layer);
    if (head && head.id && viewer.nodeById.has(head.id)) ids.push(head.id);
  }
  if (viewer.nodeById.has("unembed")) ids.push("unembed");

  const unique = [];
  const seen = new Set();
  for (const id of ids) {
    if (seen.has(id)) continue;
    seen.add(id);
    unique.push(id);
  }
  return unique;
}

function getCurrentPrompt() {
  if (currentPromptIndex < 0 || currentPromptIndex >= promptEntries.length) return null;
  return promptEntries[currentPromptIndex];
}

function applyPromptHighlights(prompt, support, paths) {
  const emph = new Set();
  for (const c of (support || []).slice(0, PROMPT_HIGHLIGHT_COMPONENTS)) {
    for (const id of componentNodeCandidates(c.label, c.kind, c.layer)) emph.add(id);
  }

  const pathIds = new Set();
  for (const path of (paths || []).slice(0, PROMPT_HIGHLIGHT_PATHS)) {
    for (const id of nodeIdsForPath(path)) pathIds.add(id);
  }

  promptBaseEmphasis = emph;
  promptBasePath = pathIds;
  syncHighlightFromState();
}

function renderComponentList(el, comps) {
  el.innerHTML = "";
  if (!comps.length) {
    const empty = document.createElement("div");
    empty.className = "small";
    empty.textContent = "none";
    el.appendChild(empty);
    return;
  }

  for (const c of comps) {
    const agr = agreement(c.score, c.ablation_drop);
    const row = document.createElement("div");
    row.className = "component-row clickable";
    row.innerHTML =
      `<div class="component-title">` +
      `<span class="mono">${esc(c.label)}</span>` +
      `<span class="badge ${agr.cls}">${agr.label}</span>` +
      `</div>` +
      `<div>pred=${fmt(c.score)} | abl=${fmt(c.ablation_drop)}</div>` +
      `<div class="small">kind=${esc(c.kind)} layer=${safeInt(c.layer, -1) >= 0 ? `L${safeInt(c.layer, -1)}` : "n/a"}</div>`;

    row.addEventListener("click", () => {
      pauseFlow();
      const ids = componentNodeCandidates(c.label, c.kind, c.layer);
      flowActiveNodeIds = new Set(ids);
      flowActivePathIds = new Set(ids);
      syncHighlightFromState();
      const node = ids.length ? viewer.nodeById.get(ids[0]) : null;
      if (node) selectNode(node, true);
      renderFlowPanel();
    });

    el.appendChild(row);
  }
}

function renderPaths(paths, targetToken) {
  pathsListEl.innerHTML = "";
  if (!paths || !paths.length) {
    const empty = document.createElement("div");
    empty.className = "small";
    empty.textContent = "No aligned attention-source paths found.";
    pathsListEl.appendChild(empty);
    return;
  }

  for (const p of paths.slice(0, 3)) {
    const layer = safeInt(p.layer, -1);
    const headProxy = viewer.topHeadByLayer.get(layer);
    const headLabel = headProxy ? headProxy.id : `L${layer}_H*`;
    const ids = nodeIdsForPath(p);

    const item = document.createElement("div");
    item.className = "path-item clickable";
    item.innerHTML =
      `<div><span class="mono">${esc(p.source_token)}</span> @pos ${esc(p.source_position)} ` +
      `-> <span class="mono">${esc(headLabel)}</span> -> <span class="mono">L${layer}_attn</span> ` +
      `-> logit(<span class="mono">${esc(targetToken)}</span>)</div>` +
      `<div class="small">source=${fmt(p.source_score)} component=${fmt(p.component_score)} path=${fmt(p.path_strength)}</div>`;

    item.addEventListener("click", () => {
      pauseFlow();
      flowActiveNodeIds = new Set(ids);
      flowActivePathIds = new Set(ids);
      syncHighlightFromState();
      const node = ids.length ? viewer.nodeById.get(ids[0]) : null;
      if (node) selectNode(node, true);
      renderFlowPanel();
    });

    pathsListEl.appendChild(item);
  }
}

function renderTokenLinks() {
  const p = getCurrentPrompt();
  const selectedNode = viewer.selectedId ? viewer.nodeById.get(viewer.selectedId) : null;

  if (!p) {
    tokenLinkTitleEl.textContent = "No prompt selected.";
    tokenLinksEl.innerHTML = "";
    return;
  }
  if (!selectedNode) {
    tokenLinkTitleEl.textContent = "Select a component/head node.";
    tokenLinksEl.innerHTML = "";
    return;
  }

  const attnKinds = new Set(["attn", "head", "head_q", "head_k", "head_v", "head_o"]);
  const mlpKinds = new Set(["mlp", "mlp_fc_in", "mlp_act", "mlp_fc_out", "neuron"]);

  let label = null;
  if (selectedNode.layer >= 0 && attnKinds.has(selectedNode.kind)) label = `${selectedNode.layer}_attn_out`;
  else if (selectedNode.layer >= 0 && mlpKinds.has(selectedNode.kind)) label = `${selectedNode.layer}_mlp_out`;
  else if (selectedNode.id === "embed") label = "embed";

  if (!label) {
    tokenLinkTitleEl.textContent = `${selectedNode.id}: no direct token-link mapping.`;
    tokenLinksEl.innerHTML = "";
    return;
  }

  const comp = p.byLabel.get(label);
  if (!comp) {
    tokenLinkTitleEl.textContent = `${selectedNode.id}: component ${label} missing in prompt data.`;
    tokenLinksEl.innerHTML = "";
    return;
  }

  tokenLinkTitleEl.textContent = `${selectedNode.id} -> ${label} (prompt ${p.prompt_id})`;
  tokenLinksEl.innerHTML = "";

  const meta = document.createElement("div");
  meta.className = "small";
  meta.textContent = `pred=${fmt(comp.score)} abl=${fmt(comp.ablation_drop)}`;
  tokenLinksEl.appendChild(meta);

  const sources = [...(comp.sources || [])].sort((a, b) => {
    const av = Math.abs(safeNum(a.abs_score ?? a.score, 0) || 0);
    const bv = Math.abs(safeNum(b.abs_score ?? b.score, 0) || 0);
    return bv - av;
  });

  if (!sources.length) {
    const empty = document.createElement("div");
    empty.className = "small";
    empty.style.marginTop = "6px";
    empty.textContent = "No source-token attribution available for this component.";
    tokenLinksEl.appendChild(empty);
    return;
  }

  const wrap = document.createElement("div");
  wrap.className = "token-chip-wrap";
  for (const src of sources) {
    const s = safeNum(src.score, 0) || 0;
    const chip = document.createElement("span");
    chip.className = `token-chip ${s >= 0 ? "pos" : "neg"}`;
    chip.textContent = `pos ${src.position}: ${src.token} (${fmt(s, 3)})`;
    wrap.appendChild(chip);
  }
  tokenLinksEl.appendChild(wrap);
}

function getModelMaxLayer() {
  let maxLayer = 0;
  for (const n of viewer.nodes) {
    if (safeInt(n.layer, -1) >= 0) {
      maxLayer = Math.max(maxLayer, safeInt(n.layer, 0));
    }
  }
  return maxLayer;
}

function buildLayerTrace(prompt) {
  const layerBuckets = new Map();
  let base = 0.0;

  for (const comp of prompt?.components || []) {
    const score = safeNum(comp.score, 0) || 0;
    const layer = safeInt(comp.layer, -1);
    const kind = String(comp.kind || "");
    if (layer < 0) {
      base += score;
      continue;
    }
    if (!layerBuckets.has(layer)) {
      layerBuckets.set(layer, { attn: 0.0, mlp: 0.0, other: 0.0 });
    }
    const bucket = layerBuckets.get(layer);
    if (kind === "attn_out") bucket.attn += score;
    else if (kind === "mlp_out") bucket.mlp += score;
    else bucket.other += score;
  }

  const rows = [];
  let cumulative = base;
  const lastLayer = Math.max(0, getModelMaxLayer());
  for (let layer = 0; layer <= lastLayer; layer += 1) {
    const bucket = layerBuckets.get(layer) || { attn: 0.0, mlp: 0.0, other: 0.0 };
    const delta = bucket.attn + bucket.mlp + bucket.other;
    cumulative += delta;
    rows.push({
      layer,
      attn: bucket.attn,
      mlp: bucket.mlp,
      other: bucket.other,
      delta,
      cumulative,
    });
  }
  return { base, rows };
}

function getActiveFlowLayer() {
  if (!flowState.steps.length) return null;
  const idx = Math.max(0, Math.min(flowState.index, flowState.steps.length - 1));
  const step = flowState.steps[idx];
  const layer = safeInt(step.layer, -1);
  return layer >= 0 ? layer : null;
}

function getActiveResidualLayer() {
  if (!residualState.rows.length) return null;
  const idx = Math.max(0, Math.min(residualState.index, residualState.rows.length - 1));
  return safeInt(residualState.rows[idx].layer, -1);
}

function drawMarginChart(trace, prompt, activeLayer) {
  const ctx = marginChartEl.getContext("2d");
  const rect = marginChartEl.getBoundingClientRect();
  const cssWidth = Math.max(220, Math.floor(rect.width || 380));
  const cssHeight = 146;
  const dpr = window.devicePixelRatio || 1;

  marginChartEl.width = Math.floor(cssWidth * dpr);
  marginChartEl.height = Math.floor(cssHeight * dpr);
  marginChartEl.style.width = `${cssWidth}px`;
  marginChartEl.style.height = `${cssHeight}px`;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssWidth, cssHeight);
  ctx.fillStyle = "#0f1822";
  ctx.fillRect(0, 0, cssWidth, cssHeight);

  const rows = trace?.rows || [];
  if (!rows.length) {
    ctx.fillStyle = "rgba(157,177,198,0.9)";
    ctx.font = "12px IBM Plex Sans, Segoe UI, sans-serif";
    ctx.fillText("No layer trace available.", 10, 22);
    return;
  }

  const left = 34;
  const right = cssWidth - 8;
  const top = 16;
  const bottom = cssHeight - 20;
  const width = Math.max(10, right - left);
  const height = Math.max(10, bottom - top);

  const points = [trace.base, ...rows.map((r) => r.cumulative)];
  const yVals = [...points, 0];
  if (prompt && prompt.clean_margin !== null && prompt.clean_margin !== undefined) {
    yVals.push(Number(prompt.clean_margin));
  }
  if (prompt && prompt.pred_margin !== null && prompt.pred_margin !== undefined) {
    yVals.push(Number(prompt.pred_margin));
  }

  let yMin = Math.min(...yVals);
  let yMax = Math.max(...yVals);
  if (Math.abs(yMax - yMin) < 1e-9) {
    yMin -= 1;
    yMax += 1;
  }
  const pad = 0.1 * (yMax - yMin);
  yMin -= pad;
  yMax += pad;

  const n = points.length;
  function xAt(i) {
    if (n <= 1) return left;
    return left + (i / (n - 1)) * width;
  }
  function yAt(v) {
    const t = (v - yMin) / (yMax - yMin);
    return bottom - t * height;
  }

  ctx.strokeStyle = "rgba(71,96,122,0.8)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(left, top);
  ctx.lineTo(left, bottom);
  ctx.lineTo(right, bottom);
  ctx.stroke();

  const yZero = yAt(0.0);
  ctx.strokeStyle = "rgba(120,145,170,0.35)";
  ctx.setLineDash([4, 3]);
  ctx.beginPath();
  ctx.moveTo(left, yZero);
  ctx.lineTo(right, yZero);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.strokeStyle = "#5CB3FF";
  ctx.lineWidth = 2;
  ctx.beginPath();
  points.forEach((v, i) => {
    const x = xAt(i);
    const y = yAt(v);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  for (let i = 0; i < points.length; i += 1) {
    const x = xAt(i);
    const y = yAt(points[i]);
    ctx.fillStyle = "#d8e8f7";
    ctx.beginPath();
    ctx.arc(x, y, 2.3, 0, 2 * Math.PI);
    ctx.fill();
  }

  if (activeLayer !== null && activeLayer >= 0 && activeLayer < rows.length) {
    const idx = activeLayer + 1;
    const x = xAt(idx);
    const y = yAt(points[idx]);
    ctx.strokeStyle = "rgba(255,209,102,0.9)";
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    ctx.moveTo(x, top);
    ctx.lineTo(x, bottom);
    ctx.stroke();
    ctx.fillStyle = "#FFD166";
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, 2 * Math.PI);
    ctx.fill();
  }

  ctx.fillStyle = "rgba(157,177,198,0.95)";
  ctx.font = "11px IBM Plex Sans, Segoe UI, sans-serif";
  ctx.fillText(`base ${fmt(trace.base, 3)}`, left + 4, top + 11);
  if (prompt && prompt.pred_margin !== null && prompt.pred_margin !== undefined) {
    ctx.fillText(`pred ${fmt(prompt.pred_margin, 3)}`, left + 90, top + 11);
  }
  if (prompt && prompt.clean_margin !== null && prompt.clean_margin !== undefined) {
    ctx.fillText(`clean ${fmt(prompt.clean_margin, 3)}`, left + 170, top + 11);
  }
}

function renderEquationPanel() {
  const prompt = getCurrentPrompt();
  if (!prompt) {
    equationTextEl.textContent = "Load tracker data to compute equations.";
    drawMarginChart({ base: 0, rows: [] }, null, null);
    return;
  }

  const trace = buildLayerTrace(prompt);
  currentLayerTrace = trace;
  const activeLayer = getActiveResidualLayer() !== null ? getActiveResidualLayer() : getActiveFlowLayer();
  const target = prompt.response_token || prompt.target_token || "target";
  const foil = prompt.foil_token || "foil";

  equationTextEl.textContent =
    `r_0 = tok_embed + pos_embed\n` +
    `r_{l+1/2} = r_l + Attn_l(LN1(r_l))\n` +
    `r_{l+1} = r_{l+1/2} + MLP_l(LN2(r_{l+1/2}))\n` +
    `logit(t) = W_U[:,t]^T r_L + b_U[t]\n` +
    `margin(y,foil) = logit(y)-logit(foil) = sum(component contributions)\n\n` +
    `y = "${target}" ; foil = "${foil}"\n` +
    `base(embed+pos) = ${fmt(trace.base, 4)}\n` +
    `pred margin_hat = ${fmt(prompt.pred_margin, 4)}\n` +
    `clean margin = ${fmt(prompt.clean_margin, 4)}\n` +
    `reconstruction error = ${fmt(prompt.reconstruction_error, 6)}\n` +
    `${activeLayer !== null ? `active layer = L${activeLayer}` : "active layer = n/a"}`;

  drawMarginChart(trace, prompt, activeLayer);
}

function setResidualStep(index, shouldFocus = false) {
  if (!residualState.rows.length) {
    residualState.index = 0;
    residualActiveNodeIds = new Set();
    residualActivePathIds = new Set();
    syncHighlightFromState();
    renderResidualPanel();
    renderEquationPanel();
    return;
  }

  const clamped = Math.max(0, Math.min(residualState.rows.length - 1, safeInt(index, 0)));
  residualState.index = clamped;
  const row = residualState.rows[clamped];
  const layer = safeInt(row.layer, -1);

  const ids = [];
  if (layer >= 0) {
    ids.push(`L${layer}_resid`);
    ids.push(`L${layer}_attn`);
    ids.push(`L${layer}_mlp`);
    if (layer > 0) ids.push(`L${layer - 1}_resid`);
  }
  residualActiveNodeIds = new Set(ids.filter((id) => viewer.nodeById.has(id)));
  residualActivePathIds = new Set([...residualActiveNodeIds]);
  syncHighlightFromState();

  if (shouldFocus) {
    const first = [...residualActiveNodeIds][0];
    if (first) {
      const node = viewer.nodeById.get(first);
      if (node) selectNode(node, true);
    }
  }
  renderResidualPanel();
  renderEquationPanel();
}

function rebuildResidualForCurrentPrompt(resetIndex = true) {
  const prompt = getCurrentPrompt();
  if (!prompt) {
    residualState.rows = [];
    residualState.index = 0;
    residualState.playing = false;
    residualState.accumulatorMs = 0;
    residualActiveNodeIds = new Set();
    residualActivePathIds = new Set();
    syncHighlightFromState();
    renderResidualPanel();
    return;
  }

  const trace = buildLayerTrace(prompt);
  residualState.rows = trace.rows;
  residualState.accumulatorMs = 0;
  if (resetIndex) residualState.index = 0;
  else residualState.index = Math.max(0, Math.min(residualState.index, Math.max(0, trace.rows.length - 1)));
  setResidualStep(residualState.index, false);
}

function updateResidualPlayback(dtMs) {
  if (!residualState.playing || residualState.rows.length <= 1) return;
  residualState.accumulatorMs += dtMs * flowState.speed;
  while (residualState.accumulatorMs >= residualState.stepMs) {
    residualState.accumulatorMs -= residualState.stepMs;
    if (residualState.index >= residualState.rows.length - 1) {
      residualState.playing = false;
      setResidualStep(residualState.rows.length - 1, false);
      return;
    }
    setResidualStep(residualState.index + 1, false);
  }
}

function renderResidualPanel() {
  const prompt = getCurrentPrompt();
  const rows = residualState.rows;
  if (!prompt || !rows.length) {
    residPlayBtn.disabled = true;
    residStepBtn.disabled = true;
    residResetBtn.disabled = true;
    residProgressEl.disabled = true;
    residProgressEl.max = "0";
    residProgressEl.value = "0";
    residCaptionEl.textContent = "Load tracker data to animate residual updates.";
    residLayerBarsEl.innerHTML = "";
    residPlayBtn.textContent = "Play";
    return;
  }

  residPlayBtn.disabled = rows.length <= 1;
  residStepBtn.disabled = rows.length <= 0;
  residResetBtn.disabled = rows.length <= 0;
  residProgressEl.disabled = rows.length <= 0;
  residProgressEl.max = String(Math.max(0, rows.length - 1));
  residProgressEl.value = String(Math.max(0, Math.min(residualState.index, rows.length - 1)));
  residPlayBtn.textContent = residualState.playing ? "Pause" : "Play";

  const idx = Math.max(0, Math.min(residualState.index, rows.length - 1));
  const row = rows[idx];
  residCaptionEl.textContent =
    `Layer ${row.layer}: attn=${fmt(row.attn, 4)}, mlp=${fmt(row.mlp, 4)}, delta=${fmt(row.delta, 4)}, cumulative=${fmt(row.cumulative, 4)}`;

  let maxAbs = 0.0;
  for (const r of rows) maxAbs = Math.max(maxAbs, Math.abs(r.delta));
  maxAbs = Math.max(maxAbs, 1e-6);

  residLayerBarsEl.innerHTML = "";
  for (let i = 0; i < rows.length; i += 1) {
    const r = rows[i];
    const isActive = i === idx;
    const widthPct = Math.min(100, (Math.abs(r.delta) / maxAbs) * 100);
    const rowEl = document.createElement("div");
    rowEl.className = `layer-bar-row${isActive ? " active" : ""}`;
    const sign = r.delta >= 0 ? "+" : "-";
    rowEl.innerHTML =
      `<div class=\"layer-bar-head\"><span class=\"mono\">L${r.layer}</span><span class=\"mono\">${sign}${fmt(Math.abs(r.delta), 4)}</span></div>` +
      `<div class=\"layer-bar-track\"><div class=\"layer-bar-fill\" style=\"width:${fmt(widthPct, 2)}%; background:${r.delta >= 0 ? "#4db380" : "#d66b7a"}\"></div></div>` +
      `<div class=\"layer-bar-meta\">attn=${fmt(r.attn, 4)} | mlp=${fmt(r.mlp, 4)} | cum=${fmt(r.cumulative, 4)}</div>`;
    rowEl.addEventListener("click", () => {
      residualState.playing = false;
      setResidualStep(i, true);
    });
    residLayerBarsEl.appendChild(rowEl);
  }
}

function drawSankeyPanel() {
  const ctx = sankeyCanvasEl.getContext("2d");
  const rect = sankeyCanvasEl.getBoundingClientRect();
  const cssWidth = Math.max(220, Math.floor(rect.width || 380));
  const cssHeight = 220;
  const dpr = window.devicePixelRatio || 1;
  sankeyCanvasEl.width = Math.floor(cssWidth * dpr);
  sankeyCanvasEl.height = Math.floor(cssHeight * dpr);
  sankeyCanvasEl.style.width = `${cssWidth}px`;
  sankeyCanvasEl.style.height = `${cssHeight}px`;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssWidth, cssHeight);
  ctx.fillStyle = "#0f1822";
  ctx.fillRect(0, 0, cssWidth, cssHeight);

  const prompt = getCurrentPrompt();
  if (!prompt) {
    ctx.fillStyle = "rgba(157,177,198,0.9)";
    ctx.font = "12px IBM Plex Sans, Segoe UI, sans-serif";
    ctx.fillText("Load tracker data to render Sankey flow.", 10, 22);
    sankeyCaptionEl.textContent = "Load tracker data to render causal flow.";
    return;
  }

  const topPaths = [...(prompt.top_paths || [])].slice(0, 5);
  if (!topPaths.length) {
    ctx.fillStyle = "rgba(157,177,198,0.9)";
    ctx.font = "12px IBM Plex Sans, Segoe UI, sans-serif";
    ctx.fillText("No top_paths available for Sankey view.", 10, 22);
    sankeyCaptionEl.textContent = "No aligned paths in tracker for this token.";
    return;
  }

  const xToken = 26;
  const xHead = Math.floor(cssWidth * 0.36);
  const xResid = Math.floor(cssWidth * 0.67);
  const xLogit = cssWidth - 26;
  const yTop = 26;
  const yBottom = cssHeight - 24;
  const slotCount = Math.max(3, topPaths.length);
  const yStep = (yBottom - yTop) / Math.max(1, slotCount - 1);

  const tokenNodes = [];
  const headNodes = [];
  const residNodes = [];
  for (let i = 0; i < topPaths.length; i += 1) {
    const p = topPaths[i];
    const layer = safeInt(p.layer, -1);
    const head = viewer.topHeadByLayer.get(layer);
    const y = yTop + i * yStep;
    tokenNodes.push({ key: `tok:${i}`, label: `${p.source_token}@${p.source_position}`, y, path: p });
    headNodes.push({ key: `head:${i}`, label: head ? head.id : `L${layer}_H*`, y, path: p });
    residNodes.push({ key: `resid:${i}`, label: `L${layer}_resid`, y, path: p });
  }
  const logitNode = { label: `logit(${prompt.response_token || prompt.target_token || "target"})`, y: Math.floor((yTop + yBottom) / 2) };

  let maxW = 0;
  for (const p of topPaths) {
    maxW = Math.max(maxW, Math.abs(safeNum(p.path_strength, 0) || 0));
  }
  maxW = Math.max(maxW, 1e-6);

  function linkWidth(v) {
    return 1.2 + 6.2 * Math.sqrt(Math.abs(v) / maxW);
  }

  function drawLink(x1, y1, x2, y2, value, positive) {
    const w = linkWidth(value);
    ctx.strokeStyle = positive ? "rgba(77,179,128,0.58)" : "rgba(214,107,122,0.6)";
    ctx.lineWidth = w;
    ctx.beginPath();
    const c1 = x1 + (x2 - x1) * 0.45;
    const c2 = x1 + (x2 - x1) * 0.72;
    ctx.moveTo(x1, y1);
    ctx.bezierCurveTo(c1, y1, c2, y2, x2, y2);
    ctx.stroke();
  }

  for (let i = 0; i < topPaths.length; i += 1) {
    const p = topPaths[i];
    const token = tokenNodes[i];
    const head = headNodes[i];
    const resid = residNodes[i];
    const score = safeNum(p.component_score, 0) || 0;
    const src = safeNum(p.source_score, 0) || 0;
    const strength = Math.abs(safeNum(p.path_strength, 0) || 0);
    drawLink(xToken + 26, token.y, xHead - 26, head.y, Math.abs(src), score >= 0);
    drawLink(xHead + 28, head.y, xResid - 30, resid.y, Math.abs(score), score >= 0);
    drawLink(xResid + 30, resid.y, xLogit - 24, logitNode.y, strength, score >= 0);
  }

  function drawNode(x, y, label, cls) {
    const color = cls === "token"
      ? "#6BCBEB"
      : cls === "head"
      ? "#88C7FF"
      : cls === "resid"
      ? "#F4D35E"
      : "#F2A65A";
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, 5.4, 0, 2 * Math.PI);
    ctx.fill();
    ctx.fillStyle = "rgba(224,236,247,0.95)";
    ctx.font = "10px IBM Plex Mono, SFMono-Regular, Menlo, monospace";
    ctx.fillText(label, x + 8, y - 2);
  }

  tokenNodes.forEach((n) => drawNode(xToken, n.y, n.label, "token"));
  headNodes.forEach((n) => drawNode(xHead, n.y, n.label, "head"));
  residNodes.forEach((n) => drawNode(xResid, n.y, n.label, "resid"));
  drawNode(xLogit, logitNode.y, logitNode.label, "logit");
  sankeyCaptionEl.textContent = `Rendered ${topPaths.length} strongest token->head->residual->logit paths.`;
}

function componentKey(c) {
  return `${String(c.label || "")}|${safeInt(c.layer, -1)}|${safeInt(c.rank, -1)}`;
}

function getCounterfactualComponents(prompt) {
  return [...(prompt?.components || [])]
    .filter((c) => c.ablation_drop !== null && c.ablation_drop !== undefined)
    .sort((a, b) => Math.abs(safeNum(b.score, 0) || 0) - Math.abs(safeNum(a.score, 0) || 0))
    .slice(0, 16);
}

function updateCounterfactualSummary() {
  const prompt = getCurrentPrompt();
  if (!prompt) {
    counterSummaryEl.textContent = "Toggle component ablations to estimate margin/top-token change.";
    return;
  }
  const ablatable = getCounterfactualComponents(prompt);
  const selected = ablatable.filter((c) => counterfactualSelection.has(componentKey(c)));
  const base = safeNum(prompt.clean_margin, 0) || 0;
  const drop = selected.reduce((acc, c) => acc + (safeNum(c.ablation_drop, 0) || 0), 0);
  const cfMargin = base - drop;

  let predictedToken = prompt.response_token || prompt.target_token || "target";
  if (cfMargin < 0) predictedToken = prompt.foil_token || "foil";
  const strongest = [...selected].sort((a, b) => Math.abs(safeNum(b.ablation_drop, 0) || 0) - Math.abs(safeNum(a.ablation_drop, 0) || 0))[0];
  if (strongest && strongest.ablation_top1_after_token) {
    predictedToken = strongest.ablation_top1_after_token;
  }

  counterSummaryEl.textContent =
    `clean_margin=${fmt(base, 4)} | cumulative_ablation_drop=${fmt(drop, 4)} | counterfactual_margin=${fmt(cfMargin, 4)} | approx_top_token=${predictedToken}`;
}

function renderCounterfactualPanel() {
  const prompt = getCurrentPrompt();
  counterComponentsEl.innerHTML = "";
  if (!prompt) {
    counterSummaryEl.textContent = "Toggle component ablations to estimate margin/top-token change.";
    return;
  }

  const ablatable = getCounterfactualComponents(prompt);
  if (!ablatable.length) {
    counterSummaryEl.textContent = "No ablation metadata available for this prompt.";
    return;
  }

  for (const comp of ablatable) {
    const key = componentKey(comp);
    const row = document.createElement("label");
    row.className = "component-row";
    const checked = counterfactualSelection.has(key) ? "checked" : "";
    row.innerHTML =
      `<div class=\"component-title\">` +
      `<span class=\"mono\">${esc(comp.label)}</span>` +
      `<span class=\"badge\">L${safeInt(comp.layer, -1)}</span>` +
      `</div>` +
      `<div><input type=\"checkbox\" data-cf-key=\"${esc(key)}\" ${checked} /> ` +
      `<span class=\"small\">score=${fmt(comp.score, 4)} | abl_drop=${fmt(comp.ablation_drop, 4)}</span></div>`;
    counterComponentsEl.appendChild(row);
  }

  for (const input of counterComponentsEl.querySelectorAll("input[type='checkbox'][data-cf-key]")) {
    input.addEventListener("change", () => {
      const key = String(input.getAttribute("data-cf-key") || "");
      if (input.checked) counterfactualSelection.add(key);
      else counterfactualSelection.delete(key);
      updateCounterfactualSummary();
    });
  }

  updateCounterfactualSummary();
}

function renderWhyPanel() {
  const p = getCurrentPrompt();
  if (!p) {
    whySummaryEl.textContent = "No prompt-level tracker data loaded.";
    supportListEl.innerHTML = "";
    opposeListEl.innerHTML = "";
    pathsListEl.innerHTML = "";
    promptBaseEmphasis = new Set();
    promptBasePath = new Set();
    flowActiveNodeIds = new Set();
    flowActivePathIds = new Set();
    syncHighlightFromState();
    renderTokenLinks();
    renderEquationPanel();
    drawSankeyPanel();
    renderCounterfactualPanel();
    renderResidualPanel();
    return;
  }

  const responseLine = p.response_text
    ? `response_text: ${clipText(p.response_text, 260)}`
    : "response_text: n/a";
  const traceLine = p.prompt_plus_response
    ? `prompt+response: ${clipText(p.prompt_plus_response, 300)}`
    : "";

  whySummaryEl.textContent =
    `prompt_id: ${p.prompt_id}\n` +
    `decision: t${safeInt(p.decision_index, 0)}\n` +
    `task: ${p.task}\n` +
    `target: ${p.target_token} | foil: ${p.foil_token}\n` +
    `response_token: ${p.response_token || "n/a"}\n` +
    `margin: ${fmt(p.clean_margin)} (${confidenceBand(p.clean_margin)})\n` +
    `pred_margin: ${fmt(p.pred_margin)}\n` +
    `reconstruction_error: ${fmt(p.reconstruction_error, 6)}\n\n` +
    `prompt: ${p.prompt}\n` +
    `${responseLine}\n` +
    `${traceLine}`;

  const support = [...(p.components || [])]
    .filter((c) => safeNum(c.score, 0) > 0)
    .sort((a, b) => safeNum(b.score, 0) - safeNum(a.score, 0))
    .slice(0, 5);
  const oppose = [...(p.components || [])]
    .filter((c) => safeNum(c.score, 0) < 0)
    .sort((a, b) => safeNum(a.score, 0) - safeNum(b.score, 0))
    .slice(0, 5);

  renderComponentList(supportListEl, support);
  renderComponentList(opposeListEl, oppose);
  const paths = (p.top_paths || []).slice(0, 3);
  renderPaths(paths, p.response_token || p.target_token || "target");
  applyPromptHighlights(p, support, paths);
  renderTokenLinks();
  renderEquationPanel();
  drawSankeyPanel();
  renderCounterfactualPanel();
}

function pauseFlow() {
  flowState.playing = false;
  flowState.accumulatorMs = 0;
  flowPlayBtn.textContent = "Play";
}

function buildFlowSteps(prompt) {
  if (!prompt) return [];
  const target = prompt.response_token || prompt.target_token || "target";
  const steps = [];

  steps.push({
    title: "Prompt enters residual stream",
    detail: "Token embeddings create the initial representation before transformer blocks process it.",
    nodeIds: viewer.nodeById.has("embed") ? ["embed"] : [],
    pathIds: viewer.nodeById.has("embed") ? ["embed"] : [],
    score: null,
    layer: null,
  });

  const byAbs = [...(prompt.components || [])]
    .filter((c) => Math.abs(safeNum(c.score, 0) || 0) > 1e-9)
    .sort((a, b) => Math.abs(safeNum(b.score, 0) || 0) - Math.abs(safeNum(a.score, 0) || 0))
    .slice(0, 9)
    .sort((a, b) => {
      const la = safeInt(a.layer, -1);
      const lb = safeInt(b.layer, -1);
      if (la !== lb) return la - lb;
      return Math.abs(safeNum(b.score, 0) || 0) - Math.abs(safeNum(a.score, 0) || 0);
    });

  const seen = new Set();
  for (const comp of byAbs) {
    if (seen.has(comp.label)) continue;
    seen.add(comp.label);
    const ids = componentNodeCandidates(comp.label, comp.kind, comp.layer);
    const score = safeNum(comp.score, 0) || 0;
    const src = [...(comp.sources || [])].sort(
      (a, b) => Math.abs(safeNum(b.score, 0) || 0) - Math.abs(safeNum(a.score, 0) || 0)
    )[0];

    let detail = "";
    if (comp.kind === "attn_out" && src) {
      detail = `Attention at layer ${comp.layer} reads source token "${src.token}" (pos ${src.position}) and ${score >= 0 ? "supports" : "opposes"} ${target}.`;
    } else if (comp.kind === "attn_out") {
      detail = `Attention at layer ${comp.layer} writes a direction that ${score >= 0 ? "supports" : "opposes"} ${target}.`;
    } else if (comp.kind === "mlp_out") {
      detail = `MLP at layer ${comp.layer} transforms residual features and ${score >= 0 ? "supports" : "opposes"} ${target}.`;
    } else {
      detail = `Component ${comp.label} contributes to the final token margin.`;
    }

    steps.push({
      title: `${comp.label} activation`,
      detail,
      nodeIds: ids,
      pathIds: ids,
      score,
      layer: safeInt(comp.layer, -1),
    });
  }

  for (const path of (prompt.top_paths || []).slice(0, 3)) {
    const ids = nodeIdsForPath(path);
    steps.push({
      title: `Causal path from "${path.source_token}"`,
      detail: `Source token "${path.source_token}" at position ${path.source_position} flows through ${path.component_label} and increases logit(${target}).`,
      nodeIds: ids,
      pathIds: ids,
      score: safeNum(path.path_strength, null),
      layer: safeInt(path.layer, -1),
    });
  }

  steps.push({
    title: "Unembedding emits token",
    detail: `Final residual state is read by unembedding to produce "${target}".`,
    nodeIds: viewer.nodeById.has("unembed") ? ["unembed"] : [],
    pathIds: viewer.nodeById.has("unembed") ? ["unembed"] : [],
    score: safeNum(prompt.pred_margin, null),
    layer: null,
  });

  return steps;
}

function renderFlowPanel() {
  const prompt = getCurrentPrompt();
  if (!prompt) {
    decisionBadgeEl.textContent = "No decision";
    flowProgressEl.max = "0";
    flowProgressEl.value = "0";
    flowProgressEl.disabled = true;
    prevDecisionBtn.disabled = true;
    nextDecisionBtn.disabled = true;
    flowPlayBtn.disabled = true;
    flowStepBtn.disabled = true;
    flowResetBtn.disabled = true;
    flowCaptionEl.textContent = "Load tracker data to build activation flow.";
    flowStepsEl.innerHTML = "";
    return;
  }

  const totalDecisions = promptEntries.length;
  decisionBadgeEl.textContent = `${currentPromptIndex + 1}/${totalDecisions} ${prompt.prompt_id}#t${safeInt(prompt.decision_index, 0)}`;
  prevDecisionBtn.disabled = totalDecisions <= 1;
  nextDecisionBtn.disabled = totalDecisions <= 1;

  const totalSteps = flowState.steps.length;
  flowProgressEl.max = String(Math.max(0, totalSteps - 1));
  flowProgressEl.value = String(Math.max(0, Math.min(flowState.index, Math.max(0, totalSteps - 1))));
  flowProgressEl.disabled = totalSteps <= 0;
  flowPlayBtn.disabled = totalSteps <= 1;
  flowStepBtn.disabled = totalSteps <= 0;
  flowResetBtn.disabled = totalSteps <= 0;
  flowPlayBtn.textContent = flowState.playing ? "Pause" : "Play";

  if (!totalSteps) {
    flowCaptionEl.textContent = "No flow steps for this decision.";
    flowStepsEl.innerHTML = "";
    return;
  }

  const step = flowState.steps[Math.max(0, Math.min(flowState.index, totalSteps - 1))];
  const scoreLine = step.score === null || step.score === undefined ? "" : `contribution=${fmt(step.score, 4)}`;

  flowCaptionEl.textContent =
    `Step ${flowState.index + 1}/${totalSteps}: ${step.title}\n` +
    `${step.detail}${scoreLine ? `\n${scoreLine}` : ""}`;

  flowStepsEl.innerHTML = "";
  for (let i = 0; i < totalSteps; i += 1) {
    const s = flowState.steps[i];
    const meta = s.score === null || s.score === undefined
      ? "structural"
      : `score=${fmt(s.score, 4)}${s.layer !== null && s.layer !== undefined ? ` | layer=${s.layer}` : ""}`;

    const item = document.createElement("div");
    item.className = `flow-step${i === flowState.index ? " active" : ""}`;
    item.innerHTML =
      `<div class="flow-step-title">${i + 1}. ${esc(s.title)}</div>` +
      `<div class="flow-step-meta">${esc(meta)}</div>`;
    item.addEventListener("click", () => {
      pauseFlow();
      setFlowStep(i, true);
    });
    flowStepsEl.appendChild(item);
  }
}

function setFlowStep(index, shouldFocus = true) {
  if (!flowState.steps.length) {
    flowState.index = 0;
    flowActiveNodeIds = new Set();
    flowActivePathIds = new Set();
    syncHighlightFromState();
    renderFlowPanel();
    renderEquationPanel();
    return;
  }

  const clamped = Math.max(0, Math.min(flowState.steps.length - 1, safeInt(index, 0)));
  flowState.index = clamped;
  const step = flowState.steps[clamped];

  flowActiveNodeIds = new Set(step.nodeIds || []);
  flowActivePathIds = new Set(step.pathIds || step.nodeIds || []);
  syncHighlightFromState();

  if (shouldFocus) {
    const first = [...flowActiveNodeIds, ...flowActivePathIds][0];
    if (first) {
      const node = viewer.nodeById.get(first);
      if (node) selectNode(node, true);
    }
  }

  renderFlowPanel();
  renderEquationPanel();
}

function rebuildFlowForCurrentPrompt(resetIndex = true, shouldFocus = true) {
  const prompt = getCurrentPrompt();
  flowState.steps = buildFlowSteps(prompt);
  flowState.accumulatorMs = 0;

  if (resetIndex) flowState.index = 0;
  else flowState.index = Math.max(0, Math.min(flowState.index, Math.max(0, flowState.steps.length - 1)));

  if (!flowState.steps.length) {
    flowActiveNodeIds = new Set();
    flowActivePathIds = new Set();
    syncHighlightFromState();
    renderFlowPanel();
    return;
  }

  setFlowStep(flowState.index, shouldFocus);
}

function updateFlowPlayback(dtMs) {
  if (!flowState.playing || flowState.steps.length <= 1) return;
  flowState.accumulatorMs += dtMs * flowState.speed;
  while (flowState.accumulatorMs >= flowState.stepMs) {
    flowState.accumulatorMs -= flowState.stepMs;
    if (flowState.index >= flowState.steps.length - 1) {
      pauseFlow();
      setFlowStep(flowState.steps.length - 1, false);
      return;
    }
    setFlowStep(flowState.index + 1, false);
  }
}

function stepFlow(delta) {
  if (!flowState.steps.length) return;
  const next = flowState.index + safeInt(delta, 0);
  if (next >= flowState.steps.length) {
    pauseFlow();
    setFlowStep(flowState.steps.length - 1, true);
    return;
  }
  if (next < 0) {
    setFlowStep(0, true);
    return;
  }
  setFlowStep(next, true);
}

function setCurrentPromptByIndex(index, shouldFocus = true) {
  if (!promptEntries.length) {
    currentPromptIndex = -1;
    pauseFlow();
    residualState.playing = false;
    residualState.accumulatorMs = 0;
    counterfactualSelection = new Set();
    rebuildResidualForCurrentPrompt(true);
    renderWhyPanel();
    renderFlowPanel();
    return;
  }

  currentPromptIndex = Math.max(0, Math.min(promptEntries.length - 1, safeInt(index, 0)));
  decisionSelectEl.value = String(currentPromptIndex);
  pauseFlow();
  residualState.playing = false;
  residualState.accumulatorMs = 0;
  counterfactualSelection = new Set();
  rebuildResidualForCurrentPrompt(true);
  renderWhyPanel();
  rebuildFlowForCurrentPrompt(true, shouldFocus);
}

function refreshPromptState(payload, statusText = "") {
  const normalized = normalizeTrackerPayload(payload);
  promptEntries = normalized.prompts.map((p) => {
    const byLabel = new Map();
    for (const c of p.components || []) byLabel.set(String(c.label || c.component_label || ""), c);
    return { ...p, byLabel };
  });

  promptCountEl.textContent = String(promptEntries.length);
  decisionSelectEl.innerHTML = "";

  if (!promptEntries.length) {
    decisionSelectEl.disabled = true;
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "No decisions";
    decisionSelectEl.appendChild(opt);
    setCurrentPromptByIndex(-1, false);
  } else {
    decisionSelectEl.disabled = false;
    for (let i = 0; i < promptEntries.length; i += 1) {
      const p = promptEntries[i];
      const opt = document.createElement("option");
      opt.value = String(i);
      opt.textContent = p.display_id || `${p.prompt_id || "unknown"}#t${safeInt(p.decision_index, 0)}`;
      decisionSelectEl.appendChild(opt);
    }
    setCurrentPromptByIndex(0, false);
  }

  if (statusText) statusEl.textContent = statusText;
}

function normalizePayload(raw) {
  if (!raw || typeof raw !== "object") throw new Error("Invalid JSON payload.");

  if (raw.graph && raw.tracker) {
    return {
      graph: raw.graph,
      tracker: raw.tracker,
      summary: raw.summary || null,
      meta: raw.meta || null,
    };
  }

  if (raw.nodes && raw.edges) {
    return {
      graph: { nodes: raw.nodes, edges: raw.edges },
      tracker: { num_prompts: 0, prompts: [] },
      summary: null,
      meta: null,
    };
  }

  if (raw.prompts) {
    return {
      graph: { nodes: [], edges: [] },
      tracker: normalizeTrackerPayload(raw),
      summary: null,
      meta: null,
    };
  }

  throw new Error("Unsupported JSON schema. Use viewer_payload.json.");
}

function normalizePromptObject(rawPrompt) {
  const p = rawPrompt || {};
  const components = (Array.isArray(p.components) ? p.components : []).map((c) => ({
    label: String(c.label || c.component_label || ""),
    kind: String(c.kind || c.component_kind || ""),
    layer: safeInt(c.layer, -1),
    rank: safeInt(c.rank ?? c.component_rank_by_abs, 0),
    score: safeNum(c.score ?? c.component_score, 0.0) ?? 0.0,
    abs_score: safeNum(c.abs_score ?? c.component_abs_score, 0.0) ?? 0.0,
    ablation_drop: safeNum(c.ablation_drop, null),
    ablation_margin: safeNum(c.ablation_margin, null),
    ablation_top1_after_id: c.ablation_top1_after_id,
    ablation_top1_after_token: c.ablation_top1_after_token,
    sources: (Array.isArray(c.sources) ? c.sources : []).map((src) => ({
      position: safeInt(src.position, -1),
      token: String(src.token || ""),
      token_id: src.token_id,
      score: safeNum(src.score, null),
      abs_score: safeNum(src.abs_score, null),
    })),
  }));

  const topPaths = (Array.isArray(p.top_paths) ? p.top_paths : []).map((path) => ({
    source_position: safeInt(path.source_position, -1),
    source_token: String(path.source_token || ""),
    source_score: safeNum(path.source_score, null),
    component_label: String(path.component_label || ""),
    layer: safeInt(path.layer, -1),
    component_kind: String(path.component_kind || ""),
    component_score: safeNum(path.component_score, null),
    path_strength: safeNum(path.path_strength, null),
  }));

  const promptId = String(p.prompt_id || "unknown");
  const decisionIndex = safeInt(p.decision_index ?? p.response_token_index, 0);

  return {
    group_key: String(p.group_key || `${promptId}::t${decisionIndex}`),
    display_id: String(p.display_id || `${promptId}#t${decisionIndex}`),
    prompt_id: promptId,
    prompt_index: safeInt(p.prompt_index, 0),
    decision_index: decisionIndex,
    task: String(p.task || "unknown"),
    prompt: String(p.prompt || ""),
    response_token: String(p.response_token || p.target_token || ""),
    response_text: String(p.response_text || p.generated_response || ""),
    prompt_plus_response: String(p.prompt_plus_response || (String(p.prompt || "") + String(p.response_text || p.generated_response || ""))),
    target_token: String(p.target_token || ""),
    foil_token: String(p.foil_token || ""),
    clean_margin: safeNum(p.clean_margin, null),
    pred_margin: safeNum(p.pred_margin ?? p.pred_margin_from_components, null),
    reconstruction_error: safeNum(p.reconstruction_error ?? p.prompt_reconstruction_error, null),
    components,
    top_paths: topPaths,
  };
}

function buildTrackerPayloadFromRows(rows, maxSourceTokens = 12, maxComponents = 80) {
  const grouped = new Map();
  for (const row of rows || []) {
    if (!row || typeof row !== "object") continue;
    const promptId = String(row.prompt_id || "unknown");
    const idx = row.response_token_index !== undefined
      ? safeInt(row.response_token_index, 0)
      : safeInt(row.decision_index, 0);
    const key = `${promptId}::t${idx}`;
    if (!grouped.has(key)) grouped.set(key, []);
    grouped.get(key).push(row);
  }

  const prompts = [];
  for (const [key, items] of grouped.entries()) {
    const sortedRows = [...items].sort((a, b) => {
      const aPrompt = safeInt(a.prompt_index, 0);
      const bPrompt = safeInt(b.prompt_index, 0);
      if (aPrompt !== bPrompt) return aPrompt - bPrompt;
      const aStep = safeInt(a.response_token_index ?? a.decision_index, 0);
      const bStep = safeInt(b.response_token_index ?? b.decision_index, 0);
      if (aStep !== bStep) return aStep - bStep;
      return safeInt(a.component_rank_by_abs, 1e9) - safeInt(b.component_rank_by_abs, 1e9);
    });

    if (!sortedRows.length) continue;
    const first = sortedRows[0];
    const promptId = String(first.prompt_id || "unknown");
    const decisionIndex = safeInt(first.response_token_index ?? first.decision_index, 0);
    const responseToken = String(first.response_token ?? first.target_token ?? "");
    const responseText = String(first.generated_response ?? first.response_text ?? "");
    const promptPlusResponse = String(first.prompt_plus_response || "") || (String(first.prompt || "") + responseText);

    const components = [];
    for (const row of sortedRows.slice(0, Math.max(1, maxComponents))) {
      const srcRaw = Array.isArray(row.source_token_attributions) ? row.source_token_attributions : [];
      const sources = srcRaw.slice(0, Math.max(1, maxSourceTokens)).map((src) => ({
        position: safeInt(src.position, -1),
        token: String(src.token || ""),
        token_id: src.token_id,
        score: safeNum(src.score, null),
        abs_score: safeNum(src.abs_score, null),
      }));

      components.push({
        label: String(row.component_label || ""),
        kind: String(row.component_kind || ""),
        layer: safeInt(row.layer, -1),
        rank: safeInt(row.component_rank_by_abs, 0),
        score: safeNum(row.component_score, 0.0) ?? 0.0,
        abs_score: safeNum(row.component_abs_score, 0.0) ?? 0.0,
        ablation_drop: safeNum(row.ablation_drop, null),
        ablation_margin: safeNum(row.ablation_margin, null),
        ablation_top1_after_id: row.ablation_top1_after_id,
        ablation_top1_after_token: row.ablation_top1_after_token,
        sources,
      });
    }

    const pathCandidates = [];
    for (const comp of components) {
      if (comp.kind !== "attn_out") continue;
      const compScore = safeNum(comp.score, 0) || 0;
      for (const src of comp.sources || []) {
        const srcScore = safeNum(src.score, null);
        if (srcScore === null) continue;
        if (compScore === 0 || compScore * srcScore <= 0) continue;
        pathCandidates.push({
          source_position: safeInt(src.position, -1),
          source_token: String(src.token || ""),
          source_score: srcScore,
          component_label: comp.label,
          layer: safeInt(comp.layer, -1),
          component_kind: comp.kind,
          component_score: compScore,
          path_strength: Math.abs(srcScore),
        });
      }
    }
    pathCandidates.sort((a, b) => (safeNum(b.path_strength, 0) || 0) - (safeNum(a.path_strength, 0) || 0));

    prompts.push({
      group_key: key,
      prompt_id: promptId,
      display_id: `${promptId}#t${decisionIndex}`,
      prompt_index: safeInt(first.prompt_index, 0),
      decision_index: decisionIndex,
      task: String(first.task || "unknown"),
      prompt: String(first.prompt || ""),
      response_token: responseToken,
      response_text: responseText,
      prompt_plus_response: promptPlusResponse,
      target_token: String(first.target_token || ""),
      foil_token: String(first.foil_token || ""),
      clean_margin: safeNum(first.clean_margin, null),
      pred_margin: safeNum(first.pred_margin_from_components, null),
      reconstruction_error: safeNum(first.prompt_reconstruction_error, null),
      components,
      top_paths: pathCandidates.slice(0, 3),
    });
  }

  prompts.sort((a, b) => {
    const ap = safeInt(a.prompt_index, 0);
    const bp = safeInt(b.prompt_index, 0);
    if (ap !== bp) return ap - bp;
    const as = safeInt(a.decision_index, 0);
    const bs = safeInt(b.decision_index, 0);
    if (as !== bs) return as - bs;
    return String(a.prompt_id || "").localeCompare(String(b.prompt_id || ""));
  });

  return { num_prompts: prompts.length, prompts };
}

function normalizeTrackerPayload(raw) {
  if (!raw) return { num_prompts: 0, prompts: [] };

  if (Array.isArray(raw)) {
    if (!raw.length) return { num_prompts: 0, prompts: [] };
    const looksAggregated = raw[0] && typeof raw[0] === "object" && (
      Array.isArray(raw[0].components) || raw[0].top_paths || raw[0].group_key || raw[0].display_id
    );
    if (looksAggregated) {
      const prompts = raw.map((p) => normalizePromptObject(p));
      return { num_prompts: prompts.length, prompts };
    }
    return buildTrackerPayloadFromRows(raw);
  }

  if (typeof raw === "object") {
    if (raw.tracker && typeof raw.tracker === "object") {
      return normalizeTrackerPayload(raw.tracker);
    }
    if (Array.isArray(raw.prompts)) {
      const prompts = raw.prompts.map((p) => normalizePromptObject(p));
      return { num_prompts: prompts.length, prompts };
    }
    if (Array.isArray(raw.rows)) {
      return buildTrackerPayloadFromRows(raw.rows);
    }
    if (raw.component_label || raw.component_kind) {
      return buildTrackerPayloadFromRows([raw]);
    }
  }

  throw new Error("Unsupported tracker format.");
}

function parseJsonlRows(text) {
  const rows = [];
  const lines = String(text || "").split(/\r?\n/);
  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i].trim();
    if (!line) continue;
    let row = null;
    try {
      row = JSON.parse(line);
    } catch (_err) {
      throw new Error(`Invalid JSONL at line ${i + 1}`);
    }
    if (row && typeof row === "object") rows.push(row);
  }
  return rows;
}

function parseTrackerText(text, fileName) {
  const trimmed = String(text || "").trim();
  if (!trimmed) throw new Error("Tracker file is empty.");

  const name = String(fileName || "").toLowerCase();
  const expectJsonl = name.endsWith(".jsonl");
  if (!expectJsonl) {
    try {
      return normalizeTrackerPayload(JSON.parse(trimmed));
    } catch (_err) {
      // Fall through to JSONL parser.
    }
  }

  const rows = parseJsonlRows(trimmed);
  return buildTrackerPayloadFromRows(rows);
}

function setCurrentPromptListFromState() {
  const normalized = normalizeTrackerPayload(state.tracker || { num_prompts: 0, prompts: [] });
  promptEntries = normalized.prompts.map((p) => {
    const byLabel = new Map();
    for (const c of p.components || []) {
      byLabel.set(String(c.label || c.component_label || ""), c);
    }
    return { ...p, byLabel };
  });

  promptCountEl.textContent = String(promptEntries.length);
  decisionSelectEl.innerHTML = "";

  if (!promptEntries.length) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "No decisions";
    decisionSelectEl.appendChild(opt);
    decisionSelectEl.disabled = true;
    setCurrentPromptByIndex(-1, false);
    return;
  }

  decisionSelectEl.disabled = false;
  for (let i = 0; i < promptEntries.length; i += 1) {
    const p = promptEntries[i];
    const opt = document.createElement("option");
    opt.value = String(i);
    opt.textContent = p.display_id || `${p.prompt_id || "unknown"}#t${safeInt(p.decision_index, 0)}`;
    decisionSelectEl.appendChild(opt);
  }

  setCurrentPromptByIndex(0, false);
}

function renderAll() {
  applyGraph(state.graph);
  setCurrentPromptListFromState();
  refreshMapStats();
}

function handleFind() {
  const q = String(searchInput.value || "").trim().toLowerCase();
  if (!q) return;

  const node = viewer.nodes.find(
    (n) => n.id.toLowerCase().includes(q) || n.label.toLowerCase().includes(q)
  );

  if (!node) {
    statusEl.textContent = `No node found for '${q}'.`;
    return;
  }

  if (!isNodeVisible(node)) {
    statusEl.textContent = `${node.id} exists but is hidden by current filters.`;
  } else {
    statusEl.textContent = `Found ${node.id}.`;
  }

  selectNode(node, true);
}

async function loadBundleFile(file) {
  hasExternalFileLoad = true;
  const text = await file.text();
  let raw = null;
  try {
    raw = JSON.parse(text);
  } catch (_err) {
    throw new Error("Selected file is not valid JSON.");
  }

  state = normalizePayload(raw);
  resetCamera();
  clearSelection();
  renderAll();
  statusEl.textContent = `Loaded ${file.name}`;
}

async function loadTrackerFile(file) {
  hasExternalFileLoad = true;
  const text = await file.text();
  const payload = parseTrackerText(text, file.name);
  state.tracker = payload;
  clearSelection();
  setCurrentPromptListFromState();
  statusEl.textContent = `Loaded ${file.name} (${payload.num_prompts || 0} prompt decisions)`;
}

async function loadInputFile(file) {
  hasExternalFileLoad = true;
  const lower = String(file?.name || "").toLowerCase();
  if (lower.endsWith(".jsonl")) {
    await loadTrackerFile(file);
    return;
  }

  const text = await file.text();
  let raw = null;
  try {
    raw = JSON.parse(text);
  } catch (_err) {
    throw new Error("Selected file is not valid JSON/JSONL.");
  }

  if (
    raw &&
    typeof raw === "object" &&
    (
      Array.isArray(raw.prompts) ||
      Array.isArray(raw.rows) ||
      raw.component_label ||
      raw.component_kind
    ) &&
    !(raw.graph && raw.tracker)
  ) {
    state.tracker = normalizeTrackerPayload(raw);
    clearSelection();
    setCurrentPromptListFromState();
    statusEl.textContent = `Loaded ${file.name} (${state.tracker.num_prompts || 0} prompt decisions)`;
    return;
  }

  // Fallback: treat as full payload or graph JSON.
  state = normalizePayload(raw);
  resetCamera();
  clearSelection();
  renderAll();
  statusEl.textContent = `Loaded ${file.name}`;
}

async function loadDefaultGraphFromStatic(force = false) {
  if (hasExternalFileLoad && !force) return false;
  try {
    const resp = await fetch("default_graph.json", { cache: "no-store" });
    if (!resp.ok) return false;
    const graph = await resp.json();
    if (!graph || !Array.isArray(graph.nodes) || !Array.isArray(graph.edges)) return false;
    // Guard again after async fetch to avoid clobbering a user-loaded payload.
    if (hasExternalFileLoad && !force) return false;

    state.graph = graph;
    state.summary = null;
    state.meta = null;
    if (!state.tracker || !Array.isArray(state.tracker.prompts)) {
      state.tracker = { num_prompts: 0, prompts: [] };
    }
    resetCamera();
    clearSelection();
    renderAll();
    statusEl.textContent = "Loaded default model graph.";
    return true;
  } catch (_err) {
    return false;
  }
}

function loadDemo() {
  state = {
    graph: {
      nodes: [
        { id: "embed", kind: "embed", x: -2.2, y: 0.15, z: 0, size: 28, color: "#6BCBEB" },
        { id: "L0_H3", kind: "head", layer: 0, x: -0.98, y: 1.05, z: 1.1, size: 16, color: "#88C7FF" },
        { id: "L0_attn", kind: "attn", layer: 0, x: -0.8, y: 0.62, z: 0.8, size: 21, color: "#4BA3C3" },
        { id: "L0_resid", kind: "resid", layer: 0, x: -0.75, y: 0.0, z: 0.05, size: 22, color: "#F4D35E" },
        { id: "L0_mlp", kind: "mlp", layer: 0, x: -0.65, y: -0.62, z: -0.8, size: 21, color: "#8D8DE8" },
        { id: "L1_H5", kind: "head", layer: 1, x: 0.58, y: 1.02, z: 1.02, size: 16, color: "#88C7FF" },
        { id: "L1_attn", kind: "attn", layer: 1, x: 0.8, y: 0.62, z: 0.74, size: 21, color: "#4BA3C3" },
        { id: "L1_resid", kind: "resid", layer: 1, x: 0.82, y: 0.0, z: 0.0, size: 22, color: "#F4D35E" },
        { id: "L1_mlp", kind: "mlp", layer: 1, x: 0.92, y: -0.62, z: -0.78, size: 21, color: "#8D8DE8" },
        { id: "unembed", kind: "unembed", x: 2.25, y: 0.1, z: 0.04, size: 28, color: "#F2A65A" },
      ],
      edges: [
        { source: "embed", target: "L0_resid", kind: "residual_bus" },
        { source: "L0_H3", target: "L0_attn", kind: "head_to_attn" },
        { source: "L0_attn", target: "L0_resid", kind: "residual_write" },
        { source: "L0_mlp", target: "L0_resid", kind: "residual_write" },
        { source: "L0_resid", target: "L1_resid", kind: "residual_bus" },
        { source: "L1_H5", target: "L1_attn", kind: "head_to_attn" },
        { source: "L1_attn", target: "L1_resid", kind: "residual_write" },
        { source: "L1_mlp", target: "L1_resid", kind: "residual_write" },
        { source: "L1_resid", target: "unembed", kind: "residual_bus" },
      ],
    },
    tracker: {
      num_prompts: 1,
      prompts: [
        {
          display_id: "my_copy_001#t0",
          prompt_id: "my_copy_001",
          decision_index: 0,
          task: "copy",
          prompt: "The secret code is 73914. Repeat the secret code exactly:",
          response_token: "\\n",
          response_text: "\\n",
          prompt_plus_response: "The secret code is 73914. Repeat the secret code exactly:\\n",
          target_token: "\\n",
          foil_token: " 7",
          clean_margin: 1.1082,
          pred_margin: 1.1082,
          reconstruction_error: 0.000001,
          components: [
            { label: "1_mlp_out", kind: "mlp_out", layer: 1, score: 1.24, ablation_drop: 1.19, sources: [] },
            { label: "1_attn_out", kind: "attn_out", layer: 1, score: -1.57, ablation_drop: -0.51, sources: [{ position: 5, token: " 7", score: -1.62, abs_score: 1.62 }] },
          ],
          top_paths: [
            {
              source_position: 5,
              source_token: " 7",
              source_score: -1.6248,
              component_label: "1_attn_out",
              layer: 1,
              component_kind: "attn_out",
              component_score: -1.575,
              path_strength: 1.6248,
            },
          ],
        },
      ],
    },
    summary: null,
    meta: { model: "demo" },
  };

  renderAll();
  statusEl.textContent = "Loaded demo payload.";
}

fileInput.addEventListener("change", async () => {
  const file = fileInput.files && fileInput.files[0];
  if (!file) return;
  hasExternalFileLoad = true;
  statusEl.textContent = `Loading ${file.name}...`;
  try {
    await loadInputFile(file);
  } catch (err) {
    statusEl.textContent = `Load failed: ${err.message || "unknown error"}`;
  }
});

loadDemoBtn.addEventListener("click", () => {
  loadDefaultGraphFromStatic(true).then((ok) => {
    if (!ok) loadDemo();
  });
});

complexitySelect.addEventListener("change", () => {
  if (viewer.selectedId) {
    const node = viewer.nodeById.get(viewer.selectedId);
    if (!node || !isNodeVisible(node)) {
      selectNode(null, false);
    }
  }
  refreshMapStats();
});

findBtn.addEventListener("click", () => {
  handleFind();
});

searchInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    handleFind();
  }
});

resetCameraBtn.addEventListener("click", () => {
  resetCamera();
});

clearSelectionBtn.addEventListener("click", () => {
  clearSelection();
});

decisionSelectEl.addEventListener("change", () => {
  const idx = safeInt(decisionSelectEl.value, -1);
  if (idx >= 0) setCurrentPromptByIndex(idx, true);
});

prevDecisionBtn.addEventListener("click", () => {
  if (promptEntries.length <= 1) return;
  setCurrentPromptByIndex(currentPromptIndex - 1, true);
});

nextDecisionBtn.addEventListener("click", () => {
  if (promptEntries.length <= 1) return;
  setCurrentPromptByIndex(currentPromptIndex + 1, true);
});

flowPlayBtn.addEventListener("click", () => {
  if (flowState.steps.length <= 1) return;
  flowState.playing = !flowState.playing;
  flowState.accumulatorMs = 0;
  renderFlowPanel();
});

flowStepBtn.addEventListener("click", () => {
  pauseFlow();
  stepFlow(1);
});

flowResetBtn.addEventListener("click", () => {
  pauseFlow();
  setFlowStep(0, true);
});

flowSpeedSelect.addEventListener("change", () => {
  const speed = safeNum(flowSpeedSelect.value, 1.0);
  flowState.speed = speed && speed > 0 ? speed : 1.0;
  renderFlowPanel();
});

flowProgressEl.addEventListener("input", () => {
  pauseFlow();
  setFlowStep(safeInt(flowProgressEl.value, 0), false);
});

residPlayBtn.addEventListener("click", () => {
  if (residualState.rows.length <= 1) return;
  residualState.playing = !residualState.playing;
  residualState.accumulatorMs = 0;
  renderResidualPanel();
});

residStepBtn.addEventListener("click", () => {
  residualState.playing = false;
  residualState.accumulatorMs = 0;
  setResidualStep(residualState.index + 1, true);
});

residResetBtn.addEventListener("click", () => {
  residualState.playing = false;
  residualState.accumulatorMs = 0;
  setResidualStep(0, true);
});

residProgressEl.addEventListener("input", () => {
  residualState.playing = false;
  residualState.accumulatorMs = 0;
  setResidualStep(safeInt(residProgressEl.value, 0), false);
});

counterClearBtn.addEventListener("click", () => {
  counterfactualSelection = new Set();
  renderCounterfactualPanel();
});

window.addEventListener("resize", () => {
  resizeViewer();
  renderEquationPanel();
  drawSankeyPanel();
  renderResidualPanel();
});

initCanvasInteractions();
resizeViewer();
resetCamera();
flowState.speed = safeNum(flowSpeedSelect.value, 1.0) || 1.0;

requestAnimationFrame(animate);

loadDefaultGraphFromStatic().then((ok) => {
  if (!ok) loadDemo();
});
