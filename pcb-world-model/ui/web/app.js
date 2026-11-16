// Interactive PCB Engineering World Model Canvas & WebSocket Client

const CHANNEL_META = [
  { key: "board_outline", label: "Ch0 Outline", color: [56, 189, 248] },
  { key: "courtyard_mask", label: "Ch1 Courtyard", color: [148, 163, 184] },
  { key: "copper_L1", label: "Ch2 Copper L1", color: [244, 63, 94] },
  { key: "copper_L2", label: "Ch3 Copper L2", color: [56, 189, 248] },
  { key: "via_mask", label: "Ch4 Vias", color: [245, 158, 11] },
  { key: "active_net_pins", label: "Ch5 Active Pins", color: [16, 185, 129] },
  { key: "unrouted_demand", label: "Ch6 Demand", color: [251, 146, 60] },
  { key: "congestion_L1", label: "M_t+1 Congest", color: [168, 85, 247] },
];

const NET_COLORS = [
  "#f43f5e", "#38bdf8", "#10b981", "#f59e0b",
  "#a855f7", "#ec4899", "#22d3ee", "#84cc16",
];

const state = {
  presets: [],
  board: null,
  channels: {},
  congestionL1: null,
  congestionL2: null,
  drcPenalty: null,
  drcErrors: [],
  soloChannel: null,
  viewMode: "2d", // "2d" | "3d"
  zoom: 1.0,
  panX: 0,
  panY: 0,
  tiltDeg: 44,
  yawDeg: -24,
  isDragging: false,
  lastPointer: { x: 0, y: 0 },
  running: false,
  ws: null,
  kicadText: "",
  layers: {
    l1: true,
    l2: true,
    vias: true,
    courtyard: true,
    ratsnest: true,
    congestion: true,
    penalty: true,
  },
};

const els = {
  presetSelect: document.getElementById("preset-select"),
  candidatesRange: document.getElementById("candidates-range"),
  candidatesVal: document.getElementById("candidates-val"),
  paceRange: document.getElementById("pace-range"),
  paceVal: document.getElementById("pace-val"),
  btnView2d: document.getElementById("btn-view-2d"),
  btnView3d: document.getElementById("btn-view-3d"),
  btnRun: document.getElementById("btn-run"),
  btnInspectKicad: document.getElementById("btn-inspect-kicad"),
  btnResetView: document.getElementById("btn-reset-view"),
  btnClearLog: document.getElementById("btn-clear-log"),
  canvasStage: document.getElementById("canvas-stage"),
  canvas: document.getElementById("pcb-canvas"),
  cursorCoords: document.getElementById("cursor-coords"),
  statusDot: document.getElementById("status-dot"),
  statusText: document.getElementById("status-text"),
  hud3dControls: document.getElementById("hud-3d-controls"),
  tiltRange: document.getElementById("tilt-range"),
  yawRange: document.getElementById("yaw-range"),
  tensorGrid: document.getElementById("tensor-grid"),
  overlayLabel: document.getElementById("overlay-label"),
  drcBadge: document.getElementById("drc-badge"),
  valCritic: document.getElementById("val-critic"),
  criticProgress: document.getElementById("critic-progress"),
  valNets: document.getElementById("val-nets"),
  valDrc: document.getElementById("val-drc"),
  valHpwl: document.getElementById("val-hpwl"),
  valWire: document.getElementById("val-wire"),
  valVias: document.getElementById("val-vias"),
  valIter: document.getElementById("val-iter"),
  candTbody: document.getElementById("cand-tbody"),
  candCountTag: document.getElementById("cand-count-tag"),
  eventLog: document.getElementById("event-log"),
  kicadDialog: document.getElementById("kicad-dialog"),
  kicadCode: document.getElementById("kicad-code"),
  btnCopyKicad: document.getElementById("btn-copy-kicad"),
  btnDownloadKicad: document.getElementById("btn-download-kicad"),
  btnCloseDialog: document.getElementById("btn-close-dialog"),
};

const miniCanvases = {};

function initTensorStrip() {
  els.tensorGrid.innerHTML = "";
  for (const ch of CHANNEL_META) {
    const card = document.createElement("button");
    card.type = "button";
    card.className = "tensor-card";
    card.dataset.channel = ch.key;

    const cvs = document.createElement("canvas");
    cvs.width = 50;
    cvs.height = 50;
    miniCanvases[ch.key] = cvs;

    const lbl = document.createElement("span");
    lbl.textContent = ch.label;

    card.appendChild(cvs);
    card.appendChild(lbl);
    card.addEventListener("click", () => {
      if (state.soloChannel === ch.key) {
        state.soloChannel = null;
        els.overlayLabel.textContent = "Click any channel to solo-overlay on board";
      } else {
        state.soloChannel = ch.key;
        els.overlayLabel.textContent = `Overlaying ${ch.label} (click again to clear)`;
      }
      document.querySelectorAll(".tensor-card").forEach((el) => {
        el.classList.toggle("selected", el.dataset.channel === state.soloChannel);
      });
      renderCanvas();
    });
    els.tensorGrid.appendChild(card);
  }
}

function renderMiniHeatmaps() {
  for (const ch of CHANNEL_META) {
    const cvs = miniCanvases[ch.key];
    const grid = ch.key === "congestion_L1" ? state.congestionL1 : state.channels[ch.key];
    if (!cvs) continue;
    const ctx = cvs.getContext("2d");
    const w = cvs.width;
    const h = cvs.height;
    ctx.fillStyle = "#070b12";
    ctx.fillRect(0, 0, w, h);
    if (!grid || !grid.length) continue;

    const rows = grid.length;
    const cols = grid[0].length;
    const img = ctx.createImageData(cols, rows);
    const [r, g, b] = ch.color;
    for (let y = 0; y < rows; y++) {
      for (let x = 0; x < cols; x++) {
        const val = Math.max(0, Math.min(1, grid[y][x]));
        const idx = (y * cols + x) * 4;
        img.data[idx] = Math.round(7 + val * (r - 7));
        img.data[idx + 1] = Math.round(11 + val * (g - 11));
        img.data[idx + 2] = Math.round(18 + val * (b - 18));
        img.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }
}

function appendLog(text, level = "info") {
  const li = document.createElement("li");
  if (level !== "info") li.className = level;
  const ts = new Date().toLocaleTimeString([], { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" });
  li.textContent = `[${ts}] ${text}`;
  els.eventLog.prepend(li);
  while (els.eventLog.children.length > 80) {
    els.eventLog.removeChild(els.eventLog.lastChild);
  }
}

function setStatus(mode, text) {
  els.statusDot.className = `dot ${mode}`;
  els.statusText.textContent = text;
}

function updateTelemetry(board, extras = {}) {
  if (!board) return;
  els.valNets.textContent = `${board.routed_nets} / ${board.total_nets}`;
  els.valHpwl.textContent = `${board.hpwl_mm.toFixed(1)} mm`;
  els.valWire.textContent = `${board.routed_wirelength_mm.toFixed(1)} mm`;
  els.valVias.textContent = String(board.via_count);

  if (extras.iteration !== undefined) {
    els.valIter.textContent = `${extras.iteration} / 5`;
  }
  if (extras.value !== undefined) {
    const pct = Math.round(extras.value * 100);
    els.valCritic.textContent = `${extras.value.toFixed(3)} (${pct}%)`;
    els.criticProgress.value = pct;
  }
  if (extras.errorCount !== undefined) {
    els.valDrc.textContent = String(extras.errorCount);
    if (extras.completed) {
      if (extras.errorCount === 0 && board.routed_nets === board.total_nets) {
        els.drcBadge.className = "drc-badge pass";
        els.drcBadge.textContent = "DRC CLEAN";
      } else {
        els.drcBadge.className = "drc-badge fail";
        els.drcBadge.textContent = `${extras.errorCount} VIOLATIONS`;
      }
    }
  }
}

function renderCandidatesTable(candidates, winnerId = null) {
  if (!candidates || !candidates.length) return;
  els.candTbody.innerHTML = "";
  for (const c of candidates) {
    const tr = document.createElement("tr");
    if (winnerId !== null && c.candidate_id === winnerId) {
      tr.className = "winner";
    }
    tr.innerHTML = `
      <td>#${c.candidate_id}</td>
      <td>${c.world_model_value.toFixed(3)}</td>
      <td>${c.routability_score.toFixed(3)}</td>
      <td>${c.hpwl_mm.toFixed(1)}</td>
    `;
    els.candTbody.appendChild(tr);
  }
}

function drawHeatmapOverlay(ctx, grid, x0, y0, boardPxW, boardPxH, rgb, alphaMax) {
  if (!grid || !grid.length) return;
  const rows = grid.length;
  const cols = grid[0].length;
  const cw = boardPxW / cols;
  const ch = boardPxH / rows;
  const [r, g, b] = rgb;
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      const v = grid[y][x];
      if (v <= 0.04) continue;
      const a = Math.min(alphaMax, v * alphaMax);
      ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${a.toFixed(3)})`;
      ctx.fillRect(x0 + x * cw, y0 + y * ch, cw + 0.5, ch + 0.5);
    }
  }
}

function render2D(ctx, width, height) {
  const board = state.board;
  if (!board) return;

  const padPx = 36;
  const baseScale = Math.min(
    (width - 2 * padPx) / board.width_mm,
    (height - 2 * padPx) / board.height_mm
  );
  const scale = baseScale * state.zoom;
  const boardPxW = board.width_mm * scale;
  const boardPxH = board.height_mm * scale;
  const originX = (width - boardPxW) * 0.5 + state.panX;
  const originY = (height - boardPxH) * 0.5 + state.panY;

  const toPxX = (mm) => originX + mm * scale;
  const toPxY = (mm) => originY + mm * scale;

  // FR-4 Solder Mask Substrate
  ctx.fillStyle = "#0b151e";
  ctx.fillRect(originX, originY, boardPxW, boardPxH);

  // Grid (1mm minor, 5mm major)
  ctx.lineWidth = 1;
  for (let x = 0; x <= board.width_mm; x += 2.5) {
    ctx.strokeStyle = x % 10 === 0 ? "rgba(148, 163, 184, 0.14)" : "rgba(148, 163, 184, 0.05)";
    ctx.beginPath();
    ctx.moveTo(toPxX(x), originY);
    ctx.lineTo(toPxX(x), originY + boardPxH);
    ctx.stroke();
  }
  for (let y = 0; y <= board.height_mm; y += 2.5) {
    ctx.strokeStyle = y % 10 === 0 ? "rgba(148, 163, 184, 0.14)" : "rgba(148, 163, 184, 0.05)";
    ctx.beginPath();
    ctx.moveTo(originX, toPxY(y));
    ctx.lineTo(originX + boardPxW, toPxY(y));
    ctx.stroke();
  }

  // Optional World Model Congestion M_{t+1} & DRC Penalty Overlays
  if (state.layers.congestion && state.congestionL1) {
    drawHeatmapOverlay(ctx, state.congestionL1, originX, originY, boardPxW, boardPxH, [168, 85, 247], 0.36);
  }
  if (state.layers.penalty && state.drcPenalty) {
    drawHeatmapOverlay(ctx, state.drcPenalty, originX, originY, boardPxW, boardPxH, [239, 68, 68], 0.48);
  }
  if (state.soloChannel) {
    const chMeta = CHANNEL_META.find((c) => c.key === state.soloChannel);
    const grid = state.soloChannel === "congestion_L1" ? state.congestionL1 : state.channels[state.soloChannel];
    if (chMeta && grid) {
      drawHeatmapOverlay(ctx, grid, originX, originY, boardPxW, boardPxH, chMeta.color, 0.55);
    }
  }

  // Edge.Cuts Board Outline
  ctx.strokeStyle = "#eab308";
  ctx.lineWidth = 2;
  ctx.strokeRect(originX, originY, boardPxW, boardPxH);

  // Rat's Nest (Unrouted MST segments)
  if (state.layers.ratsnest && board.ratsnest) {
    ctx.save();
    ctx.strokeStyle = "rgba(148, 163, 184, 0.42)";
    ctx.lineWidth = 1.1;
    ctx.setLineDash([4, 4]);
    for (const r of board.ratsnest) {
      ctx.beginPath();
      ctx.moveTo(toPxX(r.x1_mm), toPxY(r.y1_mm));
      ctx.lineTo(toPxX(r.x2_mm), toPxY(r.y2_mm));
      ctx.stroke();
    }
    ctx.restore();
  }

  // B.Cu (Layer 1) Bottom Copper Traces
  if (state.layers.l2 && board.traces) {
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    for (const seg of board.traces) {
      if (seg.layer !== 1) continue;
      ctx.strokeStyle = "#38bdf8";
      ctx.lineWidth = Math.max(1.5, seg.width_mm * scale);
      ctx.beginPath();
      ctx.moveTo(toPxX(seg.x1_mm), toPxY(seg.y1_mm));
      ctx.lineTo(toPxX(seg.x2_mm), toPxY(seg.y2_mm));
      ctx.stroke();
    }
  }

  // F.Cu (Layer 0) Top Copper Traces
  if (state.layers.l1 && board.traces) {
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    for (const seg of board.traces) {
      if (seg.layer !== 0) continue;
      ctx.strokeStyle = "#f43f5e";
      ctx.lineWidth = Math.max(1.5, seg.width_mm * scale);
      ctx.beginPath();
      ctx.moveTo(toPxX(seg.x1_mm), toPxY(seg.y1_mm));
      ctx.lineTo(toPxX(seg.x2_mm), toPxY(seg.y2_mm));
      ctx.stroke();
    }
  }

  // Component Courtyards & Pads
  for (const comp of board.components) {
    const cx = toPxX(comp.x_mm);
    const cy = toPxY(comp.y_mm);
    const cw = comp.width * scale;
    const ch = comp.height * scale;

    if (state.layers.courtyard) {
      ctx.fillStyle = comp.is_locked ? "rgba(245, 158, 11, 0.08)" : "rgba(100, 116, 139, 0.14)";
      ctx.strokeStyle = comp.is_locked ? "rgba(245, 158, 11, 0.72)" : "rgba(148, 163, 184, 0.55)";
      ctx.lineWidth = 1.25;
      ctx.fillRect(cx - cw / 2, cy - ch / 2, cw, ch);
      ctx.strokeRect(cx - cw / 2, cy - ch / 2, cw, ch);
    }

    // Component Pads
    for (const pad of comp.pads) {
      const px = toPxX(pad.x_mm);
      const py = toPxY(pad.y_mm);
      const pw = Math.max(3, pad.width * scale);
      const ph = Math.max(3, pad.height * scale);

      if (pad.layer_mask === 3) {
        // Through-hole copper annulus + drill hole
        ctx.fillStyle = "#f59e0b";
        ctx.beginPath();
        ctx.arc(px, py, Math.max(pw, ph) * 0.48, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = "#070b12";
        ctx.beginPath();
        ctx.arc(px, py, Math.max(pw, ph) * 0.24, 0, Math.PI * 2);
        ctx.fill();
      } else {
        ctx.fillStyle = pad.net_id > 0 ? "#fbbf24" : "#64748b";
        ctx.fillRect(px - pw / 2, py - ph / 2, pw, ph);
      }
    }

    // Silkscreen RefDes label
    ctx.fillStyle = "#e2e8f0";
    ctx.font = `600 ${Math.max(10, Math.min(13, Math.round(scale * 1.15)))}px "JetBrains Mono", monospace`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(comp.comp_id, cx, cy);
  }

  // Plated Through-Hole Vias
  if (state.layers.vias && board.vias) {
    for (const via of board.vias) {
      const vx = toPxX(via.x_mm);
      const vy = toPxY(via.y_mm);
      const vr = Math.max(2.6, (via.diameter_mm * 0.5) * scale);
      ctx.fillStyle = "#f59e0b";
      ctx.beginPath();
      ctx.arc(vx, vy, vr, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#070b12";
      ctx.beginPath();
      ctx.arc(vx, vy, vr * 0.45, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // DRC Error Markers
  if (state.drcErrors && state.drcErrors.length) {
    for (const err of state.drcErrors) {
      const ex = toPxX(err.x_mm);
      const ey = toPxY(err.y_mm);
      ctx.strokeStyle = "#ef4444";
      ctx.lineWidth = 2.2;
      ctx.beginPath();
      ctx.arc(ex, ey, 8, 0, Math.PI * 2);
      ctx.moveTo(ex - 5, ey - 5);
      ctx.lineTo(ex + 5, ey + 5);
      ctx.moveTo(ex + 5, ey - 5);
      ctx.lineTo(ex - 5, ey + 5);
      ctx.stroke();
    }
  }

  state._viewport = { originX, originY, scale, boardPxW, boardPxH };
}

function render3D(ctx, width, height) {
  const board = state.board;
  if (!board) return;

  const yaw = (state.yawDeg * Math.PI) / 180;
  const tilt = (state.tiltDeg * Math.PI) / 180;
  const cosY = Math.cos(yaw);
  const sinY = Math.sin(yaw);
  const sinT = Math.sin(tilt);
  const cosT = Math.cos(tilt);

  const scale = Math.min(width, height) / 72 * state.zoom;
  const cxCenter = width * 0.5 + state.panX;
  const cyCenter = height * 0.53 + state.panY;

  const proj = (xMm, yMm, zMm) => {
    const dx = xMm - board.width_mm * 0.5;
    const dy = yMm - board.height_mm * 0.5;
    const rx = dx * cosY - dy * sinY;
    const ry = dx * sinY + dy * cosY;
    const sx = cxCenter + rx * scale;
    const sy = cyCenter + ry * sinT * scale - zMm * cosT * scale * 1.4;
    return [sx, sy];
  };

  const drawQuad = (z, fill, stroke) => {
    const p0 = proj(0, 0, z);
    const p1 = proj(board.width_mm, 0, z);
    const p2 = proj(board.width_mm, board.height_mm, z);
    const p3 = proj(0, board.height_mm, z);
    ctx.beginPath();
    ctx.moveTo(p0[0], p0[1]);
    ctx.lineTo(p1[0], p1[1]);
    ctx.lineTo(p2[0], p2[1]);
    ctx.lineTo(p3[0], p3[1]);
    ctx.closePath();
    if (fill) { ctx.fillStyle = fill; ctx.fill(); }
    if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = 1.5; ctx.stroke(); }
  };

  // Bottom B.Cu Copper Plane (z = -2.4mm exploded stackup)
  drawQuad(-2.4, "rgba(15, 23, 42, 0.72)", "rgba(56, 189, 248, 0.55)");

  if (state.layers.l2 && board.traces) {
    ctx.lineCap = "round";
    for (const seg of board.traces) {
      if (seg.layer !== 1) continue;
      const a = proj(seg.x1_mm, seg.y1_mm, -2.4);
      const b = proj(seg.x2_mm, seg.y2_mm, -2.4);
      ctx.strokeStyle = "#38bdf8";
      ctx.lineWidth = Math.max(1.5, seg.width_mm * scale * 0.85);
      ctx.beginPath();
      ctx.moveTo(a[0], a[1]);
      ctx.lineTo(b[0], b[1]);
      ctx.stroke();
    }
  }

  // Translucent FR-4 Core
  drawQuad(-1.1, "rgba(16, 185, 129, 0.08)", "rgba(16, 185, 129, 0.25)");

  // Top F.Cu Plane (z = 0)
  drawQuad(0.0, "rgba(12, 24, 34, 0.82)", "#eab308");

  // Via Copper Barrels bridging L1 <-> L2
  if (state.layers.vias && board.vias) {
    for (const via of board.vias) {
      const top = proj(via.x_mm, via.y_mm, 0.0);
      const bot = proj(via.x_mm, via.y_mm, -2.4);
      ctx.strokeStyle = "#f59e0b";
      ctx.lineWidth = 2.4;
      ctx.beginPath();
      ctx.moveTo(top[0], top[1]);
      ctx.lineTo(bot[0], bot[1]);
      ctx.stroke();
      ctx.fillStyle = "#fbbf24";
      ctx.beginPath();
      ctx.arc(top[0], top[1], 3.2, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // Top F.Cu Traces
  if (state.layers.l1 && board.traces) {
    ctx.lineCap = "round";
    for (const seg of board.traces) {
      if (seg.layer !== 0) continue;
      const a = proj(seg.x1_mm, seg.y1_mm, 0.05);
      const b = proj(seg.x2_mm, seg.y2_mm, 0.05);
      ctx.strokeStyle = "#f43f5e";
      ctx.lineWidth = Math.max(1.6, seg.width_mm * scale * 0.9);
      ctx.beginPath();
      ctx.moveTo(a[0], a[1]);
      ctx.lineTo(b[0], b[1]);
      ctx.stroke();
    }
  }

  // 3D Extruded Component Packages (sorted back-to-front by screen Y)
  const compsSorted = [...board.components].sort((a, b) => {
    const pa = proj(a.x_mm, a.y_mm, 0);
    const pb = proj(b.x_mm, b.y_mm, 0);
    return pa[1] - pb[1];
  });

  for (const comp of compsSorted) {
    const hx = comp.width * 0.36;
    const hy = comp.height * 0.36;
    const pkgH = comp.comp_type === "connector" ? 2.4 : comp.comp_type === "ic" ? 1.4 : 0.9;
    const b0 = proj(comp.x_mm - hx, comp.y_mm - hy, 0);
    const b1 = proj(comp.x_mm + hx, comp.y_mm - hy, 0);
    const b2 = proj(comp.x_mm + hx, comp.y_mm + hy, 0);
    const b3 = proj(comp.x_mm - hx, comp.y_mm + hy, 0);

    const t0 = proj(comp.x_mm - hx, comp.y_mm - hy, pkgH);
    const t1 = proj(comp.x_mm + hx, comp.y_mm - hy, pkgH);
    const t2 = proj(comp.x_mm + hx, comp.y_mm + hy, pkgH);
    const t3 = proj(comp.x_mm - hx, comp.y_mm + hy, pkgH);

    const sideColor = comp.comp_type === "connector" ? "#1e293b" : "#0f172a";
    const topColor = comp.comp_type === "connector" ? "#334155" : comp.comp_type === "ic" ? "#1e293b" : "#475569";

    const faces = [
      [b0, b1, t1, t0],
      [b1, b2, t2, t1],
      [b2, b3, t3, t2],
      [b3, b0, t0, t3],
    ];
    for (const f of faces) {
      ctx.beginPath();
      ctx.moveTo(f[0][0], f[0][1]);
      ctx.lineTo(f[1][0], f[1][1]);
      ctx.lineTo(f[2][0], f[2][1]);
      ctx.lineTo(f[3][0], f[3][1]);
      ctx.closePath();
      ctx.fillStyle = sideColor;
      ctx.fill();
      ctx.strokeStyle = "rgba(148, 163, 184, 0.35)";
      ctx.lineWidth = 0.9;
      ctx.stroke();
    }

    ctx.beginPath();
    ctx.moveTo(t0[0], t0[1]);
    ctx.lineTo(t1[0], t1[1]);
    ctx.lineTo(t2[0], t2[1]);
    ctx.lineTo(t3[0], t3[1]);
    ctx.closePath();
    ctx.fillStyle = topColor;
    ctx.fill();
    ctx.strokeStyle = comp.is_locked ? "#f59e0b" : "#94a3b8";
    ctx.lineWidth = 1.1;
    ctx.stroke();

    const lblPos = proj(comp.x_mm, comp.y_mm, pkgH + 0.15);
    ctx.fillStyle = "#f8fafc";
    ctx.font = '600 10.5px "JetBrains Mono", monospace';
    ctx.textAlign = "center";
    ctx.fillText(comp.comp_id, lblPos[0], lblPos[1]);
  }
}

function renderCanvas() {
  const cvs = els.canvas;
  const rect = els.canvasStage.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const targetW = Math.max(300, Math.floor(rect.width * dpr));
  const targetH = Math.max(300, Math.floor(rect.height * dpr));
  if (cvs.width !== targetW || cvs.height !== targetH) {
    cvs.width = targetW;
    cvs.height = targetH;
  }

  const ctx = cvs.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = "#070b12";
  ctx.fillRect(0, 0, rect.width, rect.height);

  if (state.viewMode === "3d") {
    render3D(ctx, rect.width, rect.height);
  } else {
    render2D(ctx, rect.width, rect.height);
  }
}

async function loadPresets() {
  const resp = await fetch("/api/presets");
  state.presets = await resp.json();
  els.presetSelect.innerHTML = "";
  for (const p of state.presets) {
    const opt = document.createElement("option");
    opt.value = p.id;
    opt.textContent = `${p.id} (${p.components} comps, ${p.nets} nets)`;
    els.presetSelect.appendChild(opt);
  }
  if (state.presets.length > 0) {
    selectPreset(state.presets[0].id);
  }
}

function selectPreset(presetId) {
  const preset = state.presets.find((p) => p.id === presetId);
  if (!preset) return;
  state.board = preset.board;
  state.channels = preset.channels || {};
  state.congestionL1 = null;
  state.congestionL2 = null;
  state.drcPenalty = null;
  state.drcErrors = [];
  els.drcBadge.className = "drc-badge neutral";
  els.drcBadge.textContent = "UNROUTED";
  els.valCritic.textContent = "--";
  els.criticProgress.value = 0;
  els.valDrc.textContent = "0";
  els.valIter.textContent = "0 / 5";
  els.candTbody.innerHTML = '<tr><td colspan="4" class="empty-cell">Run pipeline to sample GNN placements</td></tr>';
  updateTelemetry(state.board);
  renderCanvas();
  renderMiniHeatmaps();
  appendLog(`Loaded preset ${preset.id}: ${preset.description}`);
}

function handleRolloutEvent(evt) {
  if (evt.board) state.board = evt.board;
  if (evt.channels) state.channels = evt.channels;
  if (evt.congestion_L1) state.congestionL1 = evt.congestion_L1;
  if (evt.congestion_L2) state.congestionL2 = evt.congestion_L2;
  if (evt.drc_penalty) state.drcPenalty = evt.drc_penalty;

  switch (evt.event) {
    case "candidate_evaluated": {
      const c = evt.candidate;
      setStatus("running", `Evaluating GNN candidate #${c.candidate_id} (V=${c.world_model_value.toFixed(3)})`);
      updateTelemetry(state.board, { value: c.world_model_value });
      appendLog(`Candidate #${c.candidate_id}: V=${c.world_model_value.toFixed(3)}, HPWL=${c.hpwl_mm.toFixed(1)}mm`);
      break;
    }
    case "placement_selected": {
      const best = evt.candidates[0];
      renderCandidatesTable(evt.candidates, best ? best.candidate_id : null);
      setStatus("running", `Selected Candidate #${best.candidate_id} (score=${evt.best_score.toFixed(3)})`);
      updateTelemetry(state.board, { value: evt.best_score });
      appendLog(`Selected Placement #${best.candidate_id} | Score=${evt.best_score.toFixed(3)}`, "ok");
      break;
    }
    case "net_routed": {
      setStatus("running", `Pass ${evt.iteration}: Routed ${evt.net_name} (${evt.segment_count} segs, ${evt.via_count} vias)`);
      updateTelemetry(state.board, { iteration: evt.iteration, value: evt.world_model_value });
      appendLog(`Pass ${evt.iteration} | ${evt.net_name}: ${evt.segment_count} segs, ${evt.via_count} vias`);
      break;
    }
    case "drc_checked": {
      state.drcErrors = evt.errors || [];
      updateTelemetry(state.board, { iteration: evt.iteration, errorCount: evt.error_count });
      appendLog(
        `DRC Pass ${evt.iteration}: ${evt.error_count} errors, ${evt.unrouted_count} unrouted`,
        evt.error_count === 0 ? "ok" : "warn"
      );
      break;
    }
    case "ripup_triggered": {
      const names = (evt.ripped_nets || []).map((n) => n.name).join(", ");
      setStatus("warn", `Rip-up & reroute triggered for: ${names}`);
      appendLog(`Rip-up & reroute: ${names}`, "warn");
      break;
    }
    case "completed": {
      state.running = false;
      els.btnRun.disabled = false;
      state.drcErrors = evt.errors || [];
      state.kicadText = evt.kicad_pcb || "";
      updateTelemetry(state.board, {
        iteration: evt.iterations_used,
        errorCount: evt.error_count,
        completed: true,
      });
      if (evt.success) {
        setStatus("ok", `Completed in ${evt.elapsed_sec}s · 100% routed · 0 DRC errors`);
        appendLog(`Pipeline PASSED in ${evt.elapsed_sec}s (${evt.iterations_used} pass)`, "ok");
      } else {
        setStatus("warn", `Finished with ${evt.error_count} DRC errors`);
        appendLog(`Pipeline finished with ${evt.error_count} DRC errors`, "err");
      }
      break;
    }
    case "error": {
      state.running = false;
      els.btnRun.disabled = false;
      setStatus("warn", `Error: ${evt.message}`);
      appendLog(`Error: ${evt.message}`, "err");
      break;
    }
  }

  renderCanvas();
  renderMiniHeatmaps();
}

function runRolloutWebSocket() {
  if (state.running) return;
  state.running = true;
  els.btnRun.disabled = true;
  els.drcBadge.className = "drc-badge running";
  els.drcBadge.textContent = "RUNNING";
  setStatus("running", "Connecting to World Model WebSocket...");

  const boardId = els.presetSelect.value || "mcu_sensor_node";
  const candidates = Number(els.candidatesRange.value) || 8;
  const paceMs = Number(els.paceRange.value) || 0;
  els.candCountTag.textContent = `K = ${candidates}`;

  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${proto}//${window.location.host}/ws/rollout`;

  if (state.ws && state.ws.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify({ board: boardId, candidates, max_iterations: 5, step_delay_ms: paceMs }));
    return;
  }

  const ws = new WebSocket(wsUrl);
  state.ws = ws;
  ws.addEventListener("open", () => {
    ws.send(JSON.stringify({ board: boardId, candidates, max_iterations: 5, step_delay_ms: paceMs }));
  });
  ws.addEventListener("message", (event) => {
    const data = JSON.parse(event.data);
    handleRolloutEvent(data);
  });
  ws.addEventListener("close", () => {
    state.ws = null;
    if (state.running) {
      state.running = false;
      els.btnRun.disabled = false;
      setStatus("warn", "WebSocket closed unexpectedly");
    }
  });
  ws.addEventListener("error", () => {
    appendLog("WebSocket error — falling back to HTTP /api/run", "warn");
    runFallbackHttp(boardId, candidates);
  });
}

async function runFallbackHttp(boardId, candidates) {
  try {
    const resp = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ board: boardId, candidates, max_iterations: 5 }),
    });
    const data = await resp.json();
    renderCandidatesTable(data.candidates, data.candidates[0]?.candidate_id ?? null);
    handleRolloutEvent({
      event: "completed",
      ...data,
    });
  } catch (err) {
    state.running = false;
    els.btnRun.disabled = false;
    appendLog(`HTTP fallback failed: ${err}`, "err");
  }
}

function wireEvents() {
  els.presetSelect.addEventListener("change", (e) => selectPreset(e.target.value));
  els.candidatesRange.addEventListener("input", (e) => {
    els.candidatesVal.textContent = e.target.value;
    els.candCountTag.textContent = `K = ${e.target.value}`;
  });
  els.paceRange.addEventListener("input", (e) => {
    els.paceVal.textContent = `${e.target.value}ms`;
  });

  els.btnView2d.addEventListener("click", () => {
    state.viewMode = "2d";
    els.btnView2d.classList.add("active");
    els.btnView3d.classList.remove("active");
    els.hud3dControls.hidden = true;
    renderCanvas();
  });

  els.btnView3d.addEventListener("click", () => {
    state.viewMode = "3d";
    els.btnView3d.classList.add("active");
    els.btnView2d.classList.remove("active");
    els.hud3dControls.hidden = false;
    renderCanvas();
  });

  els.tiltRange.addEventListener("input", (e) => {
    state.tiltDeg = Number(e.target.value);
    renderCanvas();
  });

  els.yawRange.addEventListener("input", (e) => {
    state.yawDeg = Number(e.target.value);
    renderCanvas();
  });

  els.btnRun.addEventListener("click", runRolloutWebSocket);
  els.btnClearLog.addEventListener("click", () => { els.eventLog.innerHTML = ""; });
  els.btnResetView.addEventListener("click", () => {
    state.zoom = 1.0;
    state.panX = 0;
    state.panY = 0;
    renderCanvas();
  });

  const layerMap = {
    "chk-l1": "l1",
    "chk-l2": "l2",
    "chk-vias": "vias",
    "chk-courtyard": "courtyard",
    "chk-ratsnest": "ratsnest",
    "chk-congestion": "congestion",
    "chk-penalty": "penalty",
  };
  for (const [id, key] of Object.entries(layerMap)) {
    document.getElementById(id).addEventListener("change", (e) => {
      state.layers[key] = e.target.checked;
      renderCanvas();
    });
  }

  // Pan & Zoom on PCB Canvas
  els.canvas.addEventListener("pointerdown", (e) => {
    state.isDragging = true;
    state.lastPointer = { x: e.clientX, y: e.clientY };
    els.canvas.setPointerCapture(e.pointerId);
  });
  els.canvas.addEventListener("pointermove", (e) => {
    const rect = els.canvas.getBoundingClientRect();
    if (state._viewport && state.viewMode === "2d") {
      const mmX = ((e.clientX - rect.left) - state._viewport.originX) / state._viewport.scale;
      const mmY = ((e.clientY - rect.top) - state._viewport.originY) / state._viewport.scale;
      els.cursorCoords.textContent = `X: ${mmX.toFixed(1)}mm  Y: ${mmY.toFixed(1)}mm`;
    }
    if (!state.isDragging) return;
    const dx = e.clientX - state.lastPointer.x;
    const dy = e.clientY - state.lastPointer.y;
    state.lastPointer = { x: e.clientX, y: e.clientY };
    if (state.viewMode === "3d" && e.shiftKey) {
      state.yawDeg = Math.max(-75, Math.min(75, state.yawDeg + dx * 0.4));
      state.tiltDeg = Math.max(15, Math.min(75, state.tiltDeg + dy * 0.3));
      els.yawRange.value = String(Math.round(state.yawDeg));
      els.tiltRange.value = String(Math.round(state.tiltDeg));
    } else {
      state.panX += dx;
      state.panY += dy;
    }
    renderCanvas();
  });
  els.canvas.addEventListener("pointerup", () => { state.isDragging = false; });
  els.canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.1 : 0.9;
    state.zoom = Math.max(0.45, Math.min(4.5, state.zoom * factor));
    renderCanvas();
  }, { passive: false });

  // Native <dialog> Inspector
  els.btnInspectKicad.addEventListener("click", async () => {
    if (!state.kicadText) {
      const boardId = els.presetSelect.value || "mcu_sensor_node";
      const resp = await fetch(`/api/export-kicad?board=${encodeURIComponent(boardId)}`);
      state.kicadText = await resp.text();
    }
    els.kicadCode.textContent = state.kicadText;
    els.btnDownloadKicad.href = `/api/export-kicad?board=${encodeURIComponent(els.presetSelect.value)}&download=1`;
    els.kicadDialog.showModal();
  });
  els.btnCloseDialog.addEventListener("click", () => els.kicadDialog.close());
  els.btnCopyKicad.addEventListener("click", async () => {
    await navigator.clipboard.writeText(els.kicadCode.textContent || "");
    els.btnCopyKicad.textContent = "Copied!";
    setTimeout(() => { els.btnCopyKicad.textContent = "Copy"; }, 1400);
  });

  window.addEventListener("resize", renderCanvas);
}

initTensorStrip();
wireEvents();
loadPresets();
