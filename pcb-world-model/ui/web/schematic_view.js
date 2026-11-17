/**
 * Auto-Generated Functional-Block Schematic View & Cross-Probing (rev9).
 * Renders an orthogonal SVG circuit diagram grouped by functional blocks
 * (Power & Regulation, MCU/FPGA Core, Sensors/Display/Peripherals, Passives)
 * with live cross-probing synced to 2D CAD, 3D PBR, and Pin Inspector.
 */

import { classifyPinRole, PIN_ROLE_COLORS } from './pin_diagram.js';

const SVG_NS = 'http://www.w3.org/2000/svg';

function svgEl(tag, attrs = {}, text = null) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    el.setAttribute(k, String(v));
  }
  if (text !== null) el.textContent = String(text);
  return el;
}

function classifyFunctionalBlock(comp) {
  const id = String(comp.comp_id || '').toUpperCase();
  const val = String(comp.value || '').toUpperCase();
  const ct = String(comp.comp_type || '').toLowerCase();
  const fp = String(comp.footprint || '').toUpperCase();

  if (/(USB|JST|BAT|PWR|LDO|BUCK|MP2307|AP2112|MCP73831|AMS1117|LM|TPS|INDUCTOR|SCHOTTKY)/.test(`${id} ${val} ${fp}`) || ct === 'connector' || ct === 'inductor') {
    return 'power';
  }
  if (/(ESP32|RP2040|STM32|ATTINY|CH552|CH340|ICE40|FPGA|MCU|XTAL|CRYSTAL)/.test(`${id} ${val} ${fp}`) || ct === 'crystal') {
    return 'core';
  }
  if (/(BME280|LIS3DH|OLED|SSD1306|IMU|MPU|WS2812|LED|SW|KEY|RELAY|OPTO)/.test(`${id} ${val} ${fp}`) || ct === 'led' || ct === 'switch' || ct === 'display') {
    return 'peripherals';
  }
  if (ct === 'ic') return 'core';
  return 'passives';
}

const BLOCK_COLUMNS = [
  { key: 'power', title: '1. Power Input & Regulation', accent: '#f59e0b' },
  { key: 'core', title: '2. MCU / FPGA Core & Clock', accent: '#38bdf8' },
  { key: 'peripherals', title: '3. Sensors, Display & I/O', accent: '#10b981' },
  { key: 'passives', title: '4. Decoupling & Support Passives', accent: '#a855f7' },
];

/**
 * Render an interactive orthogonal schematic SVG inside `container`.
 */
export function renderSchematicSvg(container, board, opts = {}) {
  container.innerHTML = '';
  if (!board || !(board.components || []).length) {
    const p = document.createElement('p');
    p.className = 'pin-empty-msg';
    p.textContent = 'Select or synthesize a board to inspect its auto-generated hierarchical schematic.';
    container.appendChild(p);
    return;
  }

  const comps = (board.components || []).filter((c) => c.comp_type !== 'mounting_hole');
  const groups = { power: [], core: [], peripherals: [], passives: [] };
  for (const c of comps) {
    groups[classifyFunctionalBlock(c)].push(c);
  }

  const colW = 248;
  const colGap = 44;
  const padX = 28;
  const topY = 58;
  const symW = 198;
  const rowGap = 24;

  // Compute symbol box height & position for each component
  const layouts = new Map();
  const pinCoords = new Map(); // `${compId}:${padId}` -> { x, y, netId, netName }
  let maxColHeight = 420;

  BLOCK_COLUMNS.forEach((col, colIdx) => {
    const colX = padX + colIdx * (colW + colGap);
    let cursorY = topY + 18;
    for (const comp of groups[col.key]) {
      const pads = (comp.pads || []).slice(0, 18);
      const half = Math.max(1, Math.ceil(pads.length / 2));
      const boxH = Math.max(54, 28 + half * 16);
      const boxX = colX + (colW - symW) * 0.5;
      const boxY = cursorY;

      layouts.set(comp.comp_id, { x: boxX, y: boxY, w: symW, h: boxH, colIdx });

      pads.forEach((pad, idx) => {
        const isLeft = idx < half;
        const row = isLeft ? idx : idx - half;
        const py = boxY + 26 + row * 16;
        const px = isLeft ? boxX - 10 : boxX + symW + 10;
        pinCoords.set(`${comp.comp_id}:${pad.pad_id}`, {
          x: px,
          y: py,
          isLeft,
          pad,
          compId: comp.comp_id,
        });
      });

      cursorY += boxH + rowGap;
    }
    maxColHeight = Math.max(maxColHeight, cursorY + 28);
  });

  const totalW = padX * 2 + BLOCK_COLUMNS.length * colW + (BLOCK_COLUMNS.length - 1) * colGap;
  const totalH = maxColHeight;

  const svg = svgEl('svg', {
    viewBox: `0 0 ${totalW} ${totalH}`,
    class: 'schematic-svg',
    role: 'img',
    'aria-label': `Hierarchical schematic for ${board.board_name || 'PCB'}`,
  });

  // Blueprint dot grid pattern
  const defs = svgEl('defs');
  const pat = svgEl('pattern', { id: 'sch-grid', width: 20, height: 20, patternUnits: 'userSpaceOnUse' });
  pat.appendChild(svgEl('circle', { cx: 1, cy: 1, r: 0.9, fill: '#1e293b' }));
  defs.appendChild(pat);
  svg.appendChild(defs);
  svg.appendChild(svgEl('rect', { x: 0, y: 0, width: totalW, height: totalH, fill: '#070b12' }));
  svg.appendChild(svgEl('rect', { x: 0, y: 0, width: totalW, height: totalH, fill: 'url(#sch-grid)' }));

  // Functional Group Column Containers
  BLOCK_COLUMNS.forEach((col, idx) => {
    const x = padX + idx * (colW + colGap);
    svg.appendChild(svgEl('rect', {
      x,
      y: 22,
      width: colW,
      height: totalH - 42,
      rx: 10,
      fill: 'rgba(17, 24, 39, 0.55)',
      stroke: '#263248',
      'stroke-width': 1.2,
      'stroke-dasharray': '5 4',
    }));
    svg.appendChild(svgEl('rect', {
      x,
      y: 22,
      width: colW,
      height: 28,
      rx: 10,
      fill: 'rgba(30, 41, 59, 0.85)',
    }));
    svg.appendChild(svgEl('text', {
      x: x + 12,
      y: 40,
      fill: col.accent,
      'font-size': 11,
      'font-family': 'IBM Plex Sans, sans-serif',
      'font-weight': 700,
    }, col.title));
  });

  // Orthogonal Signal Wires (for non-global-power nets; power/GND show crisp rail flags)
  const netsList = board.nets || [];
  const hasFilter = opts.selectedCompId !== null || opts.selectedNetId !== null;
  const wireLayer = svgEl('g');

  netsList.forEach((net, netIdx) => {
    const netName = String(net.name || '');
    const role = classifyPinRole(netName);
    const isGlobalRail = role === 'power' || role === 'gnd';
    const isNetSelected = opts.selectedNetId !== null && opts.selectedNetId === net.net_id;
    if (isGlobalRail && !isNetSelected) return; // Global rails drawn as clean pin rail flags unless soloed

    const pts = (net.pins || [])
      .map((p) => pinCoords.get(`${p.comp_id}:${p.pad_id}`))
      .filter(Boolean);
    if (pts.length < 2) return;

    const stroke = isNetSelected
      ? '#38bdf8'
      : role === 'usb'
      ? '#3b82f6'
      : role === 'bus'
      ? '#14b8a6'
      : '#64748b';
    const opacity = hasFilter ? (isNetSelected ? 1.0 : 0.16) : 0.78;
    const laneOffset = ((netIdx % 7) - 3) * 5;

    for (let i = 0; i < pts.length - 1; i++) {
      const a = pts[i];
      const b = pts[i + 1];
      const midX = (a.x + b.x) * 0.5 + laneOffset;
      const d = `M ${a.x} ${a.y} L ${midX} ${a.y} L ${midX} ${b.y} L ${b.x} ${b.y}`;
      const path = svgEl('path', {
        d,
        fill: 'none',
        stroke,
        'stroke-width': isNetSelected ? 2.6 : 1.5,
        'stroke-opacity': opacity,
        class: 'sch-wire',
      });
      path.addEventListener('click', (e) => {
        e.stopPropagation();
        opts.onSelectNet?.(net.net_id, net.name);
      });
      wireLayer.appendChild(path);
    }
  });
  svg.appendChild(wireLayer);

  // Component Symbol Blocks + Pins + Power/GND Flags
  for (const comp of comps) {
    const box = layouts.get(comp.comp_id);
    if (!box) continue;
    const isCompSelected = opts.selectedCompId === comp.comp_id;
    const compDimmed = opts.selectedCompId && !isCompSelected;

    const compG = svgEl('g', {
      class: 'sch-comp',
      opacity: compDimmed ? 0.35 : 1.0,
      tabindex: 0,
      role: 'button',
      'aria-label': `Schematic symbol ${comp.comp_id} ${comp.value || ''}`,
    });

    compG.appendChild(svgEl('rect', {
      x: box.x,
      y: box.y,
      width: box.w,
      height: box.h,
      rx: 6,
      fill: '#111827',
      stroke: isCompSelected ? '#38bdf8' : '#475569',
      'stroke-width': isCompSelected ? 2.4 : 1.3,
    }));

    compG.appendChild(svgEl('text', {
      x: box.x + 8,
      y: box.y + 14,
      fill: '#f8fafc',
      'font-size': 10.5,
      'font-family': 'JetBrains Mono, monospace',
      'font-weight': 700,
    }, comp.comp_id));

    compG.appendChild(svgEl('text', {
      x: box.x + box.w - 8,
      y: box.y + 14,
      'text-anchor': 'end',
      fill: '#94a3b8',
      'font-size': 9.2,
      'font-family': 'JetBrains Mono, monospace',
    }, String(comp.value || comp.comp_type || '').slice(0, 14)));

    const activateComp = (e) => {
      e.stopPropagation();
      opts.onSelectComponent?.(comp.comp_id);
    };
    compG.addEventListener('click', activateComp);
    compG.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        activateComp(e);
      }
    });

    const pads = (comp.pads || []).slice(0, 18);
    for (const pad of pads) {
      const pc = pinCoords.get(`${comp.comp_id}:${pad.pad_id}`);
      if (!pc) continue;
      const role = classifyPinRole(pad.net_name);
      const pal = PIN_ROLE_COLORS[role] || PIN_ROLE_COLORS.nc;
      const pinActive = opts.selectedNetId !== null && opts.selectedNetId === pad.net_id && pad.net_id > 0;
      const edgeX = pc.isLeft ? box.x : box.x + box.w;

      const pinG = svgEl('g', { class: 'pin-interactive' });
      pinG.appendChild(svgEl('line', {
        x1: edgeX,
        y1: pc.y,
        x2: pc.x,
        y2: pc.y,
        stroke: pinActive ? '#38bdf8' : pal.stroke,
        'stroke-width': pinActive ? 2.4 : 1.4,
      }));
      pinG.appendChild(svgEl('circle', {
        cx: pc.x,
        cy: pc.y,
        r: pinActive ? 3.5 : 2.5,
        fill: pinActive ? '#38bdf8' : pal.fill,
        stroke: pal.stroke,
        'stroke-width': 1,
      }));

      const pinTxt = `${pad.pad_id}:${String(pad.net_name || 'NC').slice(0, 9)}`;
      pinG.appendChild(svgEl('text', {
        x: pc.isLeft ? edgeX + 5 : edgeX - 5,
        y: pc.y + 3,
        'text-anchor': pc.isLeft ? 'start' : 'end',
        fill: pinActive ? '#38bdf8' : '#cbd5e1',
        'font-size': 8.2,
        'font-family': 'JetBrains Mono, monospace',
      }, pinTxt));

      pinG.addEventListener('click', (e) => {
        e.stopPropagation();
        opts.onSelectPin?.(pad.net_id, pad.net_name, comp.comp_id, pad.pad_id);
      });
      compG.appendChild(pinG);
    }

    svg.appendChild(compG);
  }

  svg.addEventListener('click', () => opts.onClearSelection?.());
  container.appendChild(svg);
}
