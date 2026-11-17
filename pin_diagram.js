/**
 * Interactive Vector SVG Pinout Diagram & Component Inspector (rev9).
 * Renders Altium/datasheet-style physical package pinout SVGs (QFN 4-sided, SOIC/SOT-23
 * dual-inline, USB-C, 2-pin passives/headers) with color-coded electrical roles
 * and click-to-cross-probe net highlighting.
 */

const SVG_NS = 'http://www.w3.org/2000/svg';

export const PIN_ROLE_COLORS = {
  power: { fill: '#dc2626', stroke: '#fca5a5', label: 'Power', text: '#ffffff' },
  gnd: { fill: '#1e293b', stroke: '#64748b', label: 'GND', text: '#e2e8f0' },
  bus: { fill: '#0d9488', stroke: '#5eead4', label: 'I2C/SPI/UART', text: '#ffffff' },
  usb: { fill: '#2563eb', stroke: '#93c5fd', label: 'USB Diff', text: '#ffffff' },
  gpio: { fill: '#d97706', stroke: '#fcd34d', label: 'GPIO/Clk/Wake', text: '#ffffff' },
  nc: { fill: '#334155', stroke: '#475569', label: 'NC / Passive', text: '#cbd5e1' },
};

export function classifyPinRole(netName = '') {
  const n = String(netName || '').trim().toUpperCase();
  if (!n) return 'nc';
  if (n === 'GND' || n.includes('GND') || n.includes('VSS') || n === 'EP') return 'gnd';
  if (/(VCC|VDD|VBUS|VBAT|VIN|VOUT|3V3|5V|12V|PWR|LDO|AVDD)/.test(n)) return 'power';
  if (/(USB|D\+|D-|DP|DM|CC1|CC2|VCONN)/.test(n)) return 'usb';
  if (/(SDA|SCL|MOSI|MISO|SCK|SCLK|SPI|I2C|TX|RX|UART|CS|SS|WS2812|QUAD|ENC)/.test(n)) return 'bus';
  return 'gpio';
}

function svgEl(tag, attrs = {}, text = null) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    el.setAttribute(k, String(v));
  }
  if (text !== null) el.textContent = String(text);
  return el;
}

/**
 * Render an interactive Vector SVG Pin Diagram into `container`.
 * @param {HTMLElement} container
 * @param {object} comp Component summary object
 * @param {object} opts { selectedNetId, onSelectPin }
 */
export function renderPinDiagramSvg(container, comp, opts = {}) {
  container.innerHTML = '';
  if (!comp) {
    const p = document.createElement('p');
    p.className = 'pin-empty-msg';
    p.textContent = 'Click any IC, connector, or passive in 2D, 3D PBR, Schematic, or Explorer to inspect its vector pinout.';
    container.appendChild(p);
    return;
  }

  const pads = (comp.pads || []).slice().sort((a, b) => {
    const na = parseInt(a.pad_id, 10);
    const nb = parseInt(b.pad_id, 10);
    if (!Number.isNaN(na) && !Number.isNaN(nb)) return na - nb;
    return String(a.pad_id).localeCompare(String(b.pad_id));
  });

  const nonEpPads = pads.filter((p) => String(p.pad_id).toUpperCase() !== 'EP');
  const epPad = pads.find((p) => String(p.pad_id).toUpperCase() === 'EP');
  const fpUpper = String(comp.footprint || '').toUpperCase();
  const isFourSided =
    nonEpPads.length >= 12 ||
    fpUpper.includes('QFN') ||
    fpUpper.includes('LGA') ||
    fpUpper.includes('LQFP');

  const W = 380;
  const H = isFourSided ? 310 : Math.max(190, 96 + Math.ceil(nonEpPads.length / 2) * 26);
  const svg = svgEl('svg', {
    viewBox: `0 0 ${W} ${H}`,
    class: 'pin-diagram-svg',
    role: 'img',
    'aria-label': `Pinout diagram for ${comp.comp_id} (${comp.value || comp.footprint || ''})`,
  });

  // Subtle schematic background frame
  svg.appendChild(svgEl('rect', { x: 2, y: 2, width: W - 4, height: H - 4, rx: 10, fill: '#0b1220', stroke: '#263248', 'stroke-width': 1.2 }));

  const cx = W * 0.5;
  const cy = H * 0.5;
  const bodyW = isFourSided ? 128 : 132;
  const bodyH = isFourSided ? 128 : Math.max(76, H - 72);

  // IC Package Body
  svg.appendChild(svgEl('rect', {
    x: cx - bodyW / 2,
    y: cy - bodyH / 2,
    width: bodyW,
    height: bodyH,
    rx: 6,
    fill: '#161b26',
    stroke: '#475569',
    'stroke-width': 1.8,
  }));

  // Pin-1 Chamfer Dot
  svg.appendChild(svgEl('circle', {
    cx: cx - bodyW / 2 + 12,
    cy: cy - bodyH / 2 + 12,
    r: 4.2,
    fill: '#f59e0b',
  }));

  // Exposed Thermal Pad (EP) if present
  if (epPad) {
    const epActive = opts.selectedNetId !== null && opts.selectedNetId === epPad.net_id;
    const epG = svgEl('g', { class: 'pin-interactive', tabindex: 0, role: 'button', 'aria-label': `EP ${epPad.net_name || 'GND'}` });
    epG.appendChild(svgEl('rect', {
      x: cx - 22,
      y: cy - 22,
      width: 44,
      height: 44,
      rx: 4,
      fill: epActive ? '#0ea5e9' : '#334155',
      stroke: '#e2b857',
      'stroke-width': 1.5,
      'stroke-dasharray': '3 2',
    }));
    epG.appendChild(svgEl('text', {
      x: cx,
      y: cy + 3,
      'text-anchor': 'middle',
      fill: '#f8fafc',
      'font-size': 9,
      'font-family': 'JetBrains Mono, monospace',
      'font-weight': 700,
    }, `EP:${epPad.net_name || 'GND'}`));
    epG.addEventListener('click', () => opts.onSelectPin?.(epPad.net_id, epPad.net_name, comp.comp_id, epPad.pad_id));
    svg.appendChild(epG);
  } else {
    svg.appendChild(svgEl('text', {
      x: cx,
      y: cy - 6,
      'text-anchor': 'middle',
      fill: '#f8fafc',
      'font-size': 12,
      'font-family': 'JetBrains Mono, monospace',
      'font-weight': 700,
    }, comp.comp_id));
    svg.appendChild(svgEl('text', {
      x: cx,
      y: cy + 10,
      'text-anchor': 'middle',
      fill: '#94a3b8',
      'font-size': 9.5,
      'font-family': 'JetBrains Mono, monospace',
    }, String(comp.value || comp.comp_type || '').slice(0, 16)));
  }

  const drawPin = (pad, px, py, side) => {
    const role = classifyPinRole(pad.net_name);
    const palette = PIN_ROLE_COLORS[role] || PIN_ROLE_COLORS.nc;
    const isSelected = opts.selectedNetId !== null && opts.selectedNetId === pad.net_id && pad.net_id > 0;
    const g = svgEl('g', {
      class: `pin-interactive${isSelected ? ' pin-selected' : ''}`,
      tabindex: 0,
      role: 'button',
      'aria-label': `Pin ${pad.pad_id} ${pad.net_name || 'NC'}`,
    });

    const netText = String(pad.net_name || 'NC').slice(0, 11);
    const pinNum = String(pad.pad_id).slice(0, 4);

    if (side === 'left' || side === 'right') {
      const dir = side === 'left' ? -1 : 1;
      const stubX = side === 'left' ? px - 14 : px;
      g.appendChild(svgEl('rect', {
        x: stubX,
        y: py - 5,
        width: 14,
        height: 10,
        rx: 2,
        fill: isSelected ? '#38bdf8' : '#e2b857',
        stroke: isSelected ? '#ffffff' : 'none',
        'stroke-width': 1.2,
      }));
      // Pin number inside body edge
      g.appendChild(svgEl('text', {
        x: px - dir * 6,
        y: py + 3,
        'text-anchor': side === 'left' ? 'start' : 'end',
        fill: '#cbd5e1',
        'font-size': 8.5,
        'font-family': 'JetBrains Mono, monospace',
      }, pinNum));

      // Role color badge + net name outside pin
      const badgeW = 84;
      const badgeX = side === 'left' ? px - 18 - badgeW : px + 18;
      g.appendChild(svgEl('rect', {
        x: badgeX,
        y: py - 8,
        width: badgeW,
        height: 16,
        rx: 4,
        fill: palette.fill,
        stroke: isSelected ? '#38bdf8' : palette.stroke,
        'stroke-width': isSelected ? 2.0 : 0.9,
      }));
      g.appendChild(svgEl('text', {
        x: badgeX + badgeW / 2,
        y: py + 3,
        'text-anchor': 'middle',
        fill: palette.text,
        'font-size': 8.5,
        'font-family': 'JetBrains Mono, monospace',
        'font-weight': 600,
      }, netText));
    } else {
      const dir = side === 'top' ? -1 : 1;
      const stubY = side === 'top' ? py - 12 : py;
      g.appendChild(svgEl('rect', {
        x: px - 5,
        y: stubY,
        width: 10,
        height: 12,
        rx: 2,
        fill: isSelected ? '#38bdf8' : '#e2b857',
      }));
      g.appendChild(svgEl('text', {
        x: px,
        y: py - dir * 8,
        'text-anchor': 'middle',
        fill: '#cbd5e1',
        'font-size': 8,
        'font-family': 'JetBrains Mono, monospace',
      }, pinNum));

      const labelY = side === 'top' ? py - 16 : py + 17;
      g.appendChild(svgEl('text', {
        x: px,
        y: labelY,
        'text-anchor': side === 'top' ? 'start' : 'end',
        transform: `rotate(-52 ${px} ${labelY})`,
        fill: isSelected ? '#38bdf8' : palette.stroke,
        'font-size': 8.2,
        'font-family': 'JetBrains Mono, monospace',
        'font-weight': 600,
      }, netText));
    }

    const activate = () => opts.onSelectPin?.(pad.net_id, pad.net_name, comp.comp_id, pad.pad_id);
    g.addEventListener('click', activate);
    g.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        activate();
      }
    });
    svg.appendChild(g);
  };

  if (isFourSided) {
    const perSide = Math.ceil(nonEpPads.length / 4);
    const maxDisplay = Math.min(nonEpPads.length, 48);
    const slice = nonEpPads.slice(0, maxDisplay);
    const sides = [
      slice.slice(0, perSide),
      slice.slice(perSide, perSide * 2),
      slice.slice(perSide * 2, perSide * 3),
      slice.slice(perSide * 3),
    ];
    // Left (top-to-bottom), Bottom (left-to-right), Right (bottom-to-top), Top (right-to-left)
    sides[0].forEach((p, idx) => {
      const y = cy - bodyH * 0.40 + (idx + 0.5) * ((bodyH * 0.80) / Math.max(1, sides[0].length));
      drawPin(p, cx - bodyW / 2, y, 'left');
    });
    sides[1].forEach((p, idx) => {
      const x = cx - bodyW * 0.40 + (idx + 0.5) * ((bodyW * 0.80) / Math.max(1, sides[1].length));
      drawPin(p, x, cy + bodyH / 2, 'bottom');
    });
    sides[2].forEach((p, idx) => {
      const y = cy + bodyH * 0.40 - (idx + 0.5) * ((bodyH * 0.80) / Math.max(1, sides[2].length));
      drawPin(p, cx + bodyW / 2, y, 'right');
    });
    sides[3].forEach((p, idx) => {
      const x = cx + bodyW * 0.40 - (idx + 0.5) * ((bodyW * 0.80) / Math.max(1, sides[3].length));
      drawPin(p, x, cy - bodyH / 2, 'top');
    });
  } else {
    const half = Math.ceil(nonEpPads.length / 2);
    const leftPads = nonEpPads.slice(0, half);
    const rightPads = nonEpPads.slice(half).reverse();
    leftPads.forEach((p, idx) => {
      const y = cy - bodyH * 0.40 + (idx + 0.5) * ((bodyH * 0.80) / Math.max(1, leftPads.length));
      drawPin(p, cx - bodyW / 2, y, 'left');
    });
    rightPads.forEach((p, idx) => {
      const y = cy - bodyH * 0.40 + (idx + 0.5) * ((bodyH * 0.80) / Math.max(1, rightPads.length));
      drawPin(p, cx + bodyW / 2, y, 'right');
    });
  }

  container.appendChild(svg);
}
