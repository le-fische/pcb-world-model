/**
 * Procedural "STEP-Quality" 3D Component Library for Three.js PBR Viewport (rev9).
 * Synthesizes chamfered epoxy ICs, gull-wing leads, brushed-nickel RF shields + gold
 * meandering PCB antennas, EIA-labeled resistors, ceramic MLCCs, translucent LEDs,
 * USB-C receptacles, JST-SH connectors, PinHeaders, switches, and MX keyswitches.
 */

import * as THREE from 'three';

const _labelTexCache = new Map();

function createPartLabelTexture(text, bg = '#151922', fg = '#cbd5e1', subText = '') {
  const key = `${text}|${bg}|${fg}|${subText}`;
  if (_labelTexCache.has(key)) return _labelTexCache.get(key);

  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 128;
  const ctx = canvas.getContext('2d');
  if (ctx) {
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, 256, 128);

    // Subtle laser-etched Pin-1 orientation dot in top-left
    ctx.fillStyle = '#94a3b8';
    ctx.beginPath();
    ctx.arc(24, 24, 9, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = fg;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    const fontSize = text.length > 10 ? 22 : text.length > 6 ? 28 : 36;
    ctx.font = `700 ${fontSize}px "JetBrains Mono", monospace`;
    ctx.fillText(text.slice(0, 14), 128, subText ? 52 : 64);

    if (subText) {
      ctx.fillStyle = 'rgba(203, 213, 225, 0.72)';
      ctx.font = '500 18px "JetBrains Mono", monospace';
      ctx.fillText(subText.slice(0, 16), 128, 92);
    }
  }
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 4;
  _labelTexCache.set(key, tex);
  return tex;
}

function parseResistorEiaCode(valueStr = '') {
  const s = String(valueStr || '').trim().toUpperCase();
  const match = s.match(/^(\d+(?:\.\d+)?)\s*([KMR]?)(?:OHM)?/);
  if (!match) return '103';
  const num = parseFloat(match[1]);
  const unit = match[2] || '';
  let ohms = num;
  if (unit === 'K') ohms = num * 1000;
  else if (unit === 'M') ohms = num * 1e6;
  if (!Number.isFinite(ohms) || ohms <= 0) return '103';
  if (ohms < 10) return `${ohms.toFixed(1).replace('.', 'R')}`;
  const exp = Math.floor(Math.log10(ohms)) - 1;
  const sig = Math.round(ohms / Math.pow(10, Math.max(0, exp)));
  return `${sig}${Math.max(0, exp)}`.slice(0, 4);
}

function createChamferedBoxGeometry(w, d, h, bevel = 0.08) {
  const b = Math.min(bevel, w * 0.16, d * 0.16, h * 0.28);
  const hw = Math.max(0.05, w * 0.5 - b);
  const hd = Math.max(0.05, d * 0.5 - b);
  const shape = new THREE.Shape();
  shape.moveTo(-hw, -hd);
  shape.lineTo(hw, -hd);
  shape.lineTo(hw, hd);
  shape.lineTo(-hw, hd);
  shape.closePath();

  const extrudeH = Math.max(0.04, h - b * 2);
  const geo = new THREE.ExtrudeGeometry(shape, {
    depth: extrudeH,
    bevelEnabled: true,
    bevelSegments: 2,
    steps: 1,
    bevelSize: b,
    bevelThickness: b,
  });
  // ExtrudeGeometry extrudes along +Z; keep Z as height and translate so bottom sits at z=0
  geo.translate(0, 0, b);
  return geo;
}

const SHARED_MATS = {
  epoxyDark: new THREE.MeshPhysicalMaterial({
    color: 0x161920,
    roughness: 0.44,
    metalness: 0.05,
    clearcoat: 0.12,
    clearcoatRoughness: 0.4,
  }),
  nickelShield: new THREE.MeshStandardMaterial({
    color: 0xc2c8d0,
    metalness: 0.88,
    roughness: 0.24,
  }),
  stainlessSteel: new THREE.MeshStandardMaterial({
    color: 0xb7bec7,
    metalness: 0.90,
    roughness: 0.22,
  }),
  tinnedLead: new THREE.MeshStandardMaterial({
    color: 0xd8dde4,
    metalness: 0.86,
    roughness: 0.22,
  }),
  enigGold: new THREE.MeshStandardMaterial({
    color: 0xe2b857,
    metalness: 0.92,
    roughness: 0.18,
  }),
  ceramicCap: new THREE.MeshStandardMaterial({
    color: 0xc49563,
    roughness: 0.38,
    metalness: 0.06,
  }),
  ceramicWhite: new THREE.MeshStandardMaterial({
    color: 0xe6e1d3,
    roughness: 0.42,
    metalness: 0.04,
  }),
  jstCream: new THREE.MeshStandardMaterial({
    color: 0xebe5d5,
    roughness: 0.52,
    metalness: 0.02,
  }),
  ferriteDark: new THREE.MeshStandardMaterial({
    color: 0x2a2d34,
    roughness: 0.55,
    metalness: 0.14,
  }),
  blackPlastic: new THREE.MeshStandardMaterial({
    color: 0x1e222a,
    roughness: 0.50,
    metalness: 0.04,
  }),
  mxTranslucent: new THREE.MeshPhysicalMaterial({
    color: 0x94a3b8,
    roughness: 0.22,
    metalness: 0.05,
    transmission: 0.45,
    transparent: true,
    opacity: 0.78,
  }),
  mxStemRed: new THREE.MeshStandardMaterial({
    color: 0xe11d48,
    roughness: 0.35,
    metalness: 0.05,
  }),
  oledGlass: new THREE.MeshPhysicalMaterial({
    color: 0x090d16,
    roughness: 0.10,
    metalness: 0.15,
    clearcoat: 0.85,
    clearcoatRoughness: 0.08,
  }),
};

function addTopSilkLabel(group, w, d, zTop, text, bg = '#161920', fg = '#cbd5e1', subText = '') {
  const tex = createPartLabelTexture(text, bg, fg, subText);
  const mat = new THREE.MeshStandardMaterial({
    map: tex,
    roughness: 0.45,
    metalness: 0.06,
  });
  const plane = new THREE.Mesh(
    new THREE.PlaneGeometry(Math.max(0.3, w * 0.82), Math.max(0.25, d * 0.65)),
    mat
  );
  plane.position.set(0, 0, zTop + 0.012);
  group.add(plane);
}

function buildQfnOrLgaMesh(comp, w, d, h) {
  const group = new THREE.Group();
  const bodyW = w * 0.88;
  const bodyD = d * 0.88;
  const body = new THREE.Mesh(createChamferedBoxGeometry(bodyW, bodyD, h, 0.08), SHARED_MATS.epoxyDark);
  body.castShadow = true;
  body.receiveShadow = true;
  group.add(body);

  // Perimeter metallic QFN lands along edges
  const padCount = (comp.pads || []).length;
  for (const pad of comp.pads || []) {
    if (String(pad.pad_id).toUpperCase() === 'EP') continue;
    const lx = (pad.x_mm - comp.x_mm);
    const ly = -(pad.y_mm - comp.y_mm);
    const land = new THREE.Mesh(
      new THREE.BoxGeometry(Math.max(0.18, pad.width * 0.88), Math.max(0.18, pad.height * 0.88), 0.12),
      SHARED_MATS.tinnedLead
    );
    land.position.set(lx, ly, 0.06);
    group.add(land);
  }

  // Un-rotate local pad deltas since parent group applies comp.theta
  if (padCount > 0 && comp.theta) {
    const rad = ((comp.theta % 360) * Math.PI) / 180;
    const cos = Math.cos(rad);
    const sin = Math.sin(rad);
    for (let i = 1; i < group.children.length; i++) {
      const ch = group.children[i];
      const wx = ch.position.x;
      const wy = -ch.position.y;
      ch.position.x = wx * cos + wy * sin;
      ch.position.y = -( -wx * sin + wy * cos );
    }
  }

  addTopSilkLabel(group, bodyW, bodyD, h, comp.value || comp.comp_id, '#161920', '#e2e8f0', comp.comp_id);
  return group;
}

function buildLeadedIcMesh(comp, w, d, h) {
  const group = new THREE.Group();
  const isHorizontal = w >= d;
  const bodyW = isHorizontal ? w * 0.90 : w * 0.64;
  const bodyD = isHorizontal ? d * 0.64 : d * 0.90;

  const body = new THREE.Mesh(createChamferedBoxGeometry(bodyW, bodyD, h * 0.88, 0.09), SHARED_MATS.epoxyDark);
  body.position.z = h * 0.12;
  body.castShadow = true;
  body.receiveShadow = true;
  group.add(body);

  // Bent gull-wing leads for each pad
  const rad = ((comp.theta % 360) * Math.PI) / 180;
  const cos = Math.cos(rad);
  const sin = Math.sin(rad);

  for (const pad of comp.pads || []) {
    const dx = pad.x_mm - comp.x_mm;
    const dy = pad.y_mm - comp.y_mm;
    const lx = dx * cos + dy * sin;
    const ly = -(-dx * sin + dy * cos);

    const leadW = Math.min(0.45, Math.max(0.20, pad.width * 0.65));
    const leadD = Math.min(0.45, Math.max(0.20, pad.height * 0.65));
    // Foot on pad
    const foot = new THREE.Mesh(new THREE.BoxGeometry(leadW, leadD, 0.14), SHARED_MATS.tinnedLead);
    foot.position.set(lx, ly, 0.07);
    group.add(foot);

    // Vertical gull-wing shoulder
    const shoulder = new THREE.Mesh(
      new THREE.BoxGeometry(leadW * 0.82, leadD * 0.82, h * 0.45),
      SHARED_MATS.tinnedLead
    );
    shoulder.position.set(lx * 0.86, ly * 0.86, h * 0.30);
    group.add(shoulder);
  }

  addTopSilkLabel(group, bodyW, bodyD, h, comp.value || comp.comp_id, '#161920', '#e2e8f0', comp.comp_id);
  return group;
}

function buildRfModuleMesh(comp, w, d) {
  const group = new THREE.Group();
  // Dark substrate interposer
  const interposer = new THREE.Mesh(
    createChamferedBoxGeometry(w * 0.94, d * 0.94, 0.65, 0.04),
    SHARED_MATS.blackPlastic
  );
  interposer.castShadow = true;
  group.add(interposer);

  // Brushed nickel EMI shield can over 72% of module body
  const shieldW = w * 0.72;
  const shieldD = d * 0.84;
  const shieldH = 1.95;
  const shield = new THREE.Mesh(
    createChamferedBoxGeometry(shieldW, shieldD, shieldH, 0.10),
    SHARED_MATS.nickelShield
  );
  shield.position.set(-w * 0.10, 0, 0.65);
  shield.castShadow = true;
  group.add(shield);

  addTopSilkLabel(group, shieldW, shieldD, 0.65 + shieldH, comp.value || 'ESP32-C3', '#9aa2ad', '#111827', comp.comp_id);

  // Gold meandering PCB trace antenna on the overhang side
  const antX = w * 0.30;
  const meanderPts = [
    [-0.12, -d * 0.28],
    [0.12, -d * 0.28],
    [0.12, -d * 0.10],
    [-0.10, -d * 0.10],
    [-0.10, d * 0.10],
    [0.12, d * 0.10],
    [0.12, d * 0.28],
    [-0.12, d * 0.28],
  ];
  for (let i = 0; i < meanderPts.length - 1; i++) {
    const [x1, y1] = meanderPts[i];
    const [x2, y2] = meanderPts[i + 1];
    const segLen = Math.hypot(x2 - x1, y2 - y1);
    const trace = new THREE.Mesh(
      new THREE.BoxGeometry(Math.max(0.22, Math.abs(x2 - x1) + 0.18), Math.max(0.22, Math.abs(y2 - y1) + 0.18), 0.06),
      SHARED_MATS.enigGold
    );
    trace.position.set(antX + (x1 + x2) * 0.5, (y1 + y2) * 0.5, 0.68);
    if (segLen > 0) group.add(trace);
  }
  return group;
}

function buildPassiveChipMesh(comp, w, d) {
  const group = new THREE.Group();
  const isHoriz = w >= d;
  const len = Math.max(w, d) * 0.84;
  const wid = Math.min(w, d) * 0.72;
  const h = comp.comp_type === 'cap' ? Math.max(0.45, wid * 0.7) : comp.comp_type === 'led' ? 0.70 : 0.42;

  const bw = isHoriz ? len : wid;
  const bd = isHoriz ? wid : len;

  if (comp.comp_type === 'led') {
    const base = new THREE.Mesh(new THREE.BoxGeometry(bw, bd, 0.28), SHARED_MATS.ceramicWhite);
    base.position.z = 0.14;
    group.add(base);

    const valUpper = String(comp.value || comp.comp_id).toUpperCase();
    const domeColor = valUpper.includes('RED') || valUpper.includes('ERR') || valUpper.includes('PWR')
      ? 0xf43f5e
      : valUpper.includes('BLUE')
      ? 0x38bdf8
      : valUpper.includes('AMBER') || valUpper.includes('TX')
      ? 0xf59e0b
      : 0x10b981;
    const domeMat = new THREE.MeshPhysicalMaterial({
      color: domeColor,
      emissive: domeColor,
      emissiveIntensity: 0.42,
      roughness: 0.18,
      transmission: 0.55,
      transparent: true,
      opacity: 0.90,
    });
    const dome = new THREE.Mesh(new THREE.SphereGeometry(Math.min(bw, bd) * 0.36, 16, 12), domeMat);
    dome.scale.set(1, 1, 0.75);
    dome.position.z = 0.32;
    group.add(dome);
    return group;
  }

  const bodyMat = comp.comp_type === 'cap' ? SHARED_MATS.ceramicCap : SHARED_MATS.epoxyDark;
  const midScale = 0.68;
  const body = new THREE.Mesh(
    createChamferedBoxGeometry(isHoriz ? bw * midScale : bw, isHoriz ? bd : bd * midScale, h, 0.04),
    bodyMat
  );
  body.castShadow = true;
  group.add(body);

  // Silver/nickel end-cap terminals
  const capThick = len * 0.18;
  for (const dir of [-1, 1]) {
    const cap = new THREE.Mesh(
      new THREE.BoxGeometry(isHoriz ? capThick : bw * 1.02, isHoriz ? bd * 1.02 : capThick, h * 1.02),
      SHARED_MATS.tinnedLead
    );
    cap.position.set(
      isHoriz ? dir * (len * 0.5 - capThick * 0.5) : 0,
      isHoriz ? 0 : dir * (len * 0.5 - capThick * 0.5),
      h * 0.5
    );
    cap.castShadow = true;
    group.add(cap);
  }

  if (comp.comp_type === 'res') {
    const eia = parseResistorEiaCode(comp.value);
    addTopSilkLabel(group, bw * 0.72, bd * 0.85, h, eia, '#161920', '#f8fafc');
  }
  return group;
}

function buildUsbCMesh(w, d) {
  const group = new THREE.Group();
  const h = 3.16;
  const shellW = Math.min(w * 0.94, 9.0);
  const shellD = Math.min(d * 0.94, 7.4);

  const shell = new THREE.Mesh(
    createChamferedBoxGeometry(shellW, shellD, h, 0.25),
    SHARED_MATS.stainlessSteel
  );
  shell.castShadow = true;
  shell.receiveShadow = true;
  group.add(shell);

  // Recessed black cavity + internal tongue + gold contact pins
  const cavity = new THREE.Mesh(
    new THREE.BoxGeometry(shellW * 0.84, 0.6, h * 0.58),
    SHARED_MATS.blackPlastic
  );
  cavity.position.set(0, -shellD * 0.46, h * 0.48);
  group.add(cavity);

  const tongue = new THREE.Mesh(
    new THREE.BoxGeometry(shellW * 0.66, 0.9, 0.55),
    SHARED_MATS.enigGold
  );
  tongue.position.set(0, -shellD * 0.44, h * 0.48);
  group.add(tongue);
  return group;
}

function buildJstMesh(w, d) {
  const group = new THREE.Group();
  const h = 2.9;
  const body = new THREE.Mesh(createChamferedBoxGeometry(w * 0.90, d * 0.90, h, 0.12), SHARED_MATS.jstCream);
  body.castShadow = true;
  group.add(body);

  const slot = new THREE.Mesh(new THREE.BoxGeometry(w * 0.65, d * 0.5, 0.7), SHARED_MATS.blackPlastic);
  slot.position.z = h - 0.30;
  group.add(slot);

  const pins = Math.max(2, Math.min(6, Math.round(w / 1.4)));
  for (let i = 0; i < pins; i++) {
    const px = -w * 0.26 + (i / Math.max(1, pins - 1)) * (w * 0.52);
    const pin = new THREE.Mesh(new THREE.CylinderGeometry(0.14, 0.14, 1.4, 8), SHARED_MATS.enigGold);
    pin.rotation.x = Math.PI * 0.5;
    pin.position.set(px, 0, h * 0.65);
    group.add(pin);
  }
  return group;
}

function buildPinHeaderMesh(comp, w, d) {
  const group = new THREE.Group();
  const body = new THREE.Mesh(new THREE.BoxGeometry(w * 0.88, d * 0.88, 2.5), SHARED_MATS.blackPlastic);
  body.position.z = 1.25;
  body.castShadow = true;
  group.add(body);

  for (const pad of comp.pads || []) {
    const dx = pad.x_mm - comp.x_mm;
    const dy = pad.y_mm - comp.y_mm;
    const rad = ((comp.theta % 360) * Math.PI) / 180;
    const lx = dx * Math.cos(rad) + dy * Math.sin(rad);
    const ly = -(-dx * Math.sin(rad) + dy * Math.cos(rad));
    const pin = new THREE.Mesh(new THREE.BoxGeometry(0.64, 0.64, 7.2), SHARED_MATS.enigGold);
    pin.position.set(lx, ly, 3.6);
    pin.castShadow = true;
    group.add(pin);
  }
  return group;
}

function buildSwitchOrButtonMesh(comp, w, d, fpUpper) {
  const group = new THREE.Group();
  if (fpUpper.includes('KEYSWITCH') || fpUpper.includes('MX')) {
    const housing = new THREE.Mesh(createChamferedBoxGeometry(13.6, 13.6, 5.2, 0.35), SHARED_MATS.mxTranslucent);
    housing.castShadow = true;
    group.add(housing);
    const stem = new THREE.Mesh(new THREE.BoxGeometry(4.0, 4.0, 3.2), SHARED_MATS.mxStemRed);
    stem.position.z = 6.2;
    stem.castShadow = true;
    group.add(stem);
    return group;
  }

  const base = new THREE.Mesh(createChamferedBoxGeometry(w * 0.88, d * 0.88, 1.4, 0.08), SHARED_MATS.stainlessSteel);
  base.castShadow = true;
  group.add(base);

  if (fpUpper.includes('SLIDE') || String(comp.value || '').toUpperCase().includes('SPDT')) {
    const nub = new THREE.Mesh(new THREE.BoxGeometry(w * 0.28, d * 0.35, 1.1), SHARED_MATS.blackPlastic);
    nub.position.set(-w * 0.12, 0, 1.8);
    group.add(nub);
  } else {
    const dome = new THREE.Mesh(
      new THREE.CylinderGeometry(Math.min(w, d) * 0.24, Math.min(w, d) * 0.26, 0.9, 16),
      SHARED_MATS.enigGold
    );
    dome.rotation.x = Math.PI * 0.5;
    dome.position.z = 1.7;
    group.add(dome);
  }
  return group;
}

/**
 * Create a realistic procedural 3D component assembly aligned at local origin (z=0 bottom).
 * @param {object} comp Serialized BoardState component dictionary
 * @returns {THREE.Group}
 */
export function buildProceduralComponentMesh(comp) {
  const fpUpper = String(comp.footprint || '').toUpperCase();
  const valUpper = String(comp.value || '').toUpperCase();
  const ct = String(comp.comp_type || '').toLowerCase();
  const w = Math.max(0.5, Number(comp.body_width_mm || comp.width || 2.0));
  const d = Math.max(0.5, Number(comp.body_height_mm || comp.height || 2.0));

  let group;
  if (fpUpper.includes('MINI-1') || valUpper.includes('ESP32-C3-MINI') || (comp.special_layout_rules || []).includes('rf_antenna_keepout')) {
    group = buildRfModuleMesh(comp, w, d);
  } else if (fpUpper.includes('USB_C') || fpUpper.includes('USB-C') || valUpper.includes('USB_C') || valUpper.includes('MICRO-B')) {
    group = buildUsbCMesh(w, d);
  } else if (fpUpper.includes('JST') || fpUpper.includes('SH1.0')) {
    group = buildJstMesh(w, d);
  } else if (fpUpper.includes('KEYSWITCH') || fpUpper.includes('MX') || ct === 'switch' || fpUpper.includes('SLIDE') || fpUpper.includes('TACT')) {
    group = buildSwitchOrButtonMesh(comp, w, d, fpUpper);
  } else if (fpUpper.includes('PINHEADER') || ct === 'connector') {
    group = buildPinHeaderMesh(comp, w, d);
  } else if (valUpper.includes('OLED') || valUpper.includes('SSD1306') || ct === 'display') {
    group = new THREE.Group();
    const glass = new THREE.Mesh(createChamferedBoxGeometry(w * 0.94, d * 0.94, 1.45, 0.08), SHARED_MATS.oledGlass);
    glass.castShadow = true;
    group.add(glass);
    addTopSilkLabel(group, w * 0.9, d * 0.8, 1.45, 'OLED 0.49"', '#090d16', '#38bdf8', comp.comp_id);
  } else if (ct === 'inductor' || fpUpper.includes('INDUCTOR')) {
    group = new THREE.Group();
    const body = new THREE.Mesh(createChamferedBoxGeometry(w * 0.9, d * 0.9, 2.0, 0.14), SHARED_MATS.ferriteDark);
    body.castShadow = true;
    group.add(body);
    addTopSilkLabel(group, w * 0.85, d * 0.85, 2.0, comp.value || '4.7uH', '#2a2d34', '#cbd5e1', comp.comp_id);
  } else if (ct === 'crystal' || fpUpper.includes('CRYSTAL')) {
    group = new THREE.Group();
    const lid = new THREE.Mesh(createChamferedBoxGeometry(w * 0.88, d * 0.88, 0.72, 0.06), SHARED_MATS.nickelShield);
    lid.castShadow = true;
    group.add(lid);
    addTopSilkLabel(group, w * 0.8, d * 0.8, 0.72, comp.value || 'XTAL', '#b8bec8', '#0f172a');
  } else if (ct === 'cap' || ct === 'res' || ct === 'led') {
    group = buildPassiveChipMesh(comp, w, d);
  } else if (fpUpper.includes('QFN') || fpUpper.includes('LGA') || fpUpper.includes('DFN')) {
    group = buildQfnOrLgaMesh(comp, w, d, 0.85);
  } else if (fpUpper.includes('SOIC') || fpUpper.includes('SOT-23') || fpUpper.includes('LQFP') || fpUpper.includes('TSSOP')) {
    group = buildLeadedIcMesh(comp, w, d, 1.35);
  } else if (ct === 'ic' || ct === 'diode') {
    group = buildQfnOrLgaMesh(comp, w, d, ct === 'diode' ? 0.95 : 1.25);
  } else {
    group = buildQfnOrLgaMesh(comp, w, d, 0.9);
  }

  group.userData = {
    compId: comp.comp_id,
    side: comp.side || 'top',
  };
  return group;
}
