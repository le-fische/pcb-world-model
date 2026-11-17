/**
 * Real-Time PBR 3D PCB Viewport (rev9).
 * Renders drilled FR4 substrate, translucent solder mask themes, copper pours & traces,
 * ENIG immersion gold pads, high-DPI Canvas2D silkscreen, procedural STEP-quality 3D
 * components, smooth 180-deg Flip-Board inspection, and click-to-cross-probe raycasting.
 */

import * as THREE from 'three';
import { OrbitControls } from './vendor/three/OrbitControls.js';
import { RoomEnvironment } from './vendor/three/RoomEnvironment.js';
import { buildProceduralComponentMesh } from './component_meshes_3d.js';

export const MASK_THEMES = {
  green: { label: 'KiCad Green', mask: 0x0d5c32, copperTint: 0x168049, cssHex: '#0d5c32' },
  black: { label: 'Matte Black', mask: 0x14181f, copperTint: 0x262c36, cssHex: '#14181f' },
  blue: { label: 'Royal Blue', mask: 0x0f3b8c, copperTint: 0x1a56c4, cssHex: '#0f3b8c' },
  red: { label: 'Crimson Red', mask: 0x8b1422, copperTint: 0xb81d2f, cssHex: '#8b1422' },
  white: { label: 'Arctic White', mask: 0xe6ebf2, copperTint: 0xd0d8e4, cssHex: '#e6ebf2' },
};

function createFr4WeaveBumpTexture() {
  const cvs = document.createElement('canvas');
  cvs.width = 128;
  cvs.height = 128;
  const ctx = cvs.getContext('2d');
  if (ctx) {
    ctx.fillStyle = '#808080';
    ctx.fillRect(0, 0, 128, 128);
    for (let y = 0; y < 128; y += 8) {
      for (let x = 0; x < 128; x += 8) {
        const checker = ((x + y) / 8) % 2 === 0;
        ctx.fillStyle = checker ? '#9a9a9a' : '#686868';
        ctx.fillRect(x, y, 7, 7);
      }
    }
  }
  const tex = new THREE.CanvasTexture(cvs);
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(14, 14);
  return tex;
}

function createSilkscreenTexture(board, side = 'top', maskThemeKey = 'green') {
  const wMm = Math.max(10, Number(board.width_mm || 50));
  const hMm = Math.max(10, Number(board.height_mm || 50));
  const pxPerMm = 24;
  const cvs = document.createElement('canvas');
  cvs.width = Math.min(2048, Math.round(wMm * pxPerMm));
  cvs.height = Math.min(2048, Math.round(hMm * pxPerMm));
  const ctx = cvs.getContext('2d');
  if (!ctx) return null;

  const sx = cvs.width / wMm;
  const sy = cvs.height / hMm;
  const ink = maskThemeKey === 'white' ? '#1e293b' : '#f8fafc';

  ctx.clearRect(0, 0, cvs.width, cvs.height);
  ctx.strokeStyle = ink;
  ctx.fillStyle = ink;

  // Board perimeter fiducials & version silkscreen
  ctx.lineWidth = Math.max(1.5, 0.12 * sx);
  ctx.globalAlpha = 0.72;
  ctx.strokeRect(0.35 * sx, 0.35 * sy, (wMm - 0.7) * sx, (hMm - 0.7) * sy);

  const silkByComp = new Map();
  for (const adj of board.silkscreen_adjustments || []) {
    silkByComp.set(adj.comp_id, adj);
  }

  for (const comp of board.components || []) {
    const compSide = comp.side || 'top';
    if (compSide !== side || comp.comp_type === 'mounting_hole') continue;

    const cx = comp.x_mm * sx;
    const cy = side === 'bottom' ? (hMm - comp.y_mm) * sy : comp.y_mm * sy;
    const cw = (comp.width || 2) * sx;
    const ch = (comp.height || 2) * sy;

    ctx.save();
    ctx.globalAlpha = 0.85;
    ctx.strokeStyle = ink;
    ctx.lineWidth = Math.max(1.2, 0.10 * sx);
    ctx.strokeRect(cx - cw * 0.48, cy - ch * 0.48, cw * 0.96, ch * 0.96);

    // Pin-1 silkscreen dot near pad 1
    const firstPad = (comp.pads || [])[0];
    if (firstPad) {
      const px = firstPad.x_mm * sx;
      const py = side === 'bottom' ? (hMm - firstPad.y_mm) * sy : firstPad.y_mm * sy;
      ctx.beginPath();
      ctx.arc(px + (px < cx ? -2 : 2), py + (py < cy ? -2 : 2), Math.max(2.2, 0.18 * sx), 0, Math.PI * 2);
      ctx.fill();
    }

    // RefDes label placed using rev10 silkscreen-scrubbed coordinates (never overlaps pads)
    const adj = silkByComp.get(comp.comp_id);
    const labelFontMm = adj ? Number(adj.font_mm || 0.9) : 0.95;
    const fontSize = Math.max(11, Math.min(24, Math.round(labelFontMm * sx)));
    ctx.font = `700 ${fontSize}px "JetBrains Mono", monospace`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    const labelX = adj ? Number(adj.world_x_mm) * sx : cx;
    const labelY = adj
      ? (side === 'bottom' ? (hMm - Number(adj.world_y_mm)) * sy : Number(adj.world_y_mm) * sy)
      : Math.max(fontSize + 4, cy - ch * 0.52 - 4);
    ctx.fillText(comp.comp_id, labelX, labelY);
    ctx.restore();
  }

  // Board silkscreen title in bottom-left margin
  ctx.globalAlpha = 0.65;
  ctx.font = `600 ${Math.max(12, Math.round(0.9 * sx))}px "JetBrains Mono", monospace`;
  ctx.textAlign = 'left';
  ctx.textBaseline = 'bottom';
  ctx.fillText(
    `${board.board_name || 'PCB'} · ${side === 'top' ? 'F.SilkS' : 'B.SilkS'}`,
    1.0 * sx,
    (hMm - 0.65) * sy
  );

  const tex = new THREE.CanvasTexture(cvs);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 4;
  return tex;
}

export class PcbPbr3DViewer {
  /**
   * @param {HTMLElement} container
   * @param {object} callbacks { onSelectComponent, onSelectNet, onHoverCoords }
   */
  constructor(container, callbacks = {}) {
    this.container = container;
    this.callbacks = callbacks;
    this.webglAvailable = true;
    this.maskTheme = 'green';
    this.flipped = false;
    this.flipProgress = 0; // 0 = top up, 1 = bottom up
    this.targetFlip = 0;
    this.board = null;
    this.layers = null;
    this.selectedCompId = null;
    this.selectedNetId = null;
    this.clickableTargets = [];
    this.componentGroups = new Map();
    this.traceMeshes = [];
    this.padMeshes = [];
    this.viaMeshes = [];
    this._rafId = null;
    this._pointerDownPos = null;
    this._disposed = false;

    this._initScene();
  }

  _initScene() {
    let renderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    } catch (err) {
      this.webglAvailable = false;
      this._renderUnavailableNotice(err?.message || 'WebGL2 context unavailable');
      return;
    }

    this.renderer = renderer;
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.12;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.domElement.className = 'pbr-3d-canvas';
    this.container.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x070b12);

    const pmrem = new THREE.PMREMGenerator(this.renderer);
    this.scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    pmrem.dispose();

    const w = Math.max(320, this.container.clientWidth || 640);
    const h = Math.max(260, this.container.clientHeight || 480);
    this.camera = new THREE.PerspectiveCamera(36, w / h, 0.2, 800);
    this.camera.position.set(0, 48, 52);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.minDistance = 8;
    this.controls.maxDistance = 240;

    // Studio key + fill + rim lighting
    this.scene.add(new THREE.AmbientLight(0xdbeafe, 0.35));
    const keyLight = new THREE.DirectionalLight(0xfffbeb, 1.55);
    keyLight.position.set(35, 65, 42);
    keyLight.castShadow = true;
    keyLight.shadow.mapSize.set(1024, 1024);
    keyLight.shadow.bias = -0.0005;
    this.scene.add(keyLight);

    const rimLight = new THREE.DirectionalLight(0x7dd3fc, 0.65);
    rimLight.position.set(-45, 28, -38);
    this.scene.add(rimLight);

    this.pcbPivot = new THREE.Group();
    this.boardGroup = new THREE.Group();
    this.boardGroup.rotation.x = -Math.PI / 2; // Map board +Z normal to world +Y
    this.pcbPivot.add(this.boardGroup);
    this.scene.add(this.pcbPivot);

    this.fr4Bump = createFr4WeaveBumpTexture();
    this.raycaster = new THREE.Raycaster();
    this.pointerNdc = new THREE.Vector2();

    this._bindEvents();
    this.resize();
    this._startLoop();
  }

  _renderUnavailableNotice(reason) {
    const msg = document.createElement('div');
    msg.className = 'pbr-fallback-notice';
    msg.textContent = `WebGL2 3D PBR renderer unavailable (${reason}). Use 2D CAD or Schematic tab.`;
    this.container.appendChild(msg);
  }

  _bindEvents() {
    const dom = this.renderer.domElement;
    dom.addEventListener('pointerdown', (e) => {
      this._pointerDownPos = { x: e.clientX, y: e.clientY };
    });
    dom.addEventListener('pointerup', (e) => {
      if (!this._pointerDownPos) return;
      const dist = Math.hypot(e.clientX - this._pointerDownPos.x, e.clientY - this._pointerDownPos.y);
      this._pointerDownPos = null;
      if (dist > 5) return; // Ignore orbit drags
      this._handlePick(e);
    });
  }

  _handlePick(event) {
    if (!this.webglAvailable || !this.clickableTargets.length) return;
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.pointerNdc.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this.pointerNdc.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    this.raycaster.setFromCamera(this.pointerNdc, this.camera);
    const hits = this.raycaster.intersectObjects(this.clickableTargets, true);
    if (!hits.length) {
      this.callbacks.onClearSelection?.();
      return;
    }

    let obj = hits[0].object;
    while (obj && !obj.userData?.compId && obj.userData?.netId === undefined && obj.parent) {
      obj = obj.parent;
    }
    if (obj?.userData?.compId) {
      this.callbacks.onSelectComponent?.(obj.userData.compId);
    } else if (obj?.userData?.netId !== undefined) {
      this.callbacks.onSelectNet?.(obj.userData.netId, obj.userData.netName || '');
    }
  }

  setMaskTheme(themeKey) {
    this.maskTheme = MASK_THEMES[themeKey] ? themeKey : 'green';
    if (this.board) this.updateBoard(this.board, this.layers);
  }

  setFlip(flipped) {
    this.flipped = Boolean(flipped);
    this.targetFlip = this.flipped ? 1 : 0;
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reduced && this.pcbPivot) {
      this.flipProgress = this.targetFlip;
      this.pcbPivot.rotation.x = this.flipProgress * Math.PI;
    }
  }

  toggleFlip() {
    this.setFlip(!this.flipped);
    return this.flipped;
  }

  resetCamera() {
    if (!this.webglAvailable) return;
    const maxDim = Math.max(this.board?.width_mm || 50, this.board?.height_mm || 50);
    this.controls.target.set(0, 0, 0);
    this.camera.position.set(0, maxDim * 1.05, maxDim * 1.12);
    this.controls.update();
  }

  setSelection(compId, netId) {
    this.selectedCompId = compId;
    this.selectedNetId = netId;
    this._applyHighlightState();
  }

  _applyHighlightState() {
    const hasFilter = this.selectedCompId !== null || this.selectedNetId !== null;
    for (const [cid, grp] of this.componentGroups.entries()) {
      const active = !hasFilter || cid === this.selectedCompId;
      grp.traverse((child) => {
        if (!child.isMesh || !child.material) return;
        const m = child.material;
        if (m.userData?.origOpacity === undefined) {
          m.userData = { ...m.userData, origOpacity: m.opacity ?? 1, origTransparent: m.transparent ?? false };
        }
        if (hasFilter && !active) {
          // Only dim non-selected components when selecting a component specifically
          if (this.selectedCompId) {
            m.opacity = Math.min(m.userData.origOpacity, 0.35);
            m.transparent = true;
          } else {
            m.opacity = m.userData.origOpacity;
            m.transparent = m.userData.origTransparent;
          }
        } else {
          m.opacity = m.userData.origOpacity;
          m.transparent = m.userData.origTransparent;
        }
      });
    }

    for (const mesh of this.traceMeshes) {
      const isSelected = this.selectedNetId !== null && mesh.userData.netId === this.selectedNetId;
      mesh.material.opacity = !this.selectedNetId ? 0.96 : (isSelected ? 1.0 : 0.18);
      mesh.material.transparent = true;
      mesh.material.emissive.setHex(isSelected ? 0x38bdf8 : 0x000000);
      mesh.material.emissiveIntensity = isSelected ? 0.55 : 0;
    }

    for (const mesh of this.padMeshes) {
      const compMatch = this.selectedCompId && mesh.userData.compId === this.selectedCompId;
      const netMatch = this.selectedNetId !== null && mesh.userData.netId === this.selectedNetId;
      const active = !hasFilter || compMatch || netMatch;
      mesh.material.opacity = active ? 1.0 : 0.28;
      mesh.material.transparent = !active;
      mesh.material.emissive.setHex(active && hasFilter ? 0xfbbf24 : 0x000000);
      mesh.material.emissiveIntensity = active && hasFilter ? 0.35 : 0;
    }

    for (const mesh of this.viaMeshes) {
      const active = !this.selectedNetId || mesh.userData.netId === this.selectedNetId;
      mesh.material.opacity = active ? 1.0 : 0.22;
      mesh.material.transparent = !active;
    }
  }

  _clearBoardGroup() {
    this.clickableTargets = [];
    this.componentGroups.clear();
    this.traceMeshes = [];
    this.padMeshes = [];
    this.viaMeshes = [];
    while (this.boardGroup.children.length > 0) {
      const obj = this.boardGroup.children.pop();
      this.boardGroup.remove(obj);
      obj.traverse((ch) => {
        if (ch.geometry) ch.geometry.dispose();
      });
    }
  }

  updateBoard(board, layers = null) {
    if (!this.webglAvailable || !board) return;
    const isNewBoard = !this.board || this.board.board_name !== board.board_name || this.board.width_mm !== board.width_mm;
    this.board = board;
    this.layers = layers || this.layers || {
      l1: true, in1: true, in2: true, l2: true, pour: true, vias: true, courtyard: true, ratsnest: true,
    };

    this._clearBoardGroup();

    const wMm = Number(board.width_mm || 50);
    const hMm = Number(board.height_mm || 50);
    const thk = Math.max(0.4, Number(board.thickness_mm || 1.6));
    const numLayers = Number(board.layers || 2);
    const botLayer = Math.max(1, numLayers - 1);
    const theme = MASK_THEMES[this.maskTheme] || MASK_THEMES.green;

    const toLocal = (xMm, yMm) => [xMm - wMm * 0.5, -(yMm - hMm * 0.5)];

    // 1. Drilled FR4 Substrate (rev10 corner fillets + real NPTH mounting holes & via/THT drills)
    const reqFillet = Number(board.corner_fillet_radius_mm ?? 0);
    const rCorner = reqFillet > 0.1
      ? Math.min(reqFillet, wMm * 0.25, hMm * 0.25)
      : Math.min(1.5, wMm * 0.04, hMm * 0.04);
    const hw = wMm * 0.5;
    const hh = hMm * 0.5;
    const pcbShape = new THREE.Shape();
    pcbShape.moveTo(-hw + rCorner, -hh);
    pcbShape.lineTo(hw - rCorner, -hh);
    pcbShape.quadraticCurveTo(hw, -hh, hw, -hh + rCorner);
    pcbShape.lineTo(hw, hh - rCorner);
    pcbShape.quadraticCurveTo(hw, hh, hw - rCorner, hh);
    pcbShape.lineTo(-hw + rCorner, hh);
    pcbShape.quadraticCurveTo(-hw, hh, -hw, hh - rCorner);
    pcbShape.lineTo(-hw, -hh + rCorner);
    pcbShape.quadraticCurveTo(-hw, -hh, -hw + rCorner, -hh);

    const holeRecords = [];
    for (const comp of board.components || []) {
      if (comp.comp_type === 'mounting_hole') {
        const [lx, ly] = toLocal(comp.x_mm, comp.y_mm);
        const rad = Math.max(0.45, Math.min(comp.width || 2.5, comp.height || 2.5) * 0.34);
        holeRecords.push({ lx, ly, rad, isNpth: true });
      } else {
        for (const pad of comp.pads || []) {
          if (pad.layer_mask === 3) {
            const [lx, ly] = toLocal(pad.x_mm, pad.y_mm);
            const rad = Math.max(0.22, Math.min(pad.width || 1.0, pad.height || 1.0) * 0.24);
            holeRecords.push({ lx, ly, rad, isNpth: false });
          }
        }
      }
    }
    for (const via of board.vias || []) {
      const [lx, ly] = toLocal(via.x_mm, via.y_mm);
      const rad = Math.max(0.14, (via.drill_mm || 0.3) * 0.5);
      holeRecords.push({ lx, ly, rad, isNpth: false });
    }

    for (const h of holeRecords) {
      if (Math.abs(h.lx) + h.rad >= hw - 0.1 || Math.abs(h.ly) + h.rad >= hh - 0.1) continue;
      const holePath = new THREE.Path();
      holePath.absarc(h.lx, h.ly, h.rad, 0, Math.PI * 2, true);
      pcbShape.holes.push(holePath);
    }

    const fr4Geo = new THREE.ExtrudeGeometry(pcbShape, { depth: thk, bevelEnabled: false, curveSegments: 18 });
    fr4Geo.translate(0, 0, -thk * 0.5);
    const fr4Mat = new THREE.MeshPhysicalMaterial({
      color: 0x8d976e,
      roughness: 0.62,
      metalness: 0.02,
      bumpMap: this.fr4Bump,
      bumpScale: 0.015,
    });
    const fr4Mesh = new THREE.Mesh(fr4Geo, fr4Mat);
    fr4Mesh.receiveShadow = true;
    fr4Mesh.castShadow = true;
    this.boardGroup.add(fr4Mesh);

    // 2. Translucent Solder Mask Skins (Top & Bottom) over Copper Pours + Traces
    const maskShape = pcbShape.clone();
    const maskMat = new THREE.MeshPhysicalMaterial({
      color: theme.mask,
      roughness: 0.30,
      metalness: 0.04,
      clearcoat: 0.52,
      clearcoatRoughness: 0.20,
      transparent: true,
      opacity: 0.84,
      polygonOffset: true,
      polygonOffsetFactor: -1,
    });
    const maskTopGeo = new THREE.ShapeGeometry(maskShape, 18);
    const maskTop = new THREE.Mesh(maskTopGeo, maskMat);
    maskTop.position.z = thk * 0.5 + 0.042;
    maskTop.receiveShadow = true;
    this.boardGroup.add(maskTop);

    const maskBot = new THREE.Mesh(maskTopGeo.clone(), maskMat);
    maskBot.rotation.y = Math.PI;
    maskBot.position.z = -(thk * 0.5 + 0.042);
    maskBot.receiveShadow = true;
    this.boardGroup.add(maskBot);

    // 3. B.Cu Ground Pour beneath Bottom Solder Mask
    if (this.layers.pour && board.gnd_pour) {
      const pour = board.gnd_pour;
      const pw = Math.max(1, pour.x1_mm - pour.x0_mm);
      const ph = Math.max(1, pour.y1_mm - pour.y0_mm);
      const [px, py] = toLocal((pour.x0_mm + pour.x1_mm) * 0.5, (pour.y0_mm + pour.y1_mm) * 0.5);
      const pourMat = new THREE.MeshStandardMaterial({
        color: theme.copperTint,
        metalness: 0.72,
        roughness: 0.34,
        side: THREE.DoubleSide,
      });
      const pourMesh = new THREE.Mesh(new THREE.PlaneGeometry(pw, ph), pourMat);
      pourMesh.position.set(px, py, -(thk * 0.5 + 0.015));
      this.boardGroup.add(pourMesh);
    }

    // 4. Copper Trace Segments (sit beneath translucent solder mask so traces create physical copper relief)
    for (const seg of board.traces || []) {
      const isTop = seg.layer === 0;
      const isBot = seg.layer === botLayer;
      if (isTop && !this.layers.l1) continue;
      if (isBot && !this.layers.l2) continue;
      if (!isTop && !isBot && !this.layers.in1 && !this.layers.in2) continue;

      const [x1, y1] = toLocal(seg.x1_mm, seg.y1_mm);
      const [x2, y2] = toLocal(seg.x2_mm, seg.y2_mm);
      const len = Math.hypot(x2 - x1, y2 - y1);
      if (len < 0.01) continue;

      let zPos = 0;
      let color = 0xdf825f; // Bare copper under solder mask
      if (isTop) {
        zPos = thk * 0.5 + 0.024;
        color = 0xe8795e;
      } else if (isBot) {
        zPos = -(thk * 0.5 + 0.024);
        color = 0x38bdf8;
      } else {
        zPos = thk * 0.5 - (seg.layer / Math.max(1, numLayers - 1)) * thk;
        color = seg.layer === 1 ? 0x10b981 : 0xa855f7;
      }

      const traceMat = new THREE.MeshStandardMaterial({
        color,
        metalness: 0.78,
        roughness: 0.28,
        transparent: true,
        opacity: 0.96,
      });
      const traceGeo = new THREE.BoxGeometry(len + seg.width_mm * 0.75, Math.max(0.14, seg.width_mm), 0.036);
      const traceMesh = new THREE.Mesh(traceGeo, traceMat);
      traceMesh.position.set((x1 + x2) * 0.5, (y1 + y2) * 0.5, zPos);
      traceMesh.rotation.z = Math.atan2(y2 - y1, x2 - x1);
      traceMesh.userData = { netId: seg.net_id, netName: seg.net_name };
      this.boardGroup.add(traceMesh);
      this.traceMeshes.push(traceMesh);
      this.clickableTargets.push(traceMesh);
    }

    // 4b. Copper Teardrop Fillets beneath Solder Mask (rev10)
    for (const td of board.teardrops || []) {
      const isTop = td.layer === 0;
      const isBot = td.layer === botLayer;
      if (isTop && !this.layers.l1) continue;
      if (isBot && !this.layers.l2) continue;
      const pts = td.pts || [];
      if (pts.length < 3) continue;
      const tdShape = new THREE.Shape();
      const [p0x, p0y] = toLocal(pts[0][0], pts[0][1]);
      tdShape.moveTo(p0x, p0y);
      for (let i = 1; i < pts.length; i++) {
        const [px, py] = toLocal(pts[i][0], pts[i][1]);
        tdShape.lineTo(px, py);
      }
      tdShape.closePath();
      const tdMat = new THREE.MeshStandardMaterial({
        color: isBot ? 0x38bdf8 : 0xe8795e,
        metalness: 0.78,
        roughness: 0.28,
        transparent: true,
        opacity: 0.94,
      });
      const tdMesh = new THREE.Mesh(new THREE.ShapeGeometry(tdShape), tdMat);
      tdMesh.position.z = isBot ? -(thk * 0.5 + 0.025) : (thk * 0.5 + 0.025);
      if (isBot) tdMesh.rotation.y = Math.PI;
      this.boardGroup.add(tdMesh);
      this.traceMeshes.push(tdMesh);
    }

    // 5. Plated Through-Hole & Micro-Via Barrels + ENIG Gold Annular Rings
    if (this.layers.vias) {
      for (const via of board.vias || []) {
        const [vx, vy] = toLocal(via.x_mm, via.y_mm);
        const sL = via.start_layer ?? 0;
        const eL = via.end_layer ?? botLayer;
        const zStart = thk * 0.5 - (sL / Math.max(1, numLayers - 1)) * thk;
        const zEnd = thk * 0.5 - (eL / Math.max(1, numLayers - 1)) * thk;
        const span = Math.max(0.12, Math.abs(zStart - zEnd) + 0.08);
        const rOuter = Math.max(0.18, (via.diameter_mm || 0.6) * 0.5);
        const viaMat = new THREE.MeshStandardMaterial({
          color: 0xe2b857,
          metalness: 0.92,
          roughness: 0.18,
          transparent: true,
        });
        const barrel = new THREE.Mesh(new THREE.CylinderGeometry(rOuter, rOuter, span, 14), viaMat);
        barrel.rotation.x = Math.PI * 0.5;
        barrel.position.set(vx, vy, (zStart + zEnd) * 0.5);
        barrel.userData = { netId: via.net_id };
        this.boardGroup.add(barrel);
        this.viaMeshes.push(barrel);
        this.clickableTargets.push(barrel);
      }
    }

    // 6. Exposed ENIG Immersion Gold Pads (sit proud of solder mask openings!)
    for (const comp of board.components || []) {
      if (comp.comp_type === 'mounting_hole') continue;
      for (const pad of comp.pads || []) {
        const [px, py] = toLocal(pad.x_mm, pad.y_mm);
        const pw = Math.max(0.18, pad.width);
        const ph = Math.max(0.18, pad.height);
        const drawPad = (zCoord) => {
          const padMat = new THREE.MeshStandardMaterial({
            color: 0xe2b857,
            metalness: 0.92,
            roughness: 0.18,
          });
          const padGeo = pad.layer_mask === 3
            ? new THREE.CylinderGeometry(Math.max(pw, ph) * 0.48, Math.max(pw, ph) * 0.48, 0.045, 16)
            : new THREE.BoxGeometry(pw, ph, 0.045);
          const padMesh = new THREE.Mesh(padGeo, padMat);
          if (pad.layer_mask === 3) padMesh.rotation.x = Math.PI * 0.5;
          padMesh.position.set(px, py, zCoord);
          padMesh.userData = { compId: comp.comp_id, netId: pad.net_id, netName: pad.net_name };
          this.boardGroup.add(padMesh);
          this.padMeshes.push(padMesh);
          this.clickableTargets.push(padMesh);
        };
        if (pad.layer_mask & 1) drawPad(thk * 0.5 + 0.056);
        if (pad.layer_mask & 2) drawPad(-(thk * 0.5 + 0.056));
      }
    }

    // 6b. Exposed ENIG Gold 1.0mm Debug Test Points (rev10)
    for (const tp of board.test_points || []) {
      const [tpx, tpy] = toLocal(tp.x_mm, tp.y_mm);
      const tpr = Math.max(0.25, (tp.diameter_mm || 1.0) * 0.5);
      const tpMat = new THREE.MeshStandardMaterial({ color: 0xe2b857, metalness: 0.92, roughness: 0.18 });
      const tpMesh = new THREE.Mesh(new THREE.CylinderGeometry(tpr, tpr, 0.045, 20), tpMat);
      tpMesh.rotation.x = Math.PI * 0.5;
      tpMesh.position.set(tpx, tpy, (tp.layer || 0) > 0 ? -(thk * 0.5 + 0.056) : (thk * 0.5 + 0.056));
      tpMesh.userData = { netId: tp.net_id, netName: tp.net_name };
      this.boardGroup.add(tpMesh);
      this.padMeshes.push(tpMesh);
      this.clickableTargets.push(tpMesh);
    }

    // 7. High-DPI F.SilkS & B.SilkS Silkscreen Planes
    const topSilkTex = createSilkscreenTexture(board, 'top', this.maskTheme);
    if (topSilkTex) {
      const topSilk = new THREE.Mesh(
        new THREE.PlaneGeometry(wMm, hMm),
        new THREE.MeshStandardMaterial({ map: topSilkTex, transparent: true, alphaTest: 0.04, roughness: 0.6 })
      );
      topSilk.position.z = thk * 0.5 + 0.050;
      this.boardGroup.add(topSilk);
    }
    const botSilkTex = createSilkscreenTexture(board, 'bottom', this.maskTheme);
    if (botSilkTex) {
      const botSilk = new THREE.Mesh(
        new THREE.PlaneGeometry(wMm, hMm),
        new THREE.MeshStandardMaterial({ map: botSilkTex, transparent: true, alphaTest: 0.04, roughness: 0.6 })
      );
      botSilk.rotation.y = Math.PI;
      botSilk.position.z = -(thk * 0.5 + 0.050);
      this.boardGroup.add(botSilk);
    }

    // 8. Procedural STEP-Quality 3D Component Packages
    for (const comp of board.components || []) {
      if (comp.comp_type === 'mounting_hole') continue;
      const [cx, cy] = toLocal(comp.x_mm, comp.y_mm);
      const isBottom = comp.side === 'bottom';
      const meshGroup = buildProceduralComponentMesh(comp);
      meshGroup.position.set(cx, cy, isBottom ? -(thk * 0.5 + 0.055) : thk * 0.5 + 0.055);
      meshGroup.rotation.z = -((Number(comp.theta) || 0) * Math.PI) / 180;
      if (isBottom) {
        meshGroup.rotation.x = Math.PI;
      }
      this.boardGroup.add(meshGroup);
      this.componentGroups.set(comp.comp_id, meshGroup);
      this.clickableTargets.push(meshGroup);
    }

    if (isNewBoard) {
      this.resetCamera();
    }
    this._applyHighlightState();
  }

  resize() {
    if (!this.webglAvailable || !this.renderer) return;
    const w = Math.max(320, this.container.clientWidth || 640);
    const h = Math.max(260, this.container.clientHeight || 480);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h, false);
  }

  _startLoop() {
    const tick = () => {
      if (this._disposed) return;
      this._rafId = requestAnimationFrame(tick);
      if (Math.abs(this.targetFlip - this.flipProgress) > 0.002) {
        this.flipProgress += (this.targetFlip - this.flipProgress) * 0.14;
        this.pcbPivot.rotation.x = this.flipProgress * Math.PI;
      } else if (this.flipProgress !== this.targetFlip) {
        this.flipProgress = this.targetFlip;
        this.pcbPivot.rotation.x = this.flipProgress * Math.PI;
      }
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
    };
    tick();
  }
}
