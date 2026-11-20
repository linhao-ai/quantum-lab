/* ============================================================
   量子计算 · 交互模型实验室
   纯前端态矢量模拟器 + 三个交互模型
   ============================================================ */
(function () {
  'use strict';

  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const SQ2 = Math.SQRT1_2;
  const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
  const fmt = (x, d = 3) => (Math.abs(x) < 5e-4 ? 0 : x).toFixed(d);
  const signed = (x, d = 3) => (x >= 0 ? '+' : '−') + Math.abs(x).toFixed(d);
  const TAU = Math.PI * 2;

  /* ============================================================
     一、态矢量模拟器
     ============================================================ */
  // 单比特门矩阵：[ [re,im] x4 ]，行优先
  const M = {
    H: [[SQ2, 0], [SQ2, 0], [SQ2, 0], [-SQ2, 0]],
    X: [[0, 0], [1, 0], [1, 0], [0, 0]],
    Y: [[0, 0], [0, -1], [0, 1], [0, 0]],
    Z: [[1, 0], [0, 0], [0, 0], [-1, 0]],
    S: [[1, 0], [0, 0], [0, 0], [0, 1]],
    T: [[1, 0], [0, 0], [0, 0], [Math.cos(Math.PI / 4), Math.sin(Math.PI / 4)]],
  };

  function rotation(g, deg) {
    const t = (deg * Math.PI) / 180;
    const c = Math.cos(t / 2), s = Math.sin(t / 2);
    if (g === 'RX') return [[c, 0], [0, -s], [0, -s], [c, 0]];
    if (g === 'RY') return [[c, 0], [-s, 0], [s, 0], [c, 0]];
    return [[c, -s], [0, 0], [0, 0], [c, s]]; // RZ
  }

  function gateMatrix(op) {
    if (op.g === 'RX' || op.g === 'RY' || op.g === 'RZ') return rotation(op.g, op.deg || 0);
    return M[op.g];
  }

  class QState {
    constructor(n) {
      this.n = n;
      this.N = 1 << n;
      this.re = new Float64Array(this.N);
      this.im = new Float64Array(this.N);
      this.re[0] = 1;
    }
    clone() {
      const s = new QState(this.n);
      s.re.set(this.re);
      s.im.set(this.im);
      return s;
    }
    /** q0 是最高位 */
    bit(q) { return 1 << (this.n - 1 - q); }

    apply(m, q, controls) {
      const bit = this.bit(q);
      let cmask = 0;
      if (controls) for (const c of controls) cmask |= this.bit(c);
      const a = m[0], b = m[1], c = m[2], d = m[3];
      for (let i = 0; i < this.N; i++) {
        if (i & bit) continue;
        if (cmask && (i & cmask) !== cmask) continue;
        const j = i | bit;
        const r0 = this.re[i], i0 = this.im[i];
        const r1 = this.re[j], i1 = this.im[j];
        // u = a*r0 + b*r1 ; v = c*r0 + d*r1  (复数乘加)
        const ur = a[0] * r0 - a[1] * i0 + b[0] * r1 - b[1] * i1;
        const ui = a[0] * i0 + a[1] * r0 + b[0] * i1 + b[1] * r1;
        const vr = c[0] * r0 - c[1] * i0 + d[0] * r1 - d[1] * i1;
        const vi = c[0] * i0 + c[1] * r0 + d[0] * i1 + d[1] * r1;
        this.re[i] = ur; this.im[i] = ui;
        this.re[j] = vr; this.im[j] = vi;
      }
    }

    cx(control, target) {
      const cb = this.bit(control), tb = this.bit(target);
      for (let i = 0; i < this.N; i++) {
        if ((i & cb) && !(i & tb)) {
          const j = i | tb;
          const r = this.re[i], im = this.im[i];
          this.re[i] = this.re[j]; this.im[i] = this.im[j];
          this.re[j] = r; this.im[j] = im;
        }
      }
    }

    prob(i) { return this.re[i] * this.re[i] + this.im[i] * this.im[i]; }
    probs() {
      const out = new Array(this.N);
      for (let i = 0; i < this.N; i++) out[i] = this.prob(i);
      return out;
    }
    norm() {
      let s = 0;
      for (let i = 0; i < this.N; i++) s += this.prob(i);
      return Math.sqrt(s);
    }
    measure() {
      const r = Math.random();
      let acc = 0;
      for (let i = 0; i < this.N; i++) {
        acc += this.prob(i);
        if (r <= acc) return i;
      }
      return this.N - 1;
    }
    bits(i) { return i.toString(2).padStart(this.n, '0'); }
    ket(i) { return '|' + this.bits(i) + '⟩'; }
  }

  /** circuit = { n, cols: [ [op,...], ... ] }，op = {g,q} 或 {g:'CX', c, t} */
  function simulate(circuit) {
    const st = new QState(circuit.n);
    for (const col of circuit.cols) {
      for (const op of col) if (op.g !== 'CX') st.apply(gateMatrix(op), op.q);
      for (const op of col) if (op.g === 'CX') st.cx(op.c, op.t);
    }
    return st;
  }

  function sampleCounts(probs, shots) {
    const N = probs.length;
    const counts = new Array(N).fill(0);
    // 累积分布 + 二分，避免 shots 大时 O(N·shots)
    const cum = new Array(N);
    let acc = 0;
    for (let i = 0; i < N; i++) { acc += probs[i]; cum[i] = acc; }
    const top = cum[N - 1] || 1;
    for (let s = 0; s < shots; s++) {
      const r = Math.random() * top;
      let lo = 0, hi = N - 1;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (cum[mid] < r) lo = mid + 1; else hi = mid;
      }
      counts[lo]++;
    }
    return counts;
  }

  /* ============================================================
     二、标签页
     ============================================================ */
  const TABS = {};
  function initTabs() {
    document.querySelectorAll('.tab').forEach((btn) => {
      btn.addEventListener('click', () => activateTab(btn.dataset.tab));
    });
  }
  function activateTab(name) {
    document.querySelectorAll('.tab').forEach((b) => {
      const on = b.dataset.tab === name;
      b.classList.toggle('is-active', on);
      b.setAttribute('aria-selected', on ? 'true' : 'false');
    });
    document.querySelectorAll('.panel').forEach((p) => {
      const on = p.id === 'panel-' + name;
      p.classList.toggle('is-active', on);
      p.hidden = !on;
    });
    // 切回时重绘依赖真实布局的画布
    requestAnimationFrame(() => {
      if (name === 'bloch' && TABS.bloch) TABS.bloch.resize();
      if (name === 'circuit' && TABS.circuit) TABS.circuit.overlay();
      if (name === 'chsh' && TABS.chsh) TABS.chsh.redraw();
      if (name === 'grover' && TABS.grover) TABS.grover.resize();
    });
  }

  /* ============================================================
     三、布洛赫球
     ============================================================ */
  const Bloch = (() => {
    const cv = $('#bloch');
    const ctx = cv.getContext('2d');
    const B = {
      theta: Math.PI / 3,
      phi: Math.PI / 4,
      yaw: -0.62,
      pitch: 0.34,
      dragging: false,
      lx: 0, ly: 0,
      counts: [0, 0],
      particles: [],
      pulses: [],
    };
    const W = cv.width, H = cv.height;
    const cx = W / 2, cy = H / 2 + 2, R = Math.min(W, H) * 0.342;

    let cyaw, syaw, cp, sp;
    function proj(v) {
      const x = v[0], y = v[1], z = v[2];
      const x1 = x * cyaw - y * syaw;
      const y1 = x * syaw + y * cyaw;
      const y2 = y1 * cp - z * sp;
      const z2 = y1 * sp + z * cp;
      return [cx + x1 * R, cy - z2 * R, y2];
    }
    function vec() {
      const st = Math.sin(B.theta), ct = Math.cos(B.theta);
      return [st * Math.cos(B.phi), st * Math.sin(B.phi), ct];
    }

    function line3(a, b, style, width, dash) {
      const p = proj(a), q = proj(b);
      ctx.save();
      ctx.strokeStyle = style;
      ctx.lineWidth = width;
      if (dash) ctx.setLineDash(dash);
      ctx.beginPath();
      ctx.moveTo(p[0], p[1]);
      ctx.lineTo(q[0], q[1]);
      ctx.stroke();
      ctx.restore();
    }

    function wire() {
      // 外圈
      ctx.save();
      const g = ctx.createRadialGradient(cx - R * 0.35, cy - R * 0.4, R * 0.1, cx, cy, R * 1.06);
      g.addColorStop(0, 'rgba(70,130,200,.26)');
      g.addColorStop(0.55, 'rgba(30,60,110,.15)');
      g.addColorStop(1, 'rgba(8,16,32,.02)');
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(cx, cy, R, 0, TAU);
      ctx.fill();
      ctx.strokeStyle = 'rgba(130,200,255,.55)';
      ctx.lineWidth = 1.6;
      ctx.shadowColor = 'rgba(57,230,255,.35)';
      ctx.shadowBlur = 22;
      ctx.stroke();
      ctx.restore();

      // 经线
      for (let k = 0; k < 6; k++) {
        const a = (k * Math.PI) / 6;
        ctx.beginPath();
        for (let i = 0; i <= 72; i++) {
          const t = (i / 72) * Math.PI;
          const p = proj([Math.sin(t) * Math.cos(a), Math.sin(t) * Math.sin(a), Math.cos(t)]);
          i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]);
        }
        ctx.strokeStyle = 'rgba(120,180,240,.18)';
        ctx.lineWidth = 1;
        ctx.stroke();
      }
      // 纬线
      for (const t of [Math.PI / 4, Math.PI / 2, (3 * Math.PI) / 4]) {
        ctx.beginPath();
        for (let i = 0; i <= 72; i++) {
          const a = (i / 72) * TAU;
          const p = proj([Math.sin(t) * Math.cos(a), Math.sin(t) * Math.sin(a), Math.cos(t)]);
          i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]);
        }
        ctx.strokeStyle = t === Math.PI / 2 ? 'rgba(130,200,255,.45)' : 'rgba(120,180,240,.21)';
        ctx.lineWidth = t === Math.PI / 2 ? 1.4 : 1;
        ctx.stroke();
      }

      // 坐标轴
      line3([0, 0, -1.12], [0, 0, 1.12], 'rgba(150,190,235,.44)', 1.2);
      line3([0, 0, 0], [1.14, 0, 0], 'rgba(140,180,225,.28)', 1.1);
      line3([0, 0, 0], [0, 1.14, 0], 'rgba(140,180,225,.28)', 1.1);

      const pz = proj([0, 0, 1.16]);
      const pz0 = proj([0, 0, -1.16]);
      const px = proj([1.2, 0, 0]);
      const py = proj([0, 1.2, 0]);
      ctx.font = '600 20px Consolas, monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillStyle = 'rgba(190,235,255,.9)';
      ctx.fillText('|0⟩', pz[0], pz[1] - 10);
      ctx.fillStyle = 'rgba(167,139,250,.9)';
      ctx.fillText('|1⟩', pz0[0], pz0[1] + 12);
      ctx.fillStyle = 'rgba(150,180,215,.55)';
      ctx.font = '500 16px Consolas, monospace';
      ctx.fillText('x', px[0], px[1]);
      ctx.fillText('y', py[0], py[1]);
    }

    function drawArrow(v) {
      const o = proj([0, 0, 0]);
      const t = proj(v);
      const ex = t[0] - o[0], ey = t[1] - o[1];
      const len = Math.hypot(ex, ey) || 1;
      const ux = ex / len, uy = ey / len;

      ctx.save();
      ctx.shadowColor = 'rgba(57,230,255,.85)';
      ctx.shadowBlur = 22;
      const grad = ctx.createLinearGradient(o[0], o[1], t[0], t[1]);
      grad.addColorStop(0, 'rgba(57,230,255,.35)');
      grad.addColorStop(1, 'rgba(150,245,255,.98)');
      ctx.strokeStyle = grad;
      ctx.lineWidth = 4;
      ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.moveTo(o[0], o[1]);
      ctx.lineTo(t[0], t[1]);
      ctx.stroke();
      ctx.restore();

      // 箭头
      ctx.save();
      ctx.translate(t[0], t[1]);
      ctx.rotate(Math.atan2(uy, ux));
      ctx.fillStyle = '#c8f8ff';
      ctx.shadowColor = 'rgba(57,230,255,.9)';
      ctx.shadowBlur = 16;
      ctx.beginPath();
      ctx.moveTo(6, 0);
      ctx.lineTo(-9, 6);
      ctx.lineTo(-9, -6);
      ctx.closePath();
      ctx.fill();
      ctx.restore();

      // 端点光点
      ctx.save();
      ctx.shadowColor = 'rgba(57,230,255,1)';
      ctx.shadowBlur = 24;
      ctx.fillStyle = '#eafcff';
      ctx.beginPath();
      ctx.arc(t[0], t[1], 6.5, 0, TAU);
      ctx.fill();
      ctx.restore();
    }

    function drawThetaArc(v) {
      const steps = 44;
      // 在 z 轴与态矢量张成的平面内画 θ 弧
      const e = [Math.cos(B.phi), Math.sin(B.phi)];
      ctx.save();
      ctx.strokeStyle = 'rgba(57,230,255,.58)';
      ctx.lineWidth = 1.4;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      for (let i = 0; i <= steps; i++) {
        const a = (i / steps) * B.theta;
        const p = proj([Math.sin(a) * e[0] * 0.62, Math.sin(a) * e[1] * 0.62, Math.cos(a) * 0.62]);
        i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]);
      }
      ctx.stroke();
      ctx.restore();

      // 投影虚线
      if (Math.cos(B.theta) > -0.996) {
        ctx.save();
        ctx.strokeStyle = 'rgba(120,180,240,.22)';
        ctx.setLineDash([3, 5]);
        ctx.beginPath();
        const a = proj(v), b = proj([0, 0, v[2]]);
        ctx.moveTo(a[0], a[1]);
        ctx.lineTo(b[0], b[1]);
        ctx.stroke();
        ctx.restore();
      }
    }

    function drawParticles(now) {
      B.particles = B.particles.filter((p) => now - p.t0 < 900);
      for (const p of B.particles) {
        const k = clamp((now - p.t0) / 900, 0, 1);
        const ease = 1 - Math.pow(1 - k, 2.2);
        const tip = proj(vec());
        const pole = proj([0, 0, p.dir === 0 ? 1 : -1]);
        const x = tip[0] + (pole[0] - tip[0]) * ease;
        const y = tip[1] + (pole[1] - tip[1]) * ease;
        ctx.save();
        ctx.globalAlpha = (1 - k) * 0.95;
        ctx.shadowColor = p.dir === 0 ? 'rgba(57,230,255,1)' : 'rgba(167,139,250,1)';
        ctx.shadowBlur = 18;
        ctx.fillStyle = p.dir === 0 ? '#c9f7ff' : '#ded0ff';
        ctx.beginPath();
        ctx.arc(x, y, 5.5 * (1 - k * 0.45), 0, TAU);
        ctx.fill();
        ctx.restore();
      }
      B.pulses = B.pulses.filter((p) => now - p.t0 < 700);
      for (const p of B.pulses) {
        const k = clamp((now - p.t0) / 700, 0, 1);
        const pt = proj([0, 0, p.dir === 0 ? 1 : -1]);
        ctx.save();
        ctx.globalAlpha = (1 - k) * 0.7;
        ctx.strokeStyle = p.dir === 0 ? '#39e6ff' : '#a78bfa';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(pt[0], pt[1], 6 + k * 34, 0, TAU);
        ctx.stroke();
        ctx.restore();
      }
    }

    function draw(now) {
      cyaw = Math.cos(B.yaw); syaw = Math.sin(B.yaw);
      cp = Math.cos(B.pitch); sp = Math.sin(B.pitch);
      ctx.clearRect(0, 0, W, H);
      wire();
      const v = vec();
      drawThetaArc(v);
      drawArrow(v);
      drawParticles(now || 0);
    }

    function readout() {
      const ct = Math.cos(B.theta / 2), st = Math.sin(B.theta / 2);
      const p0 = ct * ct, p1 = st * st;
      const betaRe = st * Math.cos(B.phi), betaIm = st * Math.sin(B.phi);
      $('#p0').textContent = p0.toFixed(3);
      $('#p1').textContent = p1.toFixed(3);
      $('#vec').textContent =
        `(${fmt(Math.sin(B.theta) * Math.cos(B.phi))}, ${fmt(Math.sin(B.theta) * Math.sin(B.phi))}, ${fmt(Math.cos(B.theta))})`;
      $('#amps').textContent =
        `α = ${ct.toFixed(3)}, β = ${betaRe.toFixed(3)} ${betaIm >= 0 ? '+' : '−'} ${Math.abs(betaIm).toFixed(3)}i`;
      $('#theta-out').textContent = ((B.theta * 180) / Math.PI).toFixed(1) + '°';
      $('#phi-out').textContent = Math.round((B.phi * 180) / Math.PI) + '°';
    }

    function histo() {
      const total = B.counts[0] + B.counts[1];
      const max = Math.max(1, B.counts[0], B.counts[1]);
      const h0 = total ? (B.counts[0] / max) * 92 : 2;
      const h1 = total ? (B.counts[1] / max) * 92 : 2;
      $('#h0').style.height = h0 + 'px';
      $('#h1').style.height = h1 + 'px';
      $('#h0n').textContent = B.counts[0];
      $('#h1n').textContent = B.counts[1];
    }

    function measure() {
      const p0 = Math.cos(B.theta / 2) ** 2;
      const out = Math.random() < p0 ? 0 : 1;
      B.counts[out]++;
      const now = performance.now();
      B.particles.push({ dir: out, t0: now });
      B.pulses.push({ dir: out, t0: now + 120 });
      histo();
      const el = $('#measure-last');
      el.textContent = `本次结果 ${out} · 共 ${B.counts[0] + B.counts[1]} 次`;
      el.style.color = out === 0 ? '#8ff0ff' : '#d8c6ff';
      return out;
    }

    function setState(thetaDeg, phiDeg) {
      B.theta = (thetaDeg * Math.PI) / 180;
      B.phi = (phiDeg * Math.PI) / 180;
      $('#theta').value = thetaDeg;
      $('#phi').value = phiDeg;
      readout();
    }

    function bind() {
      $('#theta').addEventListener('input', (e) => {
        B.theta = (+e.target.value * Math.PI) / 180;
        readout();
      });
      $('#phi').addEventListener('input', (e) => {
        B.phi = (+e.target.value * Math.PI) / 180;
        readout();
      });
      $$('.chip[data-state]').forEach((b) => {
        b.addEventListener('click', () => {
          const map = {
            '0': [0, 0], '1': [180, 0], plus: [90, 0],
            minus: [90, 180], i: [90, 90],
          };
          const [t, p] = map[b.dataset.state];
          setState(t, p);
        });
      });
      $('#measure-btn').addEventListener('click', measure);
      $('#measure-reset').addEventListener('click', () => {
        B.counts = [0, 0];
        histo();
        $('#measure-last').textContent = '—';
        $('#measure-last').style.color = '';
      });

      cv.addEventListener('pointerdown', (e) => {
        B.dragging = true;
        B.lx = e.clientX; B.ly = e.clientY;
        cv.setPointerCapture(e.pointerId);
      });
      cv.addEventListener('pointermove', (e) => {
        if (!B.dragging) return;
        B.yaw += (e.clientX - B.lx) * 0.0075;
        B.pitch = clamp(B.pitch + (e.clientY - B.ly) * 0.0055, -1.15, 1.15);
        B.lx = e.clientX; B.ly = e.clientY;
      });
      const stop = () => { B.dragging = false; };
      cv.addEventListener('pointerup', stop);
      cv.addEventListener('pointercancel', stop);
      cv.addEventListener('pointerleave', stop);
    }

    let last = 0;
    function loop(ts) {
      const dt = Math.min(64, ts - last || 16);
      last = ts;
      if (!B.dragging) B.yaw += dt * 0.00016;
      draw(ts);
      requestAnimationFrame(loop);
    }

    function init() {
      bind();
      readout();
      histo();
      requestAnimationFrame(loop);
    }
    return { init, resize() {}, measure, setState, state: B };
  })();

  /* ============================================================
     四、量子电路
     ============================================================ */
  const Circuit = (() => {
    const DEPTH = 8;
    const N = 3;
    const CIRCUIT = { n: N, cols: [] };
    let selected = null;      // 当前选中的门
    let pending = null;       // CNOT 控制点 {col,row}
    let history = [];
    let lastProbs = null;

    function reset() {
      CIRCUIT.cols = Array.from({ length: DEPTH }, () => []);
    }
    function snapshot() {
      return CIRCUIT.cols.map((c) => c.map((o) => ({ ...o })));
    }
    function pushHistory() {
      history.push(snapshot());
      if (history.length > 60) history.shift();
    }
    function opAt(col, row) {
      for (const op of CIRCUIT.cols[col]) {
        if (op.g === 'CX') {
          if (op.c === row) return { op, role: 'c' };
          if (op.t === row) return { op, role: 't' };
        } else if (op.q === row) {
          return { op, role: 's' };
        }
      }
      return null;
    }
    function gateText(op) {
      if (op.g === 'CX') return '●';
      if (op.g[0] === 'R') return 'R' + op.g[1].toLowerCase();
      return op.g;
    }
    function gateInner(op, role) {
      if (op.g === 'CX') {
        return `<span class="gate g-CX">${role === 't' ? '⊕' : '●'}</span>`;
      }
      const deg = op.g[0] === 'R' ? `<i>${op.deg}°</i>` : '';
      return `<span class="gate g-${op.g}" title="点击删除">${gateText(op)}${deg}</span>`;
    }

    function render() {
      const body = $('#circuit-body');
      body.innerHTML = '';

      const head = document.createElement('div');
      head.className = 'c-row c-head';
      head.innerHTML = `<span class="qh">列</span><div class="cells">` +
        Array.from({ length: DEPTH }, (_, i) => `<span>${i + 1}</span>`).join('') + `</div>`;
      body.appendChild(head);

      for (let r = 0; r < N; r++) {
        const row = document.createElement('div');
        row.className = 'c-row';
        const lab = document.createElement('div');
        lab.className = 'q-label';
        lab.innerHTML = `<span class="q-name">q${r}</span><span class="q-ket">|0⟩</span>`;
        row.appendChild(lab);
        const cells = document.createElement('div');
        cells.className = 'cells';
        for (let c = 0; c < DEPTH; c++) {
          const cell = document.createElement('button');
          cell.type = 'button';
          cell.className = 'cell';
          cell.dataset.col = c;
          cell.dataset.row = r;
          cell.setAttribute('aria-label', `第 ${c + 1} 列 第 ${r + 1} 行`);
          const hit = opAt(c, r);
          if (hit) {
            cell.classList.add('filled');
            cell.innerHTML = gateInner(hit.op, hit.role);
          }
          if (pending && pending.col === c && pending.row === r) cell.classList.add('pending');
          cells.appendChild(cell);
        }
        row.appendChild(cells);
        body.appendChild(row);
      }
      const depth = CIRCUIT.cols.filter((c) => c.length).length;
      $('#depth-tag').textContent = depth + ' 层';
      requestAnimationFrame(drawOverlay);
    }

    function drawOverlay() {
      const wrap = $('#circuit-wrap');
      const svg = $('#cx-overlay');
      if (!wrap || !svg) return;
      const wr = wrap.getBoundingClientRect();
      if (!wr.width) return;
      svg.setAttribute('viewBox', `0 0 ${wr.width} ${wr.height}`);
      svg.setAttribute('width', wr.width);
      svg.setAttribute('height', wr.height);
      let out = '';
      for (let c = 0; c < DEPTH; c++) {
        for (const op of CIRCUIT.cols[c]) {
          if (op.g !== 'CX') continue;
          const a = document.querySelector(`.cell[data-col="${c}"][data-row="${op.c}"]`);
          const b = document.querySelector(`.cell[data-col="${c}"][data-row="${op.t}"]`);
          if (!a || !b) continue;
          const ar = a.getBoundingClientRect(), br = b.getBoundingClientRect();
          const x = ar.left - wr.left + ar.width / 2;
          const y1 = ar.top - wr.top + ar.height / 2;
          const y2 = br.top - wr.top + br.height / 2;
          out += `<line x1="${x}" y1="${y1}" x2="${x}" y2="${y2}" stroke="rgba(150,215,255,.5)" stroke-width="2" stroke-linecap="round"/>`;
        }
      }
      svg.innerHTML = out;
    }

    function hint(msg, warn) {
      const el = $('#circuit-hint');
      el.textContent = msg;
      el.classList.toggle('warn', !!warn);
    }

    function clearColumnCell(col, row) {
      CIRCUIT.cols[col] = CIRCUIT.cols[col].filter((op) => {
        if (op.g === 'CX') return op.c !== row && op.t !== row;
        return op.q !== row;
      });
    }

    function hit({ col, row }) {
      const hitOp = opAt(col, row);

      if (selected === 'CX') {
        if (hitOp && hitOp.op.g === 'CX') {
          pushHistory();
          clearColumnCell(col, row);
          pending = null;
          hint('已删除 CNOT。');
          update();
          return;
        }
        if (pending && pending.col === col) {
          if (pending.row === row) {
            pending = null;
            hint('控制点已取消。');
            update();
            return;
          }
          const ctrl = pending.row;
          pushHistory();
          clearColumnCell(col, ctrl);
          clearColumnCell(col, row);
          CIRCUIT.cols[col].push({ g: 'CX', c: ctrl, t: row });
          pending = null;
          hint(`已放置 CNOT：控制 q${ctrl}，目标 q${row}。`);
          update();
          return;
        }
        pending = { col, row };
        hint(`控制点已选 q${row}（第 ${col + 1} 列）→ 再点同列另一行作为目标。`, true);
        update();
        return;
      }

      if (!selected) {
        if (hitOp) {
          pushHistory();
          clearColumnCell(col, row);
          hint('已删除该门。');
          update();
        }
        return;
      }

      // 放置单比特门
      pushHistory();
      if (hitOp && hitOp.op.g[0] === 'R' && hitOp.op.g === selected) {
        // 已存在同类型旋转门：更新角度
        hitOp.op.deg = currentAngle();
        hint(`已更新 ${selected} 角度为 ${currentAngle()}°。`);
      } else {
        clearColumnCell(col, row);
        CIRCUIT.cols[col].push({
          g: selected,
          q: row,
          ...(selected[0] === 'R' ? { deg: currentAngle() } : {}),
        });
        hint(`已放置 ${selected} 门于 q${row}，第 ${col + 1} 列。`);
      }
      update();
    }

    function currentAngle() { return +$('#gate-angle').value; }

    function update() {
      render();
      renderState();
      // 电路变化后自动重采样，保证面板始终有数据
      lastProbs = null;
      requestAnimationFrame(() => runShots());
    }

    function renderState() {
      const st = simulate(CIRCUIT);
      const probs = st.probs();

      const bars = $('#pbars');
      bars.innerHTML = probs.map((p, i) => {
        const pct = p * 100;
        return `<div class="pbar ${p < 1e-6 ? 'zero' : ''}">
          <span class="k">${st.ket(i)}</span>
          <div class="track"><div class="fill" style="width:0%"></div></div>
          <span class="v">${pct >= 0.05 ? pct.toFixed(1) + '%' : '—'}</span>
        </div>`;
      }).join('');
      // 下一帧再赋宽度，触发过渡动画
      requestAnimationFrame(() => {
        $$('#pbars .fill').forEach((f, i) => { f.style.width = (probs[i] * 100).toFixed(2) + '%'; });
      });

      $('#norm-tag').textContent = '‖ψ‖ = ' + st.norm().toFixed(3);

      const amps = $('#amps-table');
      amps.innerHTML = `<div class="amp-head"><span>态</span><span>复振幅</span><span>相位</span></div>` +
        probs.map((p, i) => {
          const a = Math.atan2(st.im[i], st.re[i]);
          const deg = Math.round(((a * 180) / Math.PI + 360) % 360);
          const zero = p < 1e-9;
          return `<div class="amp-row ${zero ? 'zero' : ''}">
            <span>${st.ket(i)}</span>
            <b>${signed(st.re[i], 4)} ${signed(st.im[i], 4)}i</b>
            <span class="amp-ph">${zero ? '—' : deg + '°'}</span>
          </div>`;
        }).join('');
    }

    function runShots() {
      const st = simulate(CIRCUIT);
      const probs = st.probs();
      const shots = +$('#shots').value;
      const counts = sampleCounts(probs, shots);
      const max = Math.max(...counts);
      const barHtml = counts.map((n, i) => {
        const pct = (n / shots) * 100;
        return `<div class="pbar compact ${n === 0 ? 'zero' : ''}">
          <span class="k">${st.ket(i)}</span>
          <div class="track"><div class="fill" style="width:${max ? (n / max) * 100 : 0}%"></div></div>
          <span class="v">${n}</span>
        </div>`;
      }).join('');
      $('#shot-bars').innerHTML = barHtml;
      const top = counts.indexOf(max);
      $('#shot-note').textContent =
        `共 ${shots} 次：最常出现 ${st.ket(top)}（${counts[top]} 次，${((counts[top] / shots) * 100).toFixed(1)}%）。` +
        `理论值 ${(probs[top] * 100).toFixed(1)}%。`;
      lastProbs = probs;
    }

    function loadPreset(name) {
      pushHistory();
      reset();
      const add = (col, op) => CIRCUIT.cols[col].push(op);
      if (name === 'bell') {
        add(0, { g: 'H', q: 0 });
        add(1, { g: 'CX', c: 0, t: 1 });
      } else if (name === 'ghz') {
        add(0, { g: 'H', q: 0 });
        add(1, { g: 'CX', c: 0, t: 1 });
        add(2, { g: 'CX', c: 1, t: 2 });
      } else if (name === 'uniform') {
        add(0, { g: 'H', q: 0 });
        add(0, { g: 'H', q: 1 });
        add(0, { g: 'H', q: 2 });
      } else if (name === 'interference') {
        add(0, { g: 'H', q: 0 });
        add(1, { g: 'Z', q: 0 });
        add(2, { g: 'H', q: 0 });
      }
      pending = null;
      update();
      const notes = {
        bell: '贝尔态：两个比特只有 |000⟩ 与 |110⟩，测量结果完全相关。',
        ghz: 'GHZ 态：三个比特同生共死，是全有或全无的叠加。',
        uniform: '三个 H 门：8 个基态等幅叠加，测量完全随机。',
        interference: 'H→Z→H 等价于 X：中间插入的相位在第二次 H 上发生干涉，最终表现为一次确定的比特翻转。',
      };
      hint(notes[name] || '');
    }

    function bind() {
      $$('#gate-palette .gate-btn').forEach((b) => {
        b.addEventListener('click', () => {
          const g = b.dataset.gate;
          if (selected === g) {
            selected = null;
            b.classList.remove('is-active');
            hint('已取消选择。');
          } else {
            selected = g;
            $$('#gate-palette .gate-btn').forEach((x) => x.classList.toggle('is-active', x === b));
            hint(g === 'CX'
              ? 'CNOT：先点控制点所在格子，再点同列的目标行。'
              : `已选择 ${g}，点击电路格子放置；点击已有门可删除。`);
          }
          pending = null;
          render();
        });
      });
      $('#gate-angle').addEventListener('input', (e) => {
        $('#gate-angle-out').textContent = e.target.value + '°';
      });
      $('#circuit-body').addEventListener('click', (e) => {
        const cell = e.target.closest('.cell');
        if (!cell) return;
        hit({ col: +cell.dataset.col, row: +cell.dataset.row });
      });
      $('#undo').addEventListener('click', () => {
        if (!history.length) { hint('没有可撤销的操作。', true); return; }
        CIRCUIT.cols = history.pop();
        pending = null;
        update();
        hint('已撤销。');
      });
      $('#clear').addEventListener('click', () => {
        pushHistory();
        reset();
        pending = null;
        update();
        hint('已清空电路，态回到 |000⟩。');
      });
      $$('.chip[data-preset]').forEach((b) => {
        b.addEventListener('click', () => loadPreset(b.dataset.preset));
      });
      $('#run-shots').addEventListener('click', runShots);
      if (window.ResizeObserver) {
        new ResizeObserver(() => drawOverlay()).observe($('#circuit-wrap'));
      }
    }

    function init() {
      reset();
      loadPreset('bell');
      bind();
      history = [];
      renderState();
    }

    return { init, overlay: drawOverlay, update, loadPreset, circuit: CIRCUIT, simulate, runShots };
  })();

  /* ============================================================
     五、CHSH / 贝尔不等式
     ============================================================ */
  const CHSH = (() => {
    const CH = { a: 0, ap: 90, b: 45, bp: 135, mode: 'quantum', S: null, E: null };

    // |Φ⁺⟩ 的联合分布：P(+,+)=P(−,−)=(1+cosΔ)/4
    function quantumCounts(alphaDeg, betaDeg, shots) {
      const d = ((alphaDeg - betaDeg) * Math.PI) / 180;
      const c = Math.cos(d);
      const pSame = (1 + c) / 4;  // P(+,+) 与 P(−,−)
      const pDiff = (1 - c) / 4;  // P(+,−) 与 P(−,+)
      const counts = [0, 0, 0, 0]; // ++, +−, −+, −−
      for (let i = 0; i < shots; i++) {
        const r = Math.random();
        if (r < pSame) counts[0]++;
        else if (r < pSame + pDiff) counts[1]++;
        else if (r < pSame + pDiff + pDiff) counts[2]++;
        else counts[3]++;
      }
      return counts;
    }

    function classicalStrategy() {
      const A = [CH.a, CH.ap], B = [CH.b, CH.bp];
      let best = null;
      for (let mask = 0; mask < 16; mask++) {
        const t = [];
        for (let i = 0; i < 4; i++) t.push(mask & (1 << i) ? 1 : -1);
        const S = t[0] * t[2] - t[0] * t[3] + t[1] * t[2] + t[1] * t[3];
        if (!best || S > best.S) best = { S, t };
      }
      return best;
    }

    function run() {
      const shots = 20000;
      const pairs = [[CH.a, CH.b], [CH.a, CH.bp], [CH.ap, CH.b], [CH.ap, CH.bp]];
      const Es = [], countsAll = [];

      if (CH.mode === 'classical') {
        const best = classicalStrategy();
        const t = best.t;
        // t = [A(a), A(a'), B(b), B(b')]
        const tbl = [
          [t[0], t[2]], [t[0], t[3]], [t[1], t[2]], [t[1], t[3]],
        ];
        for (const [a, b] of tbl) {
          const counts = [0, 0, 0, 0];
          counts[a > 0 && b > 0 ? 0 : a > 0 && b < 0 ? 1 : a < 0 && b > 0 ? 2 : 3] = shots;
          countsAll.push(counts);
          Es.push((counts[0] + counts[3] - counts[1] - counts[2]) / shots);
        }
      } else {
        for (const [al, be] of pairs) {
          const counts = quantumCounts(al, be, shots);
          countsAll.push(counts);
          const n = counts[0] + counts[1] + counts[2] + counts[3];
          Es.push((counts[0] + counts[3] - counts[1] - counts[2]) / n);
        }
      }

      const S = Es[0] - Es[1] + Es[2] + Es[3];
      const err = Math.sqrt(Es.reduce((acc, e) => acc + (1 - e * e) / shots, 0));
      CH.S = S; CH.E = Es; CH.err = err;

      // 输出四组 E
      const labels = ['E(a,b)', 'E(a,b′)', 'E(a′,b)', 'E(a′,b′)'];
      $('#chsh-out').innerHTML = Es.map((v, i) => `
        <div class="e-cell">
          <span class="k">${labels[i]}</span>
          <b class="${v > 0.02 ? 'pos' : v < -0.02 ? 'neg' : ''}">${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(3)}</b>
        </div>`).join('') + `
        <div class="e-cell" style="grid-column:1/-1">
          <span class="k">每组合计</span><b>${shots} 次符合</b>
        </div>`;

      $('#s-value').textContent = Math.abs(S).toFixed(3);
      $('#s-err').textContent = err > 1e-6 ? `± ${err.toFixed(3)}` : '';
      const pct = clamp((Math.abs(S) / 4) * 100, 0, 100);
      $('#gauge-marker').style.left = pct + '%';
      $('#chsh-mode').textContent = CH.mode === 'classical' ? '经典策略' : '量子';

      // 第一组设置（a,b）的联合结果分布
      const c0 = countsAll[0];
      const tot0 = c0.reduce((x, y) => x + y, 0) || 1;
      const names = ['(+,+)', '(+,−)', '(−,+)', '(−,−)'];
      $('#joint-title').textContent = `联合结果分布 · a=${CH.a}° / b=${CH.b}° · ${tot0} 次`;
      $('#joint-bars').innerHTML = c0.map((n, i) => `
        <div class="jb">
          <span class="k">P${names[i]}</span>
          <div class="track"><i style="width:${((n / tot0) * 100).toFixed(1)}%"></i></div>
          <b>${((n / tot0) * 100).toFixed(1)}%</b>
        </div>`).join('');

      const verdict = $('#chsh-verdict');
      const sTxt = Math.abs(S).toFixed(3) + (err > 1e-6 ? ` ± ${err.toFixed(3)}` : '');
      if (Math.abs(S) > 2.02) {
        verdict.innerHTML = `|S| = ${sTxt} &gt; 2：<b style="color:#ffd98a">局域隐变量模型被排除</b>。` +
          `这种关联无法用“提前约好答案”的经典策略复现。` +
          (Math.abs(S) > 2.7 ? '最优角度下的理论值正是 Tsirelson 界 2√2 ≈ 2.828。' : '');
      } else if (Math.abs(S) > 1.98) {
        verdict.innerHTML = `|S| = ${sTxt} ≈ 2：正好落在经典上限上。${CH.mode === 'classical' ? '确定性经典策略可以做到 2，但一步也超不过去。' : ''}`;
      } else {
        verdict.innerHTML = `|S| = ${sTxt} &lt; 2：这组设置下经典与量子的差别不明显，把角度调回“最优角度”再看。`;
      }
      drawCorr();
    }

    function drawCorr() {
      const cv = $('#corr');
      const ctx = cv.getContext('2d');
      const W = cv.width, H = cv.height;
      const pad = { l: 52, r: 18, t: 20, b: 34 };
      const x0 = pad.l, x1 = W - pad.r, y0 = pad.t, y1 = H - pad.b;
      ctx.clearRect(0, 0, W, H);

      const X = (deg) => x0 + (deg / 360) * (x1 - x0);
      const Y = (v) => y0 + ((1 - v) / 2) * (y1 - y0);

      // 网格
      ctx.font = '12px Consolas, monospace';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      for (const v of [-1, -0.5, 0, 0.5, 1]) {
        ctx.strokeStyle = v === 0 ? 'rgba(130,180,235,.28)' : 'rgba(130,180,235,.10)';
        ctx.beginPath();
        ctx.moveTo(x0, Y(v));
        ctx.lineTo(x1, Y(v));
        ctx.stroke();
        ctx.fillStyle = 'rgba(140,170,205,.65)';
        ctx.fillText(v.toFixed(1), x0 - 10, Y(v));
      }
      ctx.textAlign = 'center';
      for (const d of [0, 90, 180, 270, 360]) {
        ctx.strokeStyle = 'rgba(130,180,235,.08)';
        ctx.beginPath();
        ctx.moveTo(X(d), y0);
        ctx.lineTo(X(d), y1);
        ctx.stroke();
        ctx.fillStyle = 'rgba(140,170,205,.55)';
        ctx.fillText(d + '°', X(d), y1 + 18);
      }
      // 曲线 E = cos Δ
      ctx.beginPath();
      for (let d = 0; d <= 360; d += 2) {
        const v = Math.cos((d * Math.PI) / 180);
        d ? ctx.lineTo(X(d), Y(v)) : ctx.moveTo(X(d), Y(v));
      }
      const lg = ctx.createLinearGradient(x0, 0, x1, 0);
      lg.addColorStop(0, 'rgba(57,230,255,.9)');
      lg.addColorStop(1, 'rgba(167,139,250,.9)');
      ctx.strokeStyle = lg;
      ctx.lineWidth = 2.4;
      ctx.shadowColor = 'rgba(57,230,255,.45)';
      ctx.shadowBlur = 12;
      ctx.stroke();
      ctx.shadowBlur = 0;

      // 实测点
      if (CH.E) {
        const pairs = [[CH.a, CH.b], [CH.a, CH.bp], [CH.ap, CH.b], [CH.ap, CH.bp]];
        const colors = ['#39e6ff', '#a78bfa', '#ffc857', '#ff7ab6'];
        pairs.forEach(([a, b], i) => {
          let d = ((a - b) % 360 + 360) % 360;
          const e = CH.E[i];
          ctx.beginPath();
          ctx.arc(X(d), Y(clamp(e, -1, 1)), 6, 0, TAU);
          ctx.fillStyle = colors[i];
          ctx.shadowColor = colors[i];
          ctx.shadowBlur = 14;
          ctx.fill();
          ctx.shadowBlur = 0;
          ctx.fillStyle = 'rgba(220,240,255,.8)';
          ctx.font = '11px Consolas, monospace';
          ctx.textAlign = 'left';
          ctx.fillText('E' + (i + 1), X(d) + 9, Y(e) - 9);
        });
      }
      ctx.fillStyle = 'rgba(140,170,205,.6)';
      ctx.font = '12px Consolas, monospace';
      ctx.textAlign = 'right';
      ctx.fillText('Δ = α − β', x1 - 2, y1 - 8);
    }

    function sync() {
      $('#ang-a').value = CH.a; $('#ang-ap').value = CH.ap;
      $('#ang-b').value = CH.b; $('#ang-bp').value = CH.bp;
      $('#ang-a-out').textContent = CH.a + '°';
      $('#ang-ap-out').textContent = CH.ap + '°';
      $('#ang-b-out').textContent = CH.b + '°';
      $('#ang-bp-out').textContent = CH.bp + '°';
    }

    function bind() {
      const map = { 'ang-a': 'a', 'ang-ap': 'ap', 'ang-b': 'b', 'ang-bp': 'bp' };
      Object.keys(map).forEach((id) => {
        $('#' + id).addEventListener('input', (e) => {
          CH[map[id]] = +e.target.value;
          $('#' + id + '-out').textContent = e.target.value + '°';
          CH.S = null; CH.E = null;
          $('#chsh-verdict').textContent = '角度已改变，重新运行测量。';
        });
      });
      $$('.chip[data-chsh]').forEach((b) => {
        b.addEventListener('click', () => {
          const mode = b.dataset.chsh;
          if (mode === 'optimal') { CH.a = 0; CH.ap = 90; CH.b = 45; CH.bp = 135; CH.mode = 'quantum'; }
          else if (mode === 'aligned') { CH.a = 0; CH.ap = 0; CH.b = 0; CH.bp = 0; CH.mode = 'quantum'; }
          else { CH.mode = 'classical'; }
          sync();
          run();
        });
      });
      $('#chsh-run').addEventListener('click', run);
    }

    function init() {
      bind();
      sync();
      drawCorr();
      run();
    }
    return { init, redraw: drawCorr, run, state: CH };
  })();

  /* ============================================================
     六、Grover
     ============================================================ */
  const Grover = (() => {
    const N = 8;
    let marked = 5;           // |101⟩
    let stages = [];
    let idx = 0;
    let timer = null;

    function build(mark) {
      const st = new QState(3);
      for (let q = 0; q < 3; q++) st.apply(M.H, q);
      const out = [{ label: '初始化：H⊗H⊗H，8 个候选等幅叠加', re: st.re.slice(), im: st.im.slice() }];
      for (let it = 1; it <= 3; it++) {
        // oracle：标记项相位翻转
        const o = st.clone();
        o.re[mark] *= -1; o.im[mark] *= -1;
        out.push({ label: `第 ${it} 次 · Oracle：翻转 |${mark.toString(2).padStart(3, '0')}⟩ 的相位`, re: o.re.slice(), im: o.im.slice() });
        // diffusion：关于平均值反演
        let mr = 0, mi = 0;
        for (let i = 0; i < N; i++) { mr += o.re[i]; mi += o.im[i]; }
        mr = (2 * mr) / N; mi = (2 * mi) / N;
        for (let i = 0; i < N; i++) { o.re[i] = mr - o.re[i]; o.im[i] = mi - o.im[i]; }
        out.push({ label: `第 ${it} 次 · 扩散：关于平均振幅反演`, re: o.re.slice(), im: o.im.slice() });
        st.re.set(o.re); st.im.set(o.im);
      }
      return out;
    }

    function probOf(s, i) { return s.re[i] * s.re[i] + s.im[i] * s.im[i]; }

    function render() {
      const s = stages[idx];
      const bars = $('#g-bars');
      const hAvail = Math.max(120, bars.clientHeight - 34);
      const maxAmp = 1;
      bars.innerHTML = Array.from({ length: N }, (_, i) => {
        const a = Math.hypot(s.re[i], s.im[i]);
        const h = Math.max(2, (a / maxAmp) * hAvail);
        const p = probOf(s, i);
        return `<div class="gb ${s.re[i] < 0 ? 'neg' : ''} ${i === marked ? 'marked' : ''}">
          <div class="gbar" style="height:${h.toFixed(1)}px" title="|${i.toString(2).padStart(3, '0')}⟩ 概率 ${(p * 100).toFixed(1)}%"></div>
          <span class="gl">${i.toString(2).padStart(3, '0')}</span>
        </div>`;
      }).join('');

      $('#g-iter').textContent = `阶段 ${idx} / ${stages.length - 1}`;
      const pM = probOf(stages[idx], marked);
      $('#g-stage').textContent =
        (idx === 0 ? '' : `当前 P(|${marked.toString(2).padStart(3, '0')}⟩) = ${(pM * 100).toFixed(2)}%　·　`) + s.label;

      // 日志：已执行的步骤显示数值，未执行的置灰
      const log = $('#g-log');
      log.innerHTML = '';
      for (let i = 0; i < stages.length; i++) {
        const p = probOf(stages[i], marked);
        const li = document.createElement('li');
        if (i === idx) li.className = 'hi';
        else if (i > idx) li.className = 'todo';
        li.innerHTML = `<span class="n">${i}</span><span>${stages[i].label}</span><b>${i > idx ? '—' : (p * 100).toFixed(1) + '%'}</b>`;
        log.appendChild(li);
      }
      log.scrollTop = log.scrollHeight;

      // 效率对比固定用 1、2 次迭代的真实概率
      const p1 = probOf(stages[2], marked) * 100;
      const p2 = probOf(stages[4], marked) * 100;
      $('#cmp-1').style.width = p1 + '%';
      $('#cmp-2').style.width = p2 + '%';
      $('#cmp-1v').textContent = p1.toFixed(1) + '%';
      $('#cmp-2v').textContent = p2.toFixed(1) + '%';

      $('#g-step').textContent = idx >= stages.length - 1 ? '已到终点 ▸' : '下一步 ▸';
      $('#g-step').disabled = idx >= stages.length - 1;
      $('#g-step').style.opacity = idx >= stages.length - 1 ? 0.5 : 1;
    }

    function next() {
      if (idx < stages.length - 1) { idx++; render(); }
    }
    function reset() {
      clearInterval(timer); timer = null;
      $('#g-auto').textContent = '自动播放';
      idx = 0;
      render();
    }
    function pick(m) {
      clearInterval(timer); timer = null;
      $('#g-auto').textContent = '自动播放';
      marked = m;
      stages = build(marked);
      idx = 0;
      $$('#marked-pick button').forEach((b) => b.classList.toggle('is-active', +b.dataset.m === marked));
      render();
    }
    function auto() {
      if (timer) {
        clearInterval(timer); timer = null;
        $('#g-auto').textContent = '自动播放';
        return;
      }
      if (idx >= stages.length - 1) idx = 0;
      $('#g-auto').textContent = '暂停';
      timer = setInterval(() => {
        if (idx >= stages.length - 1) {
          clearInterval(timer); timer = null;
          $('#g-auto').textContent = '自动播放';
        } else { idx++; render(); }
      }, 950);
    }

    function bind() {
      $('#marked-pick').innerHTML = Array.from({ length: N }, (_, i) =>
        `<button data-m="${i}" class="${i === marked ? 'is-active' : ''}">${i.toString(2).padStart(3, '0')}</button>`).join('');
      $('#marked-pick').addEventListener('click', (e) => {
        const b = e.target.closest('button');
        if (b) pick(+b.dataset.m);
      });
      $('#g-step').addEventListener('click', next);
      $('#g-reset').addEventListener('click', reset);
      $('#g-auto').addEventListener('click', auto);
    }

    function init() {
      bind();
      stages = build(marked);
      idx = 0;
      render();
    }
    return { init, resize: render, probs: (m, it) => {
      const s = build(m)[it * 2];
      return Array.from({ length: N }, (_, i) => probOf(s, i));
    } };
  })();

  /* ============================================================
     七、启动 + 测试接口
     ============================================================ */
  function boot() {
    initTabs();
    Bloch.init();
    Circuit.init();
    CHSH.init();
    Grover.init();

    TABS.bloch = Bloch;
    TABS.circuit = Circuit;
    TABS.chsh = CHSH;
    TABS.grover = Grover;

    let raf = 0;
    window.addEventListener('resize', () => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => {
        TABS.circuit.overlay();
        TABS.grover.resize();
        TABS.chsh.redraw();
      });
    });

    window.__lab = {
      version: '1.0',
      QState,
      simulate,
      M,
      circuit: Circuit,
      chsh: CHSH,
      grover: Grover,
      bloch: Bloch,
      activateTab,
      madeState(n) { return new QState(n); },
    };
    window.__labReady = true;
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
})();
