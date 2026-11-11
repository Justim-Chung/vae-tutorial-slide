/* Interactive teaching demos for the VAE slides.
 * Each <div class="demo" data-demo="..."> is replaced by a canvas + controls.
 * Buttons with data-action inside the same <section> are bound to that demo.
 */
(function () {
  "use strict";

  // ---------- helpers ----------
  const W = 800, H = 450;                    // logical canvas size
  const TEAL = "94,234,212", ORANGE = "251,146,60", PURPLE = "167,139,250";
  const GRAY = "100,116,139";

  function mulberry32(seed) {
    return function () {
      seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gaussSampler(rng) {
    return function () {
      const u = Math.max(rng(), 1e-9), v = rng();
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    };
  }
  const lerp = (a, b, t) => a + (b - a) * t;
  const easeInOut = (t) => (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2);
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

  function setupCanvas(container) {
    const canvas = document.createElement("canvas");
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    canvas.setAttribute("role", "img");
    container.appendChild(canvas);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { canvas, ctx };
  }

  function makeSlider(parent, label, min, max, step, value, onInput) {
    const wrap = document.createElement("label");
    const span = document.createElement("span");
    span.textContent = label;
    const input = document.createElement("input");
    input.type = "range";
    input.min = min; input.max = max; input.step = step; input.value = value;
    const out = document.createElement("span");
    out.style.minWidth = "3.2em";
    out.style.textAlign = "right";
    const show = () => { out.textContent = Number(input.value).toFixed(2); };
    input.addEventListener("input", () => { show(); onInput(Number(input.value)); });
    show();
    wrap.append(span, input, out);
    parent.appendChild(wrap);
    return input;
  }

  // ---------- Demo 1: AE vs VAE latent space ----------
  function latentDemo(container, section) {
    const { canvas, ctx } = setupCanvas(container);
    const readout = document.createElement("div");
    readout.className = "readout";
    container.appendChild(readout);

    const rng = mulberry32(7), g = gaussSampler(rng);
    const centers = [[-0.55, 0.3], [0.55, 0.35], [-0.45, -0.35], [0.5, -0.3]];
    const ae = [], vae = [];
    for (const [cx, cy] of centers) {
      for (let i = 0; i < 40; i++) {
        const x = cx + g() * 0.07, y = cy + g() * 0.07;
        ae.push([x, y]);
        vae.push([x * 0.62, y * 0.62]);     // VAE 把各群拉向原點、彼此重疊
      }
    }
    const N = ae.length;
    const KERNEL_AE = 0.07, KERNEL_VAE = 0.25, THR = 0.3;

    // 世界座標 → 畫布座標
    const toPx = (x, y) => [W / 2 + x * 340, H / 2 - y * 340];
    const toWorld = (px, py) => [(px - W / 2) / 340, (H / 2 - py) / 340];

    // 每個格子的 f(x)：此處被訓練雲覆蓋的程度（見 densityAt）
    const COLS = 80, ROWS = 45;
    const off = document.createElement("canvas");
    off.width = COLS; off.height = ROWS;
    const offCtx = off.getContext("2d");
    const img = offCtx.createImageData(COLS, ROWS);

    // 覆蓋度：此處離最近的訓練雲有多近（0~1）
    function densityAt(x, y, pts, s) {
      let best = 0;
      const inv = 1 / (2 * s * s);
      for (let i = 0; i < pts.length; i++) {
        const dx = x - pts[i][0], dy = y - pts[i][1];
        const k = Math.exp(-(dx * dx + dy * dy) * inv);
        if (k > best) best = k;
      }
      return best;
    }

    let t = 0, from = 0, to = 0, startT = 0, transitioning = false;
    let marker = null, fieldDirty = true, lastT = -1;
    const DUR = 1100;

    function currentPts() {
      return ae.map((p, i) => [lerp(p[0], vae[i][0], t), lerp(p[1], vae[i][1], t)]);
    }
    function currentKernel() { return lerp(KERNEL_AE, KERNEL_VAE, t); }

    function drawField(pts, s) {
      const data = img.data;
      for (let r = 0; r < ROWS; r++) {
        for (let c = 0; c < COLS; c++) {
          const px = (c + 0.5) * (W / COLS), py = (r + 0.5) * (H / ROWS);
          const [x, y] = toWorld(px, py);
          const f = densityAt(x, y, pts, s);
          const k = (r * COLS + c) * 4;
          data[k] = 94; data[k + 1] = 234; data[k + 2] = 212;
          data[k + 3] = Math.round(clamp(f, 0, 1) * 150);
        }
      }
      offCtx.putImageData(img, 0, 0);
    }

    function render(now) {
      if (transitioning) {
        const k = clamp((now - startT) / DUR, 0, 1);
        t = lerp(from, to, easeInOut(k));
        if (k >= 1) transitioning = false;
      }
      if (t !== lastT) { fieldDirty = true; lastT = t; }
      const pts = currentPts();
      const s = currentKernel();

      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = "#0a1426";
      ctx.fillRect(0, 0, W, H);

      if (fieldDirty) { drawField(pts, s); fieldDirty = false; }
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(off, 0, 0, W, H);

      // 座標軸
      ctx.strokeStyle = "rgba(255,255,255,0.12)";
      ctx.beginPath();
      ctx.moveTo(0, H / 2); ctx.lineTo(W, H / 2);
      ctx.moveTo(W / 2, 0); ctx.lineTo(W / 2, H);
      ctx.stroke();
      ctx.fillStyle = "rgba(255,255,255,0.45)";
      ctx.font = "16px 'Noto Sans TC', sans-serif";
      ctx.fillText("z₁", W - 36, H / 2 - 8);
      ctx.fillText("z₂", W / 2 + 8, 18);

      // 訓練樣本（潛在點）
      ctx.fillStyle = "#ffffff";
      for (const p of pts) {
        const [px, py] = toPx(p[0], p[1]);
        ctx.beginPath(); ctx.arc(px, py, 2.4, 0, Math.PI * 2); ctx.fill();
      }

      // 取樣位置
      if (marker) {
        const [mx, my] = toPx(marker[0], marker[1]);
        const f = densityAt(marker[0], marker[1], pts, s);
        const ok = f >= THR;
        ctx.strokeStyle = ok ? `rgb(${TEAL})` : `rgb(${ORANGE})`;
        ctx.lineWidth = 3;
        ctx.beginPath(); ctx.arc(mx, my, 11, 0, Math.PI * 2); ctx.stroke();
        ctx.lineWidth = 1;
        readout.textContent = ok
          ? "✓ 高密度區：解碼結果接近訓練資料"
          : "✗ 空白區：解碼結果可能是雜訊或怪圖";
        readout.style.color = ok ? `rgb(${TEAL})` : `rgb(${ORANGE})`;
      } else {
        readout.textContent = "";
      }
    }

    // 互動
    function setTarget(to_) {
      from = t; to = to_; startT = performance.now(); transitioning = true;
      toggleBtnState();
    }
    function sampleAt() {
      const gg = gaussSampler(Math.random);
      marker = [clamp(gg() * 0.6, -0.95, 0.95), clamp(gg() * 0.6, -0.6, 0.6)];
    }
    let toggleBtn = null;
    function toggleBtnState() {
      if (toggleBtn) toggleBtn.textContent = to > 0.5 ? "切換為 AE" : "切換為 VAE";
    }

    canvas.addEventListener("click", (e) => {
      const r = canvas.getBoundingClientRect();
      const px = (e.clientX - r.left) * (W / r.width);
      const py = (e.clientY - r.top) * (H / r.height);
      marker = toWorld(px, py);
    });

    section.querySelectorAll("button[data-action]").forEach((b) => {
      const action = b.dataset.action;
      if (action === "toggle") { toggleBtn = b; b.addEventListener("click", () => setTarget(to > 0.5 ? 0 : 1)); }
      if (action === "sample") b.addEventListener("click", sampleAt);
    });
    const start = container.dataset.start === "vae" ? 1 : 0;
    t = from = to = start; lastT = -1; toggleBtnState();

    return { render };
  }

  // ---------- Demo 2: reparameterization ----------
  function reparamDemo(container, section) {
    const { canvas, ctx } = setupCanvas(container);
    const controls = document.createElement("div");
    controls.className = "demo-controls";
    container.appendChild(controls);
    const readout = document.createElement("div");
    readout.className = "readout";
    container.appendChild(readout);

    let eps = [];
    const M = 120;
    function resample() {
      const g = gaussSampler(Math.random);
      eps = [];
      for (let i = 0; i < M; i++) eps.push([g(), g()]);
    }
    resample();

    const st = { mx: 0.6, my: -0.4, sx: 0.8, sy: 0.5, k: 5 };
    makeSlider(controls, "μ₁", -2, 2, 0.05, st.mx, (v) => (st.mx = v));
    makeSlider(controls, "μ₂", -2, 2, 0.05, st.my, (v) => (st.my = v));
    makeSlider(controls, "σ₁", 0.05, 2, 0.01, st.sx, (v) => (st.sx = v));
    makeSlider(controls, "σ₂", 0.05, 2, 0.01, st.sy, (v) => (st.sy = v));

    const LX = 200, RX = 600, CY = H / 2, SC = 58;
    section.querySelectorAll("button[data-action]").forEach((b) => {
      if (b.dataset.action === "resample") b.addEventListener("click", resample);
    });

    function drawAxes(cx) {
      ctx.strokeStyle = "rgba(255,255,255,0.12)";
      ctx.beginPath();
      ctx.moveTo(cx - 180, CY); ctx.lineTo(cx + 180, CY);
      ctx.moveTo(cx, CY - 170); ctx.lineTo(cx, CY + 170);
      ctx.stroke();
    }
    function dot(x, y, r, color) {
      ctx.fillStyle = color;
      ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill();
    }

    function render() {
      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = "#0a1426"; ctx.fillRect(0, 0, W, H);

      // 左：ε 空間
      drawAxes(LX);
      ctx.strokeStyle = "rgba(255,255,255,0.25)";
      ctx.setLineDash([4, 4]);
      for (const R of [1, 2]) {
        ctx.beginPath(); ctx.arc(LX, CY, R * SC, 0, Math.PI * 2); ctx.stroke();
      }
      ctx.setLineDash([]);
      ctx.fillStyle = "rgba(255,255,255,0.55)";
      ctx.font = "15px 'Noto Sans TC', sans-serif";
      ctx.fillText("ε 空間（不含參數）", LX - 80, 26);
      for (let i = 0; i < eps.length; i++) {
        dot(LX + eps[i][0] * SC, CY - eps[i][1] * SC, 2.6, `rgba(${GRAY},1)`);
      }

      // 右：z 空間
      drawAxes(RX);
      ctx.fillStyle = "rgba(255,255,255,0.55)";
      ctx.fillText("z 空間：z = μ + σ⊙ε", RX - 90, 26);
      // 1σ 橢圓
      ctx.strokeStyle = `rgba(${ORANGE},0.9)`;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.ellipse(RX + st.mx * SC, CY - st.my * SC, st.sx * SC, st.sy * SC, 0, 0, Math.PI * 2);
      ctx.stroke();
      ctx.lineWidth = 1;
      for (let i = 0; i < eps.length; i++) {
        const zx = st.mx + st.sx * eps[i][0];
        const zy = st.my + st.sy * eps[i][1];
        dot(RX + zx * SC, CY - zy * SC, 2.6, `rgba(${ORANGE},0.85)`);
      }
      // μ 中心
      dot(RX + st.mx * SC, CY - st.my * SC, 5, "#ffffff");

      // 一組對應配對：ε_k 與 z_k
      const k = st.k;
      const ex = LX + eps[k][0] * SC, ey = CY - eps[k][1] * SC;
      const zx = RX + (st.mx + st.sx * eps[k][0]) * SC;
      const zy = CY - (st.my + st.sy * eps[k][1]) * SC;
      ctx.strokeStyle = `rgba(${TEAL},0.9)`;
      ctx.setLineDash([6, 4]);
      ctx.beginPath(); ctx.moveTo(ex + 6, ey); ctx.lineTo(zx - 6, zy); ctx.stroke();
      ctx.setLineDash([]);
      dot(ex, ey, 6, `rgb(${TEAL})`);
      dot(zx, zy, 6, `rgb(${TEAL})`);

      readout.textContent = `ε = (${eps[k][0].toFixed(2)}, ${eps[k][1].toFixed(2)})，青線連接同一組 ε 與 z`;
    }

    return { render };
  }

  // ---------- Demo 3: KL geometry (1-D) ----------
  function klDemo(container) {
    const { canvas, ctx } = setupCanvas(container);
    const controls = document.createElement("div");
    controls.className = "demo-controls";
    container.appendChild(controls);
    const readout = document.createElement("div");
    readout.className = "readout";
    container.appendChild(readout);

    const st = { mu: 1.5, sigma: 0.4 };
    makeSlider(controls, "μ（均值）", -3, 3, 0.05, st.mu, (v) => (st.mu = v));
    makeSlider(controls, "σ（標準差）", 0.05, 3, 0.01, st.sigma, (v) => (st.sigma = v));

    const X0 = 60, X1 = 740, YB = 330, YT = 40, XMIN = -5, XMAX = 5, YMAX = 0.9;
    const xpx = (x) => X0 + (x - XMIN) / (XMAX - XMIN) * (X1 - X0);
    const ypx = (y) => YB - Math.min(y, YMAX) / YMAX * (YB - YT);
    const pdf = (x, m, s) => Math.exp(-0.5 * ((x - m) / s) ** 2) / (s * Math.sqrt(2 * Math.PI));

    function curve(m, s, color, fill) {
      ctx.beginPath();
      for (let i = 0; i <= 400; i++) {
        const x = XMIN + (XMAX - XMIN) * i / 400;
        const px = xpx(x), py = ypx(pdf(x, m, s));
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }
      ctx.strokeStyle = color; ctx.lineWidth = 2.5; ctx.stroke();
      if (fill) {
        ctx.lineTo(xpx(XMAX), YB); ctx.lineTo(xpx(XMIN), YB); ctx.closePath();
        ctx.fillStyle = fill; ctx.fill();
      }
    }

    function render() {
      const { mu, sigma } = st;
      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = "#0a1426"; ctx.fillRect(0, 0, W, H);

      // 軸
      ctx.strokeStyle = "rgba(255,255,255,0.2)";
      ctx.beginPath(); ctx.moveTo(X0, YB); ctx.lineTo(X1, YB); ctx.stroke();
      ctx.fillStyle = "rgba(255,255,255,0.5)";
      ctx.font = "15px 'Noto Sans TC', sans-serif";
      for (let x = -4; x <= 4; x += 2) ctx.fillText(String(x), xpx(x) - 4, YB + 20);

      // 先驗 N(0,1)
      curve(0, 1, `rgba(${GRAY},1)`, `rgba(${GRAY},0.18)`);
      // 編碼器分布 N(μ,σ²)
      curve(mu, sigma, `rgb(${ORANGE})`, `rgba(${ORANGE},0.22)`);
      // 均值位置
      ctx.strokeStyle = `rgba(${ORANGE},0.8)`;
      ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(xpx(mu), YB); ctx.lineTo(xpx(mu), ypx(pdf(mu, mu, sigma))); ctx.stroke();
      ctx.setLineDash([]);

      ctx.fillStyle = "rgba(255,255,255,0.7)";
      ctx.fillText("灰：先驗 N(0, 1)", X0 + 8, YT + 6);
      ctx.fillStyle = `rgb(${ORANGE})`;
      ctx.fillText("橘：N(μ, σ²)", X0 + 8, YT + 26);

      // KL 拆解
      const termMu = 0.5 * mu * mu;
      const termSig = 0.5 * (sigma * sigma - 1 - Math.log(sigma * sigma));
      const kl = termMu + termSig;
      const barMax = 3;
      const barW = (v) => clamp(v / barMax, 0, 1) * 360;
      const BAR_X = X0 + 280;
      ctx.font = "16px 'Noto Sans TC', sans-serif";
      ctx.fillStyle = `rgb(${PURPLE})`;
      ctx.fillText(`μ 項 ½μ² = ${termMu.toFixed(2)}`, X0, 372);
      ctx.fillRect(BAR_X, 358, barW(termMu), 14);
      ctx.fillStyle = `rgb(${TEAL})`;
      ctx.fillText(`σ 項 ½(σ²−1−log σ²) = ${termSig.toFixed(2)}`, X0, 402);
      ctx.fillRect(BAR_X, 388, barW(termSig), 14);
      ctx.fillStyle = "#ffffff";
      ctx.fillText(`KL = ${kl.toFixed(2)}`, X0, 428);

      readout.textContent = Math.abs(sigma - 1) < 0.02 && Math.abs(mu) < 0.05
        ? "KL = 0：與先驗完全相同（μ = 0, σ = 1）"
        : (sigma < 0.2 ? "σ 很小：雲接近一個點 → 對數項把 KL 推得很大" : "");
    }

    return { render };
  }

  // ---------- registry ----------
  const builders = { latent: latentDemo, reparam: reparamDemo, kl: klDemo };

  function init() {
    const instances = [];
    document.querySelectorAll(".demo[data-demo]").forEach((el) => {
      const build = builders[el.dataset.demo];
      if (!build) return;
      const section = el.closest("section") || document;
      instances.push({ el, section, draw: build(el, section).render });
    });

    // 只重繪目前顯示中的投影片，省 CPU
    function loop(now) {
      for (const d of instances) {
        if (d.section.classList.contains("present")) d.draw(now);
      }
      requestAnimationFrame(loop);
    }
    requestAnimationFrame(loop);
  }

  window.VAEDemos = { init };
})();
