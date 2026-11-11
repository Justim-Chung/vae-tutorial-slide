# vae-tutorial-slide

使用 Claude 進行 VAE 教學投影片的製作。

教學對象：工業管理系學生（已了解 MLP、CNN，並會用 PyTorch 訓練影像分類模型）。

## 內容

- `slides/index.html`：reveal.js 投影片（繁體中文），含 KaTeX 數學式與互動式小實驗。
- `slides/assets/demos.js`：互動示範（AE 與 VAE 潛在空間、重參數化、KL 幾何意義）。
- `slides/assets/style.css`：版面與配色。
- `slides/media/*.mp4`：Manim 動畫（由 `animations/vae_scenes.py` 產生）。
- `animations/vae_scenes.py`：Manim 場景（`VAEPipeline`、`KLTradeoff`、`ELBOSampling`）。

## 檢視投影片

直接用瀏覽器開啟 `slides/index.html`，或在 `slides/` 目錄啟動靜態伺服器：

```bash
cd slides
python3 -m http.server 8000
```

然後開啟 http://localhost:8000 。投影片需要連網以載入 reveal.js、KaTeX 與 Google Fonts（CDN）。

## 重新產生動畫

需要 Manim（已在 0.22.1 測試）以及系統中的 CJK 字型（預設 `WenQuanYi Zen Hei`，可在 `animations/vae_scenes.py` 中修改 `CJK_FONT`）：

```bash
pip install manim
manim -qm -o vae_pipeline  animations/vae_scenes.py VAEPipeline
manim -qm -o kl_tradeoff   animations/vae_scenes.py KLTradeoff
manim -qm -o elbo_sampling animations/vae_scenes.py ELBOSampling
cp media/videos/vae_scenes/720p30/{vae_pipeline,kl_tradeoff,elbo_sampling}.mp4 slides/media/
```

## 之後的工作

- 將投影片內容譯為英文（建議另存 `slides/index.en.html`，保留相同結構）。
