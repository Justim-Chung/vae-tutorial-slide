"""Manim scenes for the VAE teaching slides.

Render from the repo root at 720p30 (-qm), then copy the mp4 into slides/media/:

    manim -qm -o vae_pipeline  animations/vae_scenes.py VAEPipeline
    manim -qm -o kl_tradeoff   animations/vae_scenes.py KLTradeoff
    manim -qm -o elbo_sampling animations/vae_scenes.py ELBOSampling

Output lands in media/videos/vae_scenes/720p30/<name>.mp4.

Chinese text uses CJK_FONT and math-like symbols use MATH_FONT; change these
if your system has different fonts installed.
"""

import numpy as np
from manim import *

config.background_color = "#0b1220"

CJK_FONT = "WenQuanYi Zen Hei"
MATH_FONT = "DejaVu Sans"
TEAL = "#5eead4"     # 潛在空間、雲
ORANGE = "#fb923c"   # 編碼器分布 q(z|x)、σ
PURPLE = "#a78bfa"   # 解碼器、KL
GRAY = "#64748b"     # 先驗 N(0, I)
RED = "#f87171"


def cjk(text, size=28, color=WHITE):
    return Text(text, font=CJK_FONT, font_size=size, color=color)


def math(text, size=30, color=WHITE):
    return Text(text, font=MATH_FONT, font_size=size, color=color)


def caption(text):
    return cjk(text, 24, WHITE).to_edge(DOWN, buff=0.35)


def digit_image(side=0.24, pattern=None):
    """A 5x5 'pixel image' (a crude 7 by default) built from squares."""
    if pattern is None:
        pattern = [
            [1, 1, 1, 1, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
        ]
    grid = VGroup()
    for row in pattern:
        for v in row:
            grid.add(
                Square(side_length=side)
                .set_stroke(color=GRAY, width=0.4)
                .set_fill(WHITE, opacity=0.95 if v else 0.06)
            )
    grid.arrange_in_grid(rows=len(pattern), cols=len(pattern[0]), buff=0)
    return grid


def noise_image(side=0.24, seed=3):
    rng = np.random.default_rng(seed)
    pattern = (rng.random((5, 5)) > 0.5).astype(int).tolist()
    return digit_image(side, pattern)


class Narrated(Scene):
    """Scene with a bottom caption that is swapped in place, not stacked."""

    _cap = None

    def say(self, text, run_time=1.0, wait=0.0):
        new = caption(text)
        if self._cap is None:
            self.play(Write(new), run_time=run_time)
        else:
            self.play(ReplacementTransform(self._cap, new), run_time=run_time)
        self._cap = new
        if wait:
            self.wait(wait)


# ---------------------------------------------------------------------------
class VAEPipeline(Narrated):
    """輸入 x → 編碼器 → 潛在雲 → 取樣 z → 解碼器 → 影像分布 → x̂"""

    def construct(self):
        # --- layout ----------------------------------------------------------
        x_img = digit_image(0.24).move_to([-5.6, 0.8, 0])
        x_lab = cjk("輸入 x", 22).next_to(x_img, DOWN, buff=0.25)

        enc = RoundedRectangle(width=1.9, height=1.1, corner_radius=0.15, color=ORANGE)
        enc.move_to([-3.3, 0.8, 0])
        enc_lab = cjk("編碼器", 26, ORANGE).move_to(enc)

        panel = Square(side_length=3.4, color=TEAL, stroke_width=2).move_to([0, 0.8, 0])
        panel_lab = cjk("潛在空間", 22, TEAL).next_to(panel, UP, buff=0.2)

        dec = RoundedRectangle(width=1.9, height=1.1, corner_radius=0.15, color=PURPLE)
        dec.move_to([3.3, 0.8, 0])
        dec_lab = cjk("解碼器", 26, PURPLE).move_to(dec)

        out_ring = Circle(radius=0.85, color=ORANGE, stroke_width=2).move_to([5.6, 0.8, 0])
        out_fill = Circle(radius=0.85, fill_color=ORANGE, fill_opacity=0.2, stroke_width=0)
        out_fill.move_to(out_ring)
        out_lab = cjk("影像分布 p(x|z)", 20, ORANGE).next_to(out_ring, UP, buff=0.2)

        mu_pt = np.array([0.45, 1.25, 0.0])
        cloud = Ellipse(width=1.6, height=0.9, color=ORANGE, fill_opacity=0.3, stroke_width=2)
        cloud.move_to(mu_pt)
        mu_dot = Dot(mu_pt, color=WHITE, radius=0.05)
        mu_lab = math("μ, σ", 22, ORANGE).next_to(mu_dot, UR, buff=0.05)

        def arrow(p, q):
            return Arrow(p, q, buff=0.03, stroke_width=3, color=WHITE, max_tip_length_to_length_ratio=0.2)

        arr_in = arrow([-4.9, 0.8, 0], [-4.35, 0.8, 0])
        arr_enc = arrow([-2.25, 0.8, 0], [-1.8, 0.8, 0])
        arr_dec_in = arrow([1.8, 0.8, 0], [2.25, 0.8, 0])
        arr_out = arrow([4.35, 0.8, 0], [4.7, 0.8, 0])

        # --- animation -------------------------------------------------------
        self.play(FadeIn(x_img), FadeIn(x_lab), run_time=0.8)
        self.play(FadeIn(enc), FadeIn(enc_lab), GrowArrow(arr_in), run_time=0.8)
        self.say("1. 編碼器讀入影像 x，輸出 μ 與 σ：潛在空間中的一團雲", run_time=1.0)
        self.play(FadeIn(panel), FadeIn(panel_lab), GrowArrow(arr_enc), run_time=0.8)
        self.play(FadeIn(cloud), FadeIn(mu_dot), FadeIn(mu_lab), run_time=1.0)
        self.wait(0.8)

        # sampling z = μ + σ ⊙ ε
        z_pt = mu_pt + np.array([0.35, -0.25, 0])
        z_dot = Dot(z_pt, color=TEAL, radius=0.07)
        z_lab = math("z", 24, TEAL).next_to(z_dot, DOWN, buff=0.12)
        formula = math("z = μ + σ ⊙ ε,    ε ~ N(0, I)", 22, WHITE).move_to([0, -1.6, 0])
        self.say("2. 從這團雲中取樣一個點 z（隨機性來自 ε）", run_time=1.0)
        self.play(Write(formula), run_time=0.8)
        self.play(FadeIn(z_dot, scale=0.4), FadeIn(z_lab), run_time=0.8)
        self.wait(0.4)

        # decoder → image distribution → x̂
        self.play(FadeIn(dec), FadeIn(dec_lab), GrowArrow(arr_dec_in), run_time=0.8)
        self.play(FadeIn(out_fill), Create(out_ring), FadeIn(out_lab), GrowArrow(arr_out), run_time=1.0)
        recon = digit_image(0.2).move_to(out_ring.get_center())
        self.say("3. 解碼器把 z 映射成影像分布 p(x|z)，中心是重建影像 x̂", run_time=1.2)
        self.play(FadeIn(recon), run_time=0.8)
        self.wait(0.8)
        self.play(out_ring.animate.scale(1.25), out_fill.animate.scale(1.25), run_time=0.7)
        self.play(out_ring.animate.scale(0.8), out_fill.animate.scale(0.8), run_time=0.7)
        self.wait(1.5)


# ---------------------------------------------------------------------------
class KLTradeoff(Narrated):
    """只重建 (σ 小) → 加入 KL (σ 變大、μ 靠近原點)：雲如何改變"""

    def construct(self):
        center = np.array([-3.1, -0.15, 0])
        frame_box = Square(side_length=4.8, color=TEAL, stroke_width=2).move_to(center)
        title = cjk("潛在空間（2 維示意）", 22, TEAL).next_to(frame_box, UP, buff=0.2)

        base = [np.array([-1.2, 0.9, 0]), np.array([1.2, 0.9, 0]),
                np.array([-1.2, -0.9, 0]), np.array([1.2, -0.9, 0])]
        a = ValueTracker(1.0)     # 均值縮放：μ = a · base
        s = ValueTracker(0.12)    # 標準差 σ（各維相同）

        def kl_value(av, sv):
            # 平均 KL：每團雲 ½ Σⱼ(μⱼ² + σⱼ² − 1 − log σⱼ²)，對四團雲取平均
            mean_mu2 = np.mean([np.dot(b, b) for b in base])
            return 0.5 * (av ** 2 * mean_mu2 + 2 * (sv ** 2 - 1 - np.log(sv ** 2)))

        clouds = VGroup()
        dots = VGroup()
        for b in base:
            clouds.add(always_redraw(
                lambda b=b: Circle(radius=max(s.get_value() * 0.9, 0.02), color=ORANGE,
                                   fill_opacity=0.35, stroke_width=1.5)
                .move_to(center + a.get_value() * b)
            ))
            dots.add(always_redraw(
                lambda b=b: Dot(center + a.get_value() * b, radius=0.05, color=WHITE)
            ))

        # 用 Text 而非 DecimalNumber（後者依賴 LaTeX）
        kl_label = math("平均 KL =", 24, WHITE).move_to([2.6, 2.0, 0])
        kl_num = always_redraw(
            lambda: math(f"{kl_value(a.get_value(), s.get_value()):.2f}", 30, WHITE)
            .next_to(kl_label, RIGHT, buff=0.2)
        )
        kl_row = VGroup(kl_label, kl_num)

        formula = math("KL = ½ (μ² + σ² − 1 − log σ²)", 22, WHITE).move_to([3.6, 1.0, 0])
        note_rec = cjk("重建項：σ 小、點分開（有空洞）", 20, TEAL).move_to([3.6, 0.1, 0])
        note_mu = cjk("μ 項：把雲拉回原點", 20, PURPLE).move_to([3.6, -0.7, 0])
        note_sig = cjk("σ 項：防止雲縮成點", 20, ORANGE).move_to([3.6, -1.4, 0])

        self.add(frame_box, title, clouds, dots, kl_row)
        self.play(Write(formula), run_time=0.9)
        self.play(FadeIn(note_rec), run_time=0.6)
        self.say("只有重建：雲退化成點，中間留下大片空白", run_time=1.0, wait=1.5)

        self.play(FadeIn(note_mu), FadeIn(note_sig), run_time=0.8)
        self.say("加入 KL：σ 變大、μ 靠近原點，雲開始重疊", run_time=0.8)
        self.play(a.animate.set_value(0.45), s.animate.set_value(0.9),
                  run_time=4.0, rate_func=smooth)
        self.wait(0.8)
        self.say("平衡：雲有適中的大小、彼此重疊，潛在空間變得連續", run_time=1.0, wait=2.0)


# ---------------------------------------------------------------------------
class ELBOSampling(Narrated):
    """訓練後：編碼器的雲覆蓋先驗 N(0, I)；從先驗取樣落在雲上才有意義"""

    def construct(self):
        rng = np.random.default_rng(11)
        center = np.array([-2.6, -0.1, 0])

        panel = Square(side_length=5.2, color=GRAY, stroke_width=1.5).move_to(center)
        title = cjk("潛在空間與先驗 N(0, I)", 22, TEAL).next_to(panel, UP, buff=0.2)
        prior1 = DashedVMobject(Circle(radius=1.0, color=GRAY), num_dashes=40).move_to(center)
        prior2 = DashedVMobject(Circle(radius=2.0, color=GRAY), num_dashes=60).move_to(center)

        # 訓練資料各自對應的雲 q(z|x)
        raw = rng.normal(0, 0.8, size=(40, 2))
        means = np.array([m for m in raw if np.linalg.norm(m) < 2.1])
        cloud_r = 0.3
        clouds = VGroup()
        for m in means:
            clouds.add(Circle(radius=cloud_r, fill_color=TEAL, fill_opacity=0.28,
                              stroke_color=TEAL, stroke_width=1.2)
                       .move_to(center + np.array([m[0], m[1], 0])))

        # 找一個在先驗範圍內、但不在任何雲上的「空白」點
        gap = None
        for r in np.linspace(1.2, 1.9, 15):
            for th in np.linspace(0, 2 * np.pi, 72, endpoint=False):
                p = np.array([r * np.cos(th), r * np.sin(th)])
                if np.min(np.linalg.norm(means - p, axis=1)) > cloud_r + 0.25:
                    gap = p
                    break
            if gap is not None:
                break
        covered = np.array([0.15, -0.25, 0.0])   # 位於某團雲內
        gap = np.append(gap, 0.0)

        dec = RoundedRectangle(width=1.7, height=1.0, corner_radius=0.15, color=PURPLE)
        dec.move_to([2.4, 0.2, 0])
        dec_lab = cjk("解碼器", 24, PURPLE).move_to(dec)
        ok_img = digit_image(0.2).move_to([5.6, 1.4, 0])
        bad_img = noise_image(0.2).move_to([5.6, -1.6, 0])
        ok_lab = cjk("✓ 有意義", 20, TEAL).next_to(ok_img, DOWN, buff=0.15)
        bad_lab = cjk("✗ 雜訊", 20, RED).next_to(bad_img, DOWN, buff=0.15)

        self.play(Create(panel), FadeIn(title), Create(prior1), Create(prior2), run_time=1.2)
        self.say("編碼器把每張訓練圖映射成一團雲 q(z|x)", run_time=1.0)
        self.play(LaggedStart(*[FadeIn(c, scale=0.6) for c in clouds], lag_ratio=0.04), run_time=2.5)
        self.wait(0.4)
        self.say("KL 項把雲拉向 N(0, I)：雲覆蓋了先驗的主要區域", run_time=1.2, wait=1.0)

        # 樣本 A：落在雲上
        self.say("從先驗抽樣 z ~ N(0, I)，落在雲上 → 解碼結果有意義", run_time=1.0)
        dot_a = Dot(center + covered, color=WHITE, radius=0.08)
        arrow_a = Arrow(center + covered + np.array([0.12, 0, 0]), dec.get_left(), buff=0.05,
                        stroke_width=3, color=WHITE)
        self.play(FadeIn(dot_a, scale=0.4), run_time=0.6)
        self.play(FadeIn(dec), FadeIn(dec_lab), GrowArrow(arrow_a), run_time=0.8)
        self.play(FadeIn(ok_img), FadeIn(ok_lab), run_time=0.8)
        self.wait(0.8)

        # 樣本 B：落在空白區
        self.say("另一個樣本落在空白區 → 解碼結果可能是雜訊", run_time=1.0)
        dot_b = Dot(center + gap, color=WHITE, radius=0.08)
        arrow_b = Arrow(center + gap + np.array([0.12, 0, 0]), dec.get_left() + np.array([0, -0.2, 0]),
                        buff=0.05, stroke_width=3, color=WHITE)
        self.play(FadeIn(dot_b, scale=0.4), run_time=0.6)
        self.play(GrowArrow(arrow_b), run_time=0.8)
        self.play(FadeIn(bad_img), FadeIn(bad_lab), run_time=0.8)
        self.wait(0.8)

        self.say("因此 KL 讓「從先驗取樣」變得有意義，這是 VAE 能生成的原因", run_time=1.5, wait=2.5)
