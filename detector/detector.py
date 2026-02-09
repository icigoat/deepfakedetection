import cv2
import numpy as np
import os
from collections import defaultdict
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import gc
from .analyzers import (
    FrequencyAnalyzer, NoiseAnalyzer, ELAAnalyzer,
    ColorAnalyzer, TextureAnalyzer, CompressionDetector,
    SRMAnalyzer, PixelAnalyzer, CompressionAnalyzer,
    MediaContext, build_image_context, Stats
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Evidence:
    def __init__(self, category, description, strength, value, expected_range):
        self.category = category
        self.description = description
        self.strength = strength
        self.value = value
        self.expected_range = expected_range
        self.direction = "ai"

    @property
    def deviation_ratio(self):
        lo, hi = self.expected_range
        if lo == 0 and hi == 0:
            return self.strength
        range_width = hi - lo
        if range_width <= 0:
            return self.strength
        if self.value < lo:
            return (lo - self.value) / (range_width + 1e-10)
        elif self.value > hi:
            return (self.value - hi) / (range_width + 1e-10)
        return 0.0


# ================================================================
# Deep Feature Analyzer (MISSING from Django - added back)
# ================================================================
class DeepFeatureAnalyzer:
    def __init__(self):
        self.model = None
        self.transform = None
        self._hooks = []
        self._features = {}
        try:
            base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self._hooks.append(
                base.layer3.register_forward_hook(self._hook('layer3')))
            self._hooks.append(
                base.layer4.register_forward_hook(self._hook('layer4')))
            self.model = base
            self.model.eval().to(device)
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])
        except Exception as e:
            print(f"Deep model failed: {e}")

    def _hook(self, name):
        def fn(m, i, o):
            self._features[name] = o.detach()
        return fn

    def analyze(self, frame):
        if self.model is None:
            return self._empty()
        try:
            self._features.clear()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            t = self.transform(rgb).unsqueeze(0).to(device)
            with torch.no_grad():
                _ = self.model(t)
            results = {}
            for layer in ['layer3', 'layer4']:
                if layer in self._features:
                    p = F.adaptive_avg_pool2d(
                        self._features[layer], 1).cpu().numpy().flatten()
                    results[f'{layer}_sparsity'] = float(
                        np.mean(np.abs(p) < 0.01))
                    results[f'{layer}_std'] = float(np.std(p))
                    results[f'{layer}_kurtosis'] = Stats.safe_kurtosis(p)
            self._features.clear()
            return results
        except:
            return self._empty()

    def _empty(self):
        r = {}
        for l in ['layer3', 'layer4']:
            for s in ['_sparsity', '_std', '_kurtosis']:
                r[f'{l}{s}'] = 0.0
        return r

    def cleanup(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ================================================================
# Temporal Analyzer (RESTORED from Colab)
# ================================================================
class TemporalAnalyzer:
    def analyze(self, frames):
        if len(frames) < 2:
            return self._empty()
        try:
            n = len(frames)
            diffs, nc, cs, fm = [], [], [], []
            for i in range(n - 1):
                f1 = frames[i].astype(np.float64)
                f2 = frames[i + 1].astype(np.float64)
                diffs.append(float(np.mean(np.abs(f1 - f2))))
                g1 = cv2.cvtColor(frames[i],
                                  cv2.COLOR_BGR2GRAY).astype(np.float64)
                g2 = cv2.cvtColor(frames[i + 1],
                                  cv2.COLOR_BGR2GRAY).astype(np.float64)
                n1 = (g1 - cv2.GaussianBlur(g1, (5, 5), 1.0)).flatten()[:10000]
                n2 = (g2 - cv2.GaussianBlur(g2, (5, 5), 1.0)).flatten()[:10000]
                nc.append(Stats.safe_corr(n1, n2))
                mc1 = np.mean(f1, axis=(0, 1))
                mc2 = np.mean(f2, axis=(0, 1))
                cs.append(float(np.linalg.norm(mc1 - mc2)))
                try:
                    flow = cv2.calcOpticalFlowFarneback(
                        g1.astype(np.uint8), g2.astype(np.uint8),
                        None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    fm.append(float(np.mean(
                        np.sqrt(flow[..., 0]**2 + flow[..., 1]**2))))
                except:
                    fm.append(0.0)

            diffs = np.array(diffs)
            nc_arr = np.array(nc)
            flicker = float(
                np.std(diffs) / (np.mean(diffs) + 1e-10))
            jerk = float(
                np.std(np.diff(diffs)) / (np.mean(diffs) + 1e-10)
            ) if len(diffs) > 2 else 0.0
            fcv = float(
                np.std(fm) / (np.mean(fm) + 1e-10)
            ) if fm and np.mean(fm) > 0.1 else 0.0

            return {
                'flicker_score': flicker,
                'motion_jerk': jerk,
                'noise_corr_mean': float(np.mean(nc_arr)),
                'noise_corr_abs_mean': float(np.mean(np.abs(nc_arr))),
                'color_shift_mean': float(np.mean(cs)),
                'color_shift_std': float(np.std(cs)),
                'flow_mean': float(np.mean(fm)) if fm else 0.0,
                'flow_cv': fcv,
                'frame_diff_mean': float(np.mean(diffs)),
            }
        except:
            return self._empty()

    def _empty(self):
        return {k: 0.0 for k in [
            'flicker_score', 'motion_jerk', 'noise_corr_mean',
            'noise_corr_abs_mean', 'color_shift_mean', 'color_shift_std',
            'flow_mean', 'flow_cv', 'frame_diff_mean']}


# ================================================================
# Metadata Analyzer (MISSING from Django - added back)
# ================================================================
class MetadataAnalyzer:
    def analyze(self, video_path, info):
        flags = []
        dur = info.get('duration', 0)
        if 0 < dur < 6:
            flags.append(Evidence('metadata', f'Very short ({dur:.1f}s)',
                                  0.25, dur, (10, 300)))
        elif 0 < dur < 15:
            flags.append(Evidence('metadata', f'Short ({dur:.1f}s)',
                                  0.12, dur, (10, 300)))
        w, h = info.get('width', 0), info.get('height', 0)
        std_res = [
            (1920, 1080), (1280, 720), (3840, 2160), (640, 480),
            (1080, 1920), (720, 1280), (1080, 1080), (720, 720),
            (854, 480), (1920, 1200), (2560, 1440)]
        if w > 0 and h > 0:
            if not any(abs(w - sw) < 10 and abs(h - sh) < 10
                       for sw, sh in std_res):
                flags.append(Evidence(
                    'metadata', f'Non-standard resolution {w}x{h}',
                    0.18, 0, (0, 0)))
        return flags


# ================================================================
# Video Scoring Engine (MATCHES COLAB EXACTLY)
# ================================================================
class VideoScoringEngine:
    def score(self, frame_metrics, temporal_metrics, metadata_flags,
              deep_metrics):
        all_evidence = []
        component_scores = {}

        def avg(key):
            vals = frame_metrics.get(key, [])
            if isinstance(vals, (int, float, np.floating)):
                return float(vals) if np.isfinite(vals) else None
            clean = [v for v in vals if np.isfinite(v)]
            return float(np.mean(clean)) if clean else None

        ev = self._eval_frequency(avg)
        all_evidence.extend(ev)
        component_scores['frequency'] = self._category_score(ev)

        ev = self._eval_noise(avg)
        all_evidence.extend(ev)
        component_scores['noise'] = self._category_score(ev)

        ev = self._eval_ela(avg)
        all_evidence.extend(ev)
        component_scores['ela'] = self._category_score(ev)

        ev = self._eval_srm(avg)
        all_evidence.extend(ev)
        component_scores['srm'] = self._category_score(ev)

        ev = self._eval_color(avg)
        all_evidence.extend(ev)
        component_scores['color'] = self._category_score(ev)

        ev = self._eval_texture(avg)
        all_evidence.extend(ev)
        component_scores['texture'] = self._category_score(ev)

        ev = self._eval_pixels(avg)
        all_evidence.extend(ev)
        component_scores['pixels'] = self._category_score(ev)

        ev = self._eval_temporal(temporal_metrics)
        all_evidence.extend(ev)
        component_scores['temporal'] = self._category_score(ev)

        all_evidence.extend(metadata_flags)
        component_scores['metadata'] = self._category_score(metadata_flags)

        ev = self._eval_deep(deep_metrics)
        all_evidence.extend(ev)
        component_scores['deep'] = self._category_score(ev)

        score, confidence = self._aggregate(all_evidence, component_scores)
        verdict = self._verdict(score, confidence)
        return score, confidence, verdict, component_scores, all_evidence

    def _eval_frequency(self, avg):
        ev = []
        slope = avg('spectral_slope')
        if slope is not None:
            if slope > -0.5:
                deviation = abs(slope - (-1.5))
                strength = min(0.9, 0.4 + deviation * 0.3)
                ev.append(Evidence('frequency',
                    f'Spectrum too flat (slope={slope:.2f}, expect -1 to -2)',
                    strength, slope, (-2.0, -1.0)))
            elif slope > -0.8:
                ev.append(Evidence('frequency',
                    f'Spectrum somewhat flat ({slope:.2f})',
                    0.4, slope, (-2.0, -1.0)))
            elif slope < -3.0:
                ev.append(Evidence('frequency',
                    f'Spectrum too steep ({slope:.2f})',
                    0.35, slope, (-2.0, -1.0)))
        hf = avg('hf_energy_ratio')
        if hf is not None:
            if hf < 0.005:
                ev.append(Evidence('frequency',
                    f'Extremely low HF energy ({hf:.4f})',
                    0.7, hf, (0.02, 0.15)))
            elif hf < 0.015:
                ev.append(Evidence('frequency',
                    f'Very low HF energy ({hf:.4f})',
                    0.55, hf, (0.02, 0.15)))
            elif hf < 0.03:
                ev.append(Evidence('frequency',
                    f'Low HF energy ({hf:.4f})',
                    0.35, hf, (0.02, 0.15)))
        anom = avg('spectral_anomalies')
        if anom is not None and anom > 10:
            ev.append(Evidence('frequency',
                f'Spectral anomalies ({anom:.0f} peaks)',
                min(0.6, anom / 25), anom, (0, 5)))
        return ev

    def _eval_noise(self, avg):
        ev = []
        nu = avg('noise_uniformity')
        if nu is not None and nu > 0:
            if nu < 0.12:
                ev.append(Evidence('noise',
                    f'Extremely uniform noise (CV={nu:.3f})',
                    0.7, nu, (0.25, 0.7)))
            elif nu < 0.2:
                ev.append(Evidence('noise',
                    f'Very uniform noise (CV={nu:.3f})',
                    0.5, nu, (0.25, 0.7)))
            elif nu < 0.28:
                ev.append(Evidence('noise',
                    f'Somewhat uniform noise (CV={nu:.3f})',
                    0.3, nu, (0.25, 0.7)))
        nb = avg('noise_brightness_corr')
        if nb is not None:
            if abs(nb) < 0.03:
                ev.append(Evidence('noise',
                    f'Noise independent of brightness (r={nb:.3f})',
                    0.55, nb, (0.1, 0.5)))
            elif abs(nb) < 0.08:
                ev.append(Evidence('noise',
                    f'Weak noise-brightness correlation ({nb:.3f})',
                    0.35, nb, (0.1, 0.5)))
            elif nb < -0.1:
                ev.append(Evidence('noise',
                    f'Inverse noise-brightness ({nb:.3f})',
                    0.6, nb, (0.1, 0.5)))
        scv = avg('noise_spatial_cv')
        if scv is not None and 0 < scv < 0.12:
            ev.append(Evidence('noise',
                f'Spatially uniform noise (CV={scv:.3f})',
                0.45, scv, (0.2, 0.6)))
        cc = avg('cross_channel_corr')
        if cc is not None:
            if cc > 0.85:
                ev.append(Evidence('noise',
                    f'Very high cross-channel noise corr ({cc:.3f})',
                    0.55, cc, (0.1, 0.5)))
            elif cc > 0.7:
                ev.append(Evidence('noise',
                    f'High cross-channel noise corr ({cc:.3f})',
                    0.4, cc, (0.1, 0.5)))
        ac = avg('noise_autocorr')
        if ac is not None and abs(ac) > 0.15:
            ev.append(Evidence('noise',
                f'High noise autocorrelation ({ac:.3f})',
                min(0.5, abs(ac) / 0.4), abs(ac), (0.0, 0.1)))
        return ev

    def _eval_ela(self, avg):
        ev = []
        ecv = avg('ela_block_cv')
        if ecv is not None and ecv > 0:
            if ecv < 0.2:
                ev.append(Evidence('ela',
                    f'Very uniform ELA (CV={ecv:.3f})',
                    0.55, ecv, (0.35, 0.8)))
            elif ecv < 0.3:
                ev.append(Evidence('ela',
                    f'Somewhat uniform ELA (CV={ecv:.3f})',
                    0.35, ecv, (0.35, 0.8)))
        er = avg('ela_block_range')
        if er is not None and 0 < er < 1.5:
            ev.append(Evidence('ela', f'Low ELA range ({er:.2f})',
                0.4, er, (3.0, 15.0)))
        return ev

    def _eval_srm(self, avg):
        ev = []
        se = avg('srm_energy_mean')
        if se is not None and se > 0:
            if se < 8:
                ev.append(Evidence('srm',
                    f'Very low SRM energy ({se:.1f})',
                    0.55, se, (15, 80)))
            elif se < 15:
                ev.append(Evidence('srm',
                    f'Low SRM energy ({se:.1f})',
                    0.35, se, (15, 80)))
            elif se < 22:
                ev.append(Evidence('srm',
                    f'Below-average SRM ({se:.1f})',
                    0.2, se, (15, 80)))
        return ev

    def _eval_color(self, avg):
        ev = []
        ea = avg('edge_alignment')
        if ea is not None:
            if ea > 0.97:
                ev.append(Evidence('color',
                    f'No chromatic aberration ({ea:.4f})',
                    0.5, ea, (0.6, 0.93)))
            elif ea > 0.94:
                ev.append(Evidence('color',
                    f'Very low chromatic aberration ({ea:.4f})',
                    0.3, ea, (0.6, 0.93)))
        ce = avg('color_entropy')
        if ce is not None and 0 < ce < 5.5:
            ev.append(Evidence('color', f'Low color entropy ({ce:.2f})',
                0.4, ce, (6.5, 7.8)))
        hr = avg('color_hist_roughness')
        if hr is not None and 0 < hr < 1.0:
            ev.append(Evidence('color',
                f'Smooth color histogram ({hr:.3f})',
                0.3, hr, (1.5, 4.0)))
        return ev

    def _eval_texture(self, avg):
        ev = []
        le = avg('lbp_entropy')
        if le is not None and 0 < le < 5.5:
            ev.append(Evidence('texture',
                f'Low texture diversity (LBP={le:.2f})',
                0.4, le, (6.0, 7.5)))
        du = avg('direction_uniformity')
        if du is not None and 0 < du < 0.75:
            ev.append(Evidence('texture',
                f'Limited edge directions ({du:.3f})',
                0.3, du, (0.85, 0.98)))
        ge = avg('glcm_energy')
        if ge is not None and ge > 0.03:
            ev.append(Evidence('texture',
                f'Repetitive texture (GLCM={ge:.4f})',
                min(0.6, 0.25 + (ge - 0.03) * 5), ge, (0.002, 0.02)))
        ecv = avg('edge_sharpness_cv')
        if ecv is not None and 0 < ecv < 0.3:
            ev.append(Evidence('texture',
                f'Uniform edge sharpness (CV={ecv:.3f})',
                0.3, ecv, (0.4, 0.8)))
        return ev

    def _eval_pixels(self, avg):
        ev = []
        cm = avg('checker_max')
        if cm is not None:
            if cm > 10:
                ev.append(Evidence('pixels',
                    f'Strong checkerboard ({cm:.2f})',
                    0.65, cm, (0, 4)))
            elif cm > 7:
                ev.append(Evidence('pixels',
                    f'Checkerboard artifacts ({cm:.2f})',
                    0.5, cm, (0, 4)))
            elif cm > 5:
                ev.append(Evidence('pixels',
                    f'Mild checkerboard ({cm:.2f})',
                    0.3, cm, (0, 4)))
        return ev

    def _eval_temporal(self, tm):
        ev = []
        if not tm:
            return ev
        nca = tm.get('noise_corr_abs_mean', 0)
        if nca > 0.4:
            ev.append(Evidence('temporal',
                f'Noise too consistent (|r|={nca:.3f})',
                0.5, nca, (0.0, 0.2)))
        elif nca > 0.25:
            ev.append(Evidence('temporal',
                f'Somewhat consistent noise ({nca:.3f})',
                0.3, nca, (0.0, 0.2)))
        fcv = tm.get('flow_cv', 0)
        fmean = tm.get('flow_mean', 0)
        if fmean > 0.5:
            if fcv > 2.0:
                ev.append(Evidence('temporal',
                    f'Erratic flow (CV={fcv:.2f})',
                    0.4, fcv, (0.2, 1.0)))
            elif fcv < 0.05 and fmean > 2.0:
                ev.append(Evidence('temporal',
                    f'Unnaturally smooth motion',
                    0.3, fcv, (0.2, 1.0)))
        fl = tm.get('flicker_score', 0)
        if fl > 1.0:
            ev.append(Evidence('temporal',
                f'High flickering ({fl:.2f})',
                0.4, fl, (0.1, 0.6)))
        cs_val = tm.get('color_shift_std', 0)
        if cs_val > 3.0:
            ev.append(Evidence('temporal',
                f'Color instability (σ={cs_val:.2f})',
                0.3, cs_val, (0, 2.0)))
        return ev

    def _eval_deep(self, dm):
        ev = []
        l4s = dm.get('layer4_sparsity', [])
        if l4s:
            m = float(np.mean(l4s)) if isinstance(l4s, list) else float(l4s)
            if m > 0.5:
                ev.append(Evidence('deep',
                    f'High feature sparsity ({m:.3f})',
                    0.3, m, (0.1, 0.4)))
        return ev

    def _category_score(self, evidence_list):
        if not evidence_list:
            return 0.0
        ai_ev = [e for e in evidence_list if e.direction == 'ai']
        if not ai_ev:
            return 0.0
        max_strength = max(e.strength for e in ai_ev)
        boost = (1.2 if len(ai_ev) >= 3
                 else 1.1 if len(ai_ev) >= 2 else 1.0)
        max_dev_ev = max(ai_ev, key=lambda e: e.deviation_ratio)
        dev_factor = min(1.5, 1.0 + max_dev_ev.deviation_ratio * 0.3)
        raw = max_strength * boost * dev_factor * 100
        return min(100.0, max(0.0, raw))

    def _aggregate(self, all_evidence, component_scores):
        weights = {
            'frequency': 0.15, 'noise': 0.14, 'ela': 0.10, 'srm': 0.09,
            'color': 0.08, 'texture': 0.09, 'pixels': 0.09,
            'temporal': 0.13, 'metadata': 0.06, 'deep': 0.07}
        weighted = sum(
            component_scores.get(k, 0) * w for k, w in weights.items())

        ai_ev = [e for e in all_evidence if e.direction == 'ai']
        strong = [e for e in ai_ev if e.strength >= 0.4]
        indep_cats = len(set(e.category for e in strong))
        all_cats_with_signal = len(
            set(e.category for e in ai_ev if e.strength >= 0.25))

        if indep_cats >= 5:
            weighted = min(100, weighted * 1.4)
        elif indep_cats >= 4:
            weighted = min(100, weighted * 1.25)
        elif indep_cats >= 3:
            weighted = min(100, weighted * 1.15)
        elif indep_cats >= 2:
            weighted = min(100, weighted * 1.05)

        if ai_ev:
            avg_deviation = np.mean([e.deviation_ratio for e in ai_ev])
            if avg_deviation > 2.0:
                weighted = min(100, weighted * 1.15)
            elif avg_deviation > 1.0:
                weighted = min(100, weighted * 1.08)

        if len(ai_ev) == 0:
            weighted = min(weighted, 15)
        elif len(ai_ev) == 1 and ai_ev[0].strength < 0.3:
            weighted = min(weighted, 30)

        if not ai_ev:
            confidence = 25.0
        elif indep_cats >= 4:
            confidence = min(85, 60 + indep_cats * 5)
        elif indep_cats >= 3:
            confidence = min(80, 55 + indep_cats * 5)
        elif all_cats_with_signal >= 3:
            confidence = min(70, 45 + all_cats_with_signal * 5)
        else:
            confidence = min(60, 35 + len(ai_ev) * 5)

        return (float(min(100, max(0, weighted))),
                float(min(90, max(15, confidence))))

    def _verdict(self, score, confidence):
        ub = 12.0
        if confidence < 30:
            return f"🔍 INSUFFICIENT EVIDENCE ({score:.0f}±{ub:.0f})"
        elif score >= 60:
            return f"🤖 STRONG AI INDICATORS ({score:.0f}±{ub:.0f})"
        elif score >= 45:
            return f"⚠️ PROBABLE AI ({score:.0f}±{ub:.0f})"
        elif score >= 30:
            return f"🤔 SUSPICIOUS ({score:.0f}±{ub:.0f})"
        elif score >= 18:
            return f"🔍 MOSTLY REAL ({score:.0f}±{ub:.0f})"
        else:
            return f"✅ CONSISTENT WITH REAL ({score:.0f}±{ub:.0f})"


# ================================================================
# Image Scoring Engine (MATCHES COLAB EXACTLY)
# ================================================================
class ImageScoringEngine:
    def __init__(self, context=None):
        self.ctx = context or MediaContext()

    def score(self, frame_metrics, deep_metrics):
        all_evidence = []
        component_scores = {}

        def avg(key):
            vals = frame_metrics.get(key, [])
            if isinstance(vals, (int, float, np.floating)):
                return float(vals) if np.isfinite(vals) else None
            clean = [v for v in vals if np.isfinite(v)]
            return float(np.mean(clean)) if clean else None

        ctx = self.ctx

        # FREQUENCY
        ev = []
        slope = avg('spectral_slope')
        if slope is not None:
            if ctx.is_heavily_compressed or ctx.is_small:
                if slope > 0.2:
                    ev.append(Evidence('frequency',
                        f'Spectrum very flat even for compressed ({slope:.2f})',
                        min(0.5, 0.2 + abs(slope) * 0.2), slope, (-2.0, -0.5)))
            else:
                if slope > -0.5:
                    ev.append(Evidence('frequency',
                        f'Spectrum too flat (slope={slope:.2f})',
                        min(0.9, 0.4 + abs(slope - (-1.5)) * 0.3),
                        slope, (-2.0, -1.0)))
                elif slope > -0.8:
                    ev.append(Evidence('frequency',
                        f'Spectrum somewhat flat ({slope:.2f})',
                        0.4, slope, (-2.0, -1.0)))
        hf = avg('hf_energy_ratio')
        if hf is not None:
            if ctx.is_heavily_compressed or ctx.is_small:
                if hf < 0.001:
                    ev.append(Evidence('frequency',
                        f'Abnormally low HF ({hf:.4f})',
                        0.4, hf, (0.005, 0.15)))
            else:
                if hf < 0.005:
                    ev.append(Evidence('frequency',
                        f'Extremely low HF ({hf:.4f})',
                        0.7, hf, (0.02, 0.15)))
                elif hf < 0.015:
                    ev.append(Evidence('frequency',
                        f'Very low HF ({hf:.4f})',
                        0.55, hf, (0.02, 0.15)))
                elif hf < 0.03:
                    ev.append(Evidence('frequency',
                        f'Low HF ({hf:.4f})',
                        0.35, hf, (0.02, 0.15)))
        all_evidence.extend(ev)
        component_scores['frequency'] = self._cat(ev)

        # NOISE
        ev = []
        if ctx.is_heavily_compressed:
            nu = avg('noise_uniformity')
            if nu is not None and nu < 0.05:
                ev.append(Evidence('noise',
                    f'Extremely uniform noise for JPEG ({nu:.3f})',
                    0.4, nu, (0.1, 0.7)))
        else:
            nu = avg('noise_uniformity')
            if nu is not None and nu > 0:
                if nu < 0.12:
                    ev.append(Evidence('noise',
                        f'Extremely uniform noise ({nu:.3f})',
                        0.7, nu, (0.25, 0.7)))
                elif nu < 0.2:
                    ev.append(Evidence('noise',
                        f'Very uniform noise ({nu:.3f})',
                        0.5, nu, (0.25, 0.7)))
            nb = avg('noise_brightness_corr')
            if nb is not None and abs(nb) < 0.03:
                ev.append(Evidence('noise',
                    f'Noise independent of brightness ({nb:.3f})',
                    0.55, nb, (0.1, 0.5)))
            if not ctx.has_jpeg_artifacts:
                cc = avg('cross_channel_corr')
                if cc is not None and cc > 0.85:
                    ev.append(Evidence('noise',
                        f'High cross-channel corr ({cc:.3f})',
                        0.55, cc, (0.1, 0.5)))
                ac = avg('noise_autocorr')
                if ac is not None and abs(ac) > 0.15:
                    ev.append(Evidence('noise',
                        f'High autocorrelation ({ac:.3f})',
                        min(0.5, abs(ac) / 0.4), abs(ac), (0.0, 0.1)))
        all_evidence.extend(ev)
        component_scores['noise'] = self._cat(ev)

        # ELA
        ev = []
        ecv = avg('ela_block_cv')
        if ecv is not None and 0 < ecv < 0.2:
            ev.append(Evidence('ela', f'Very uniform ELA ({ecv:.3f})',
                0.55, ecv, (0.35, 0.8)))
        elif ecv is not None and 0 < ecv < 0.3:
            ev.append(Evidence('ela', f'Somewhat uniform ELA ({ecv:.3f})',
                0.35, ecv, (0.35, 0.8)))
        all_evidence.extend(ev)
        component_scores['ela'] = self._cat(ev)

        # SRM
        ev = []
        se = avg('srm_energy_mean')
        if se is not None and 0 < se < 8:
            ev.append(Evidence('srm', f'Very low SRM ({se:.1f})',
                0.55, se, (15, 80)))
        elif se is not None and 0 < se < 15:
            ev.append(Evidence('srm', f'Low SRM ({se:.1f})',
                0.35, se, (15, 80)))
        all_evidence.extend(ev)
        component_scores['srm'] = self._cat(ev)

        # COLOR
        ev = []
        ea = avg('edge_alignment')
        if ea is not None and ea > 0.97:
            ev.append(Evidence('color',
                f'No chromatic aberration ({ea:.4f})',
                0.5, ea, (0.6, 0.93)))
        ce = avg('color_entropy')
        if ce is not None and 0 < ce < 5.5:
            ev.append(Evidence('color', f'Low color entropy ({ce:.2f})',
                0.4, ce, (6.5, 7.8)))
        all_evidence.extend(ev)
        component_scores['color'] = self._cat(ev)

        # TEXTURE
        ev = []
        le = avg('lbp_entropy')
        if le is not None and 0 < le < 5.5:
            ev.append(Evidence('texture',
                f'Low texture diversity ({le:.2f})',
                0.4, le, (6.0, 7.5)))
        ge = avg('glcm_energy')
        if ge is not None and ge > 0.03:
            ev.append(Evidence('texture',
                f'Repetitive texture ({ge:.4f})',
                min(0.6, 0.25 + (ge - 0.03) * 5), ge, (0.002, 0.02)))
        all_evidence.extend(ev)
        component_scores['texture'] = self._cat(ev)

        # PIXELS
        ev = []
        cm = avg('checker_max')
        if cm is not None:
            if ctx.has_jpeg_artifacts:
                if cm > 25:
                    ev.append(Evidence('pixels',
                        f'Pattern beyond JPEG ({cm:.2f})',
                        0.4, cm, (0, 15)))
            else:
                if cm > 10:
                    ev.append(Evidence('pixels',
                        f'Strong checkerboard ({cm:.2f})',
                        0.65, cm, (0, 4)))
                elif cm > 7:
                    ev.append(Evidence('pixels',
                        f'Checkerboard ({cm:.2f})',
                        0.5, cm, (0, 4)))
        all_evidence.extend(ev)
        component_scores['pixels'] = self._cat(ev)

        component_scores['temporal'] = None

        # METADATA
        meta_ev = []
        w, h = ctx.width, ctx.height
        if w == h and w in [512, 768, 1024, 2048]:
            meta_ev.append(Evidence('metadata',
                f'AI-typical square resolution {w}x{h}',
                0.35, 0, (0, 0)))
        all_evidence.extend(meta_ev)
        component_scores['metadata'] = self._cat(meta_ev)

        # DEEP
        ev = []
        l4s = deep_metrics.get('layer4_sparsity', [])
        if l4s:
            m = float(np.mean(l4s)) if isinstance(l4s, list) else float(l4s)
            if m > 0.5:
                ev.append(Evidence('deep', f'High sparsity ({m:.3f})',
                    0.3, m, (0.1, 0.4)))
        all_evidence.extend(ev)
        component_scores['deep'] = self._cat(ev)

        # AGGREGATE (matches Colab exactly)
        weights = {
            'frequency': 0.20, 'noise': 0.18, 'ela': 0.12, 'srm': 0.10,
            'color': 0.10, 'texture': 0.12, 'pixels': 0.10, 'deep': 0.08}
        weighted = sum(
            component_scores.get(k, 0) * w
            for k, w in weights.items()
            if component_scores.get(k) is not None)

        ai_ev = [e for e in all_evidence if e.direction == 'ai']
        strong = [e for e in ai_ev if e.strength >= 0.4]
        indep_cats = len(set(e.category for e in strong))

        if indep_cats >= 4:
            weighted = min(100, weighted * 1.25)
        elif indep_cats >= 3:
            weighted = min(100, weighted * 1.15)
        elif indep_cats >= 2:
            weighted = min(100, weighted * 1.05)

        if len(ai_ev) == 0:
            weighted = min(weighted, 15)

        # Context reductions
        if ctx.is_heavily_compressed:
            weighted *= 0.7
        if ctx.is_small:
            weighted *= 0.8
        if ctx.is_document_photo:
            weighted *= 0.75

        # Confidence (matches Colab)
        if not ai_ev:
            confidence = 25.0
        elif indep_cats >= 4:
            confidence = min(75, 55 + indep_cats * 5)
        elif indep_cats >= 3:
            confidence = min(70, 50 + indep_cats * 5)
        else:
            confidence = min(60, 35 + len(ai_ev) * 5)
        confidence = min(confidence, 80)
        if ctx.is_heavily_compressed:
            confidence = min(confidence, 55)
        if ctx.is_small:
            confidence = min(confidence, 60)
        if ctx.is_document_photo:
            confidence = min(confidence, 50)

        score = float(min(100, max(0, weighted)))
        confidence = float(min(90, max(15, confidence)))
        verdict = self._verdict(score, confidence)
        interp = self._interpret(score, confidence)
        return score, confidence, verdict, component_scores, all_evidence, interp

    def _cat(self, ev_list):
        if not ev_list:
            return 0.0
        ai = [e for e in ev_list if e.direction == 'ai']
        if not ai:
            return 0.0
        mx = max(e.strength for e in ai)
        boost = 1.2 if len(ai) >= 3 else 1.1 if len(ai) >= 2 else 1.0
        dev = max(ai, key=lambda e: e.deviation_ratio)
        df = min(1.5, 1.0 + dev.deviation_ratio * 0.3)
        return min(100.0, max(0.0, mx * boost * df * 100))

    def _verdict(self, score, confidence):
        ub = 12.0
        ctx = self.ctx
        tag = (" [JPEG]" if ctx.is_heavily_compressed
               else " [Small]" if ctx.is_small else "")
        if confidence < 35:
            return f"🔍 LOW CONFIDENCE{tag} ({score:.0f}±{ub:.0f})"
        elif score >= 60:
            return f"🤖 LIKELY AI ({score:.0f}±{ub:.0f})"
        elif score >= 45:
            return f"⚠️ PROBABLY AI ({score:.0f}±{ub:.0f})"
        elif score >= 30:
            return f"🤔 SUSPICIOUS{tag} ({score:.0f}±{ub:.0f})"
        elif score >= 15:
            return f"✅ LIKELY REAL{tag} ({score:.0f}±{ub:.0f})"
        else:
            return f"✅ REAL{tag} ({score:.0f}±{ub:.0f})"

    def _interpret(self, score, confidence):
        ctx = self.ctx
        issues = []
        if ctx.is_heavily_compressed:
            issues.append("heavy JPEG compression")
        if ctx.is_small:
            issues.append("small image size")
        if ctx.is_document_photo:
            issues.append("document/ID style")
        note = (f" (affected by: {', '.join(issues)})" if issues else "")

        if confidence < 35:
            if issues:
                return {
                    'conclusion': 'LIKELY REAL', 'icon': '✅',
                    'explanation':
                        f"No AI indicators found, but quality limits "
                        f"analysis{note}.",
                    'recommendation':
                        "Image appears genuine. Low confidence is "
                        "due to image quality."}
            return {
                'conclusion': 'INCONCLUSIVE', 'icon': '❓',
                'explanation':
                    "Insufficient data for reliable determination.",
                'recommendation': "Try with a higher quality version."}
        elif score >= 60:
            return {
                'conclusion': 'LIKELY AI-GENERATED', 'icon': '🤖',
                'explanation': "Multiple strong AI indicators detected.",
                'recommendation':
                    "This media shows significant signs of AI generation."}
        elif score >= 45:
            return {
                'conclusion': 'PROBABLY AI', 'icon': '⚠️',
                'explanation':
                    f"Several AI indicators detected{note}.",
                'recommendation': "Treat with caution."}
        elif score >= 30:
            return {
                'conclusion': 'SUSPICIOUS', 'icon': '🤔',
                'explanation':
                    f"Some unusual patterns{note}. Could be AI or edited.",
                'recommendation': "Cannot confirm authenticity."}
        elif score >= 15:
            return {
                'conclusion': 'LIKELY REAL', 'icon': '✅',
                'explanation':
                    f"Only minor anomalies{note}, consistent with "
                    f"authentic media.",
                'recommendation': "Appears genuine."}
        else:
            return {
                'conclusion': 'REAL', 'icon': '✅',
                'explanation':
                    f"No significant AI indicators{note}.",
                'recommendation': "This media appears authentic."}


# ================================================================
# Main Detector (MATCHES COLAB EXACTLY)
# ================================================================
class AIDetector:
    def __init__(self):
        self.freq = FrequencyAnalyzer()
        self.noise = NoiseAnalyzer()
        self.ela = ELAAnalyzer()
        self.srm = SRMAnalyzer()
        self.color = ColorAnalyzer()
        self.texture = TextureAnalyzer()
        self.pixel = PixelAnalyzer()
        self.compress = CompressionAnalyzer()
        self.compress_detect = CompressionDetector()
        self.deep = DeepFeatureAnalyzer()
        self.temporal = TemporalAnalyzer()
        self.meta = MetadataAnalyzer()

    def detect_file_type(self, file_path):
        image_exts = (
            '.jpg', '.jpeg', '.png', '.bmp', '.tiff',
            '.tif', '.webp', '.gif')
        video_exts = (
            '.mp4', '.avi', '.mov', '.mkv', '.webm',
            '.flv', '.wmv', '.m4v')
        ext = os.path.splitext(file_path)[1].lower()
        if ext in image_exts:
            return 'image'
        elif ext in video_exts:
            return 'video'
        return 'unknown'

    def load_image(self, image_path, max_dim=768):
        """Match Colab: PIL-based loading with LANCZOS4 resize."""
        try:
            pil_img = Image.open(image_path)
            if hasattr(pil_img, 'n_frames') and pil_img.n_frames > 1:
                pil_img.seek(0)
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            pil_img.close()
        except Exception as e:
            print(f"PIL loading failed: {e}, trying cv2.imread")
            frame = cv2.imread(image_path)
            if frame is None:
                raise ValueError(f"Cannot open image: {image_path}")

        h, w = frame.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            frame = cv2.resize(
                frame, (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_LANCZOS4)
        return [frame], {
            'total_frames': 1, 'fps': 0, 'width': w, 'height': h,
            'duration': 0, 'analyzed': 1, 'is_image': True}

    def extract_frames(self, video_path, max_frames=30, max_dim=768):
        """Match Colab: exact same frame extraction."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        dur = total / max(fps, 0.001)

        indices = set(
            np.linspace(0, total - 1,
                        min(max_frames, total)).astype(int))
        frames = []
        idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx in indices:
                fh, fw = frame.shape[:2]
                if max(fh, fw) > max_dim:
                    scale = max_dim / max(fh, fw)
                    frame = cv2.resize(
                        frame, (int(fw * scale), int(fh * scale)),
                        interpolation=cv2.INTER_LANCZOS4)
                frames.append(frame)
            idx += 1
        cap.release()

        return frames, {
            'total_frames': total, 'fps': fps, 'width': w,
            'height': h, 'duration': dur, 'analyzed': len(frames),
            'is_image': False}

    def analyze(self, file_path):
        ftype = self.detect_file_type(file_path)

        if ftype == 'image':
            return self._analyze_image(file_path)
        elif ftype == 'video':
            return self._analyze_video(file_path)
        else:
            raise ValueError(f"Unsupported file type")

    def _analyze_image(self, file_path):
        frames, info = self.load_image(file_path, 768)

        compression_info = self.compress_detect.analyze(frames[0], file_path)
        context = build_image_context(file_path, info, compression_info)

        metrics = defaultdict(list)
        analyzers = [
            self.freq, self.noise, self.ela, self.srm,
            self.color, self.texture, self.pixel, self.compress]
        for analyzer in analyzers:
            try:
                result = analyzer.analyze(frames[0])
                for k, v in result.items():
                    if isinstance(v, (int, float, np.floating)) \
                            and np.isfinite(v):
                        metrics[k].append(float(v))
            except:
                pass

        deep_metrics = defaultdict(list)
        try:
            r = self.deep.analyze(frames[0])
            for k, v in r.items():
                if isinstance(v, (int, float, np.floating)) \
                        and np.isfinite(v):
                    deep_metrics[k].append(float(v))
        except:
            pass

        scorer = ImageScoringEngine(context)
        score, confidence, verdict, components, evidence, interp = \
            scorer.score(dict(metrics), dict(deep_metrics))

        viz_data = self._get_viz(frames[0])

        return {
            'score': score,
            'confidence': confidence,
            'verdict': verdict,
            'components': components,
            'evidence': evidence,
            'info': info,
            'visualization': viz_data,
            'user_interpretation': interp,
            'context': {
                'is_small': context.is_small,
                'is_heavily_compressed': context.is_heavily_compressed,
                'jpeg_quality': context.jpeg_quality_estimate,
                'is_document_photo': context.is_document_photo
            }
        }

    def _analyze_video(self, file_path):
        frames, info = self.extract_frames(file_path, 30, 768)

        if len(frames) < 2:
            raise ValueError("Not enough frames")

        metrics = defaultdict(list)
        analyzers = [
            self.freq, self.noise, self.ela, self.srm,
            self.color, self.texture, self.pixel, self.compress]
        for frame in frames:
            for analyzer in analyzers:
                try:
                    result = analyzer.analyze(frame)
                    for k, v in result.items():
                        if isinstance(v, (int, float, np.floating)) \
                                and np.isfinite(v):
                            metrics[k].append(float(v))
                except:
                    pass

        deep_metrics = defaultdict(list)
        step = max(1, len(frames) // 5)
        for i in range(0, len(frames), step):
            try:
                r = self.deep.analyze(frames[i])
                for k, v in r.items():
                    if isinstance(v, (int, float, np.floating)) \
                            and np.isfinite(v):
                        deep_metrics[k].append(float(v))
            except:
                pass

        temporal_metrics = self.temporal.analyze(frames)
        metadata_flags = self.meta.analyze(file_path, info)

        scorer = VideoScoringEngine()
        score, confidence, verdict, components, evidence = \
            scorer.score(
                dict(metrics), temporal_metrics,
                metadata_flags, dict(deep_metrics))

        viz_data = self._get_viz(frames[0])

        # Generate user interpretation for videos
        interp = self._generate_video_interpretation(score, confidence, verdict, components)

        return {
            'score': score,
            'confidence': confidence,
            'verdict': verdict,
            'components': components,
            'evidence': evidence,
            'info': info,
            'visualization': viz_data,
            'user_interpretation': interp,
            'context': {}
        }

    def _generate_video_interpretation(self, score, confidence, verdict, components):
        """Generate user-friendly interpretation for video analysis"""
        if score >= 70:
            return {
                'icon': '🤖',
                'conclusion': 'Likely AI-Generated',
                'explanation': 'Multiple forensic indicators suggest this video was created or heavily modified by AI. Temporal inconsistencies and deep learning artifacts are present.',
                'recommendation': 'Exercise caution. This video shows strong signs of AI generation or manipulation.'
            }
        elif score >= 50:
            return {
                'icon': '⚠️',
                'conclusion': 'Suspicious - Possible AI',
                'explanation': 'Several forensic anomalies detected. The video may contain AI-generated content or significant editing.',
                'recommendation': 'Further verification recommended. Some indicators suggest potential AI involvement.'
            }
        elif score >= 30:
            return {
                'icon': '🔍',
                'conclusion': 'Uncertain - Mixed Signals',
                'explanation': 'Analysis shows mixed results. Some natural characteristics present, but also some unusual patterns.',
                'recommendation': 'Results are inconclusive. Consider additional verification methods.'
            }
        else:
            return {
                'icon': '✓',
                'conclusion': 'Likely Authentic',
                'explanation': 'Forensic analysis indicates natural video characteristics. Temporal consistency and authentic compression patterns detected.',
                'recommendation': 'Video appears genuine, though no detection is 100% certain.'
            }

    def _get_viz(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        from scipy import fftpack

        f_shift = fftpack.fftshift(fftpack.fft2(gray))
        magnitude = np.log1p(np.abs(f_shift))
        fft_img = ((magnitude - magnitude.min()) /
                    (magnitude.max() - magnitude.min() + 1e-10) * 255
                    ).astype(np.uint8)

        ela_result = self.ela.analyze(frame)
        ela_map = ela_result.get('ela_map', np.zeros_like(gray))
        ela_img = ((ela_map - ela_map.min()) /
                   (ela_map.max() - ela_map.min() + 1e-10) * 255
                   ).astype(np.uint8)

        noise = (gray.astype(np.float64) -
                 cv2.GaussianBlur(gray.astype(np.float64), (5, 5), 1.0))
        noise_norm = ((noise - noise.min()) /
                      (noise.max() - noise.min() + 1e-10) * 255
                      ).astype(np.uint8)

        return {
            'fft': fft_img,
            'ela': ela_img,
            'noise': noise_norm,
            'original': frame
        }

    def cleanup(self):
        if self.deep:
            self.deep.cleanup()
        gc.collect()