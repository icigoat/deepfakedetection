# ============================================================================
# CELL 1: Install Dependencies
# ============================================================================
!pip install torch torchvision torchaudio --quiet
!pip install opencv-python-headless --quiet
!pip install scipy scikit-learn matplotlib seaborn tqdm Pillow --quiet

# ============================================================================
# CELL 2: Imports
# ============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import fftpack
from scipy.stats import kurtosis, skew, norm
from scipy.signal import medfilt
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
import os, io, json, time, gc, warnings, shutil, traceback

warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Libraries loaded | PyTorch {torch.__version__} | Device: {device}")

# ============================================================================
# CELL 3: Configuration and Data Structures
# ============================================================================

@dataclass
class Config:
    max_frames: int = 30
    max_dimension: int = 768
    ela_qualities: List[int] = field(default_factory=lambda: [75, 85, 95])
    min_sample_size: int = 10000
    uncertainty_band: float = 12.0
    deep_feature_frames: int = 5
    video_extensions: Tuple[str, ...] = ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v')
    image_extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.gif')

config = Config()

@dataclass
class Evidence:
    category: str
    description: str
    strength: float
    value: float
    expected_range: Tuple[float, float]
    direction: str = "ai"

    @property
    def deviation_ratio(self) -> float:
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

@dataclass
class MediaContext:
    """Context for IMAGE analysis only. Videos don't use this."""
    is_small: bool = False
    is_heavily_compressed: bool = False
    jpeg_quality_estimate: float = 100.0
    has_jpeg_artifacts: bool = False
    is_document_photo: bool = False
    width: int = 0
    height: int = 0

# ============================================================================
# CELL 4: Statistical Utilities
# ============================================================================

class Stats:
    @staticmethod
    def safe_corr(a, b):
        n = min(len(a), len(b))
        if n < 3:
            return 0.0
        a, b = a[:n], b[:n]
        if np.std(a) < 1e-10 or np.std(b) < 1e-10:
            return 0.0
        try:
            r = float(np.corrcoef(a, b)[0, 1])
            return r if np.isfinite(r) else 0.0
        except:
            return 0.0

    @staticmethod
    def safe_kurtosis(data):
        if len(data) < 4:
            return 0.0
        try:
            k = float(kurtosis(data, fisher=True))
            return k if np.isfinite(k) else 0.0
        except:
            return 0.0

    @staticmethod
    def safe_entropy(data, bins=256, val_range=None):
        if len(data) < 10:
            return 0.0
        kw = {'range': val_range} if val_range else {}
        hist, _ = np.histogram(data, bins=bins, **kw)
        prob = hist.astype(float) / (hist.sum() + 1e-10)
        prob = prob[prob > 0]
        return float(-np.sum(prob * np.log2(prob))) if len(prob) > 0 else 0.0

# ============================================================================
# CELL 5: File Type Detection & Loaders
# ============================================================================

def detect_file_type(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext in config.image_extensions:
        return 'image'
    elif ext in config.video_extensions:
        return 'video'
    else:
        try:
            img = Image.open(file_path)
            img.verify()
            return 'image'
        except:
            pass
        try:
            cap = cv2.VideoCapture(file_path)
            if cap.isOpened() and cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0:
                cap.release()
                return 'video'
            cap.release()
        except:
            pass
        return 'unknown'

def load_image(image_path, max_dim=768):
    try:
        pil_img = Image.open(image_path)
        if hasattr(pil_img, 'n_frames') and pil_img.n_frames > 1:
            pil_img.seek(0)
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except:
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Cannot open image: {image_path}")

    h, w = frame.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_LANCZOS4)
    return [frame], {'total_frames': 1, 'fps': 0, 'width': w, 'height': h,
                     'duration': 0, 'analyzed': 1}

def extract_frames(video_path, max_frames=30, max_dim=768):
    """ORIGINAL video frame extractor - unchanged."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dur = total / max(fps, 0.001)

    indices = set(np.linspace(0, total - 1, min(max_frames, total)).astype(int))
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
                frame = cv2.resize(frame, (int(fw * scale), int(fh * scale)),
                                   interpolation=cv2.INTER_LANCZOS4)
            frames.append(frame)
        idx += 1
    cap.release()

    return frames, {
        'total_frames': total, 'fps': fps, 'width': w,
        'height': h, 'duration': dur, 'analyzed': len(frames)
    }

# ============================================================================
# CELL 6: Image Compression Detector (IMAGE ONLY)
# ============================================================================

class ImageCompressionDetector:
    def analyze(self, frame, file_path=None):
        result = {'jpeg_quality_estimate': 100.0, 'has_jpeg_artifacts': False,
                  'is_heavily_compressed': False, 'compression_type': 'unknown'}
        
        # FIX: Check file extension FIRST — PNG/BMP/TIFF are lossless
        if file_path:
            ext = os.path.splitext(file_path)[1].lower()
            if ext in ['.png', '.bmp', '.tiff', '.tif']:
                result['compression_type'] = 'lossless'
                result['jpeg_quality_estimate'] = 100.0
                result['has_jpeg_artifacts'] = False
                result['is_heavily_compressed'] = False
                return result  # Skip JPEG analysis entirely
            elif ext in ['.jpg', '.jpeg']:
                result['compression_type'] = 'jpeg'
            elif ext == '.webp':
                result['compression_type'] = 'webp'
        
        # Only run JPEG detection for JPEG/WebP/unknown files
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
            h, w = gray.shape

            boundary_diffs, interior_diffs = [], []
            for i in range(8, min(h-1, 256)):
                diff = np.mean(np.abs(gray[i, :min(w, 256)] - gray[i-1, :min(w, 256)]))
                (boundary_diffs if i % 8 == 0 else interior_diffs).append(diff)
            for j in range(8, min(w-1, 256)):
                diff = np.mean(np.abs(gray[:min(h, 256), j] - gray[:min(h, 256), j-1]))
                (boundary_diffs if j % 8 == 0 else interior_diffs).append(diff)

            if boundary_diffs and interior_diffs:
                ratio = np.mean(boundary_diffs) / (np.mean(interior_diffs) + 1e-10)
                result['has_jpeg_artifacts'] = ratio > 1.15

            dct_energies = []
            for i in range(0, min(h-8, 128), 8):
                for j in range(0, min(w-8, 128), 8):
                    block = gray[i:i+8, j:j+8]
                    dct = cv2.dct(block)
                    hf = np.sum(np.abs(dct[4:, 4:])) / (np.sum(np.abs(dct)) + 1e-10)
                    dct_energies.append(hf)
            if dct_energies:
                result['jpeg_quality_estimate'] = min(100, max(20, np.mean(dct_energies) * 500 + 30))

            result['is_heavily_compressed'] = (
                result['has_jpeg_artifacts'] or result['jpeg_quality_estimate'] < 70
            )
        except:
            pass
        return result

def build_image_context(file_path, info, compression_info):
    ctx = MediaContext()
    ctx.width = info.get('width', 0)
    ctx.height = info.get('height', 0)
    ctx.is_small = max(ctx.width, ctx.height) < 512
    ctx.jpeg_quality_estimate = compression_info.get('jpeg_quality_estimate', 100)
    ctx.has_jpeg_artifacts = compression_info.get('has_jpeg_artifacts', False)
    ctx.is_heavily_compressed = compression_info.get('is_heavily_compressed', False)
    
    # FIX: Lossless formats are NEVER heavily compressed
    comp_type = compression_info.get('compression_type', 'unknown')
    if comp_type == 'lossless':
        ctx.has_jpeg_artifacts = False
        ctx.is_heavily_compressed = False
        ctx.jpeg_quality_estimate = 100.0
    
    aspect = ctx.width / max(ctx.height, 1)
    if (0.6 < aspect < 0.9 and max(ctx.width, ctx.height) < 800) or \
       (0.9 < aspect < 1.1 and max(ctx.width, ctx.height) < 600):
        ctx.is_document_photo = True
    return ctx

# ============================================================================
# CELL 7: Frequency Analyzer (ORIGINAL - unchanged)
# ============================================================================

class FrequencyAnalyzer:
    def analyze(self, frame):
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
            rows, cols = gray.shape
            f_shift = fftpack.fftshift(fftpack.fft2(gray))
            magnitude = np.abs(f_shift)
            mag_log = np.log1p(magnitude)

            cy, cx = rows // 2, cols // 2
            Y, X = np.ogrid[:rows, :cols]
            R = np.sqrt((X - cx)**2 + (Y - cy)**2)
            max_r = min(cy, cx)

            profile = np.zeros(max_r)
            for r in range(max_r):
                mask = (R >= r) & (R < r + 1)
                if np.any(mask):
                    profile[r] = np.mean(mag_log[mask])

            start = max(2, max_r // 50)
            valid_r = np.arange(start, max_r - 1).astype(float)
            valid_p = profile[start:max_r - 1]

            spectral_slope = 0.0
            fit_r2 = 0.0
            residual_std = 0.0

            if len(valid_r) > 20:
                log_r = np.log10(valid_r)
                log_p = np.log10(valid_p + 1e-10)
                ok = np.isfinite(log_r) & np.isfinite(log_p)
                if np.sum(ok) > 10:
                    c = np.polyfit(log_r[ok], log_p[ok], 1)
                    spectral_slope = float(c[0])
                    fitted = np.polyval(c, log_r[ok])
                    res = log_p[ok] - fitted
                    residual_std = float(np.std(res))
                    ss_res = np.sum(res**2)
                    ss_tot = np.sum((log_p[ok] - np.mean(log_p[ok]))**2)
                    fit_r2 = float(1 - ss_res / (ss_tot + 1e-10))

            hf_mask = R > max_r * 0.6
            total_e = np.sum(magnitude**2) + 1e-10
            hf_energy_ratio = float(np.sum(magnitude[hf_mask]**2) / total_e)

            rp_pos = profile[profile > 0]
            flatness = float(np.exp(np.mean(np.log(rp_pos + 1e-10))) /
                             (np.mean(rp_pos) + 1e-10)) if len(rp_pos) > 0 else 0.0

            anomalies = 0
            if len(profile) > 20:
                ks = min(11, len(profile) // 2 * 2 + 1)
                if ks >= 3:
                    smooth = medfilt(profile, kernel_size=ks)
                    det = profile - smooth
                    thr = 3 * np.std(det)
                    anomalies = int(np.sum(np.abs(det) > thr))

            return {
                'spectral_slope': spectral_slope, 'spectral_fit_r2': fit_r2,
                'spectral_residual_std': residual_std, 'hf_energy_ratio': hf_energy_ratio,
                'spectral_flatness': flatness, 'spectral_anomalies': float(anomalies),
                'radial_profile': profile, 'magnitude_spectrum': mag_log
            }
        except:
            return {k: 0.0 for k in ['spectral_slope', 'spectral_fit_r2',
                'spectral_residual_std', 'hf_energy_ratio', 'spectral_flatness',
                'spectral_anomalies']}

# ============================================================================
# CELL 8: Noise Analyzer (ORIGINAL - unchanged)
# ============================================================================

class NoiseAnalyzer:
    def analyze(self, frame):
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
            h, w = gray.shape
            noise_fine = gray - cv2.medianBlur(gray.astype(np.uint8), 3).astype(np.float64)
            noise_med = gray - cv2.GaussianBlur(gray, (5, 5), 1.0)
            noise_coarse = gray - cv2.GaussianBlur(gray, (9, 9), 2.0)
            noise = noise_med

            bs, stride = 32, 16
            block_stds, block_bright = [], []
            for i in range(0, h - bs, stride):
                for j in range(0, w - bs, stride):
                    block_stds.append(np.std(noise[i:i+bs, j:j+bs]))
                    block_bright.append(np.mean(gray[i:i+bs, j:j+bs]))
            block_stds = np.array(block_stds)
            block_bright = np.array(block_bright)

            noise_uniformity = float(np.std(block_stds) / (np.mean(block_stds) + 1e-10)) if len(block_stds) > 0 else 0.0
            nb_corr = Stats.safe_corr(block_bright, block_stds) if len(block_stds) > 20 else 0.0

            qh, qw = h // 4, w // 4
            region_stds = []
            for qi in range(4):
                for qj in range(4):
                    reg = noise[qi*qh:(qi+1)*qh, qj*qw:(qj+1)*qw]
                    if reg.size > 100:
                        region_stds.append(float(np.std(reg)))
            spatial_cv = float(np.std(region_stds) / (np.mean(region_stds) + 1e-10)) if region_stds else 0.0

            flat = noise.flatten()[:config.min_sample_size]
            autocorrs = []
            for lag in [1, 2, 4, 8]:
                if len(flat) > lag + 10:
                    autocorrs.append(Stats.safe_corr(flat[:-lag], flat[lag:]))
            noise_ac = float(np.mean(autocorrs)) if autocorrs else 0.0

            b_ch, g_ch, r_ch = cv2.split(frame.astype(np.float64))
            sz = min(config.min_sample_size, b_ch.size)
            nr = (r_ch - cv2.GaussianBlur(r_ch, (5,5), 1.0)).flatten()[:sz]
            ng = (g_ch - cv2.GaussianBlur(g_ch, (5,5), 1.0)).flatten()[:sz]
            nb = (b_ch - cv2.GaussianBlur(b_ch, (5,5), 1.0)).flatten()[:sz]
            cc = float(np.mean([abs(Stats.safe_corr(nr, ng)), abs(Stats.safe_corr(nr, nb)), abs(Stats.safe_corr(ng, nb))]))
            scale_ratio = float(np.std(noise_fine)) / (float(np.std(noise_coarse)) + 1e-10)

            return {
                'noise_std': float(np.std(noise)), 'noise_uniformity': noise_uniformity,
                'noise_brightness_corr': float(nb_corr), 'noise_spatial_cv': spatial_cv,
                'noise_autocorr': float(noise_ac), 'cross_channel_corr': cc,
                'noise_scale_ratio': float(scale_ratio), 'noise_kurtosis': Stats.safe_kurtosis(noise.flatten()),
                'noise_map': noise
            }
        except:
            return {k: 0.0 for k in ['noise_std', 'noise_uniformity', 'noise_brightness_corr',
                'noise_spatial_cv', 'noise_autocorr', 'cross_channel_corr', 'noise_scale_ratio', 'noise_kurtosis']}

# ============================================================================
# CELL 9: ELA Analyzer (ORIGINAL - unchanged)
# ============================================================================

class ELAAnalyzer:
    def analyze(self, frame, qualities=None):
        if qualities is None:
            qualities = config.ela_qualities
        try:
            results = {}
            ela_maps = []
            for q in qualities:
                pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                buf = io.BytesIO()
                pil.save(buf, format='JPEG', quality=q)
                buf.seek(0)
                comp = cv2.cvtColor(np.array(Image.open(buf)), cv2.COLOR_RGB2BGR)
                ela = np.abs(frame.astype(np.float64) - comp.astype(np.float64))
                ela_gray = np.mean(ela, axis=2)
                ela_maps.append(ela_gray)
                results[f'ela_mean_q{q}'] = float(np.mean(ela_gray))
                results[f'ela_std_q{q}'] = float(np.std(ela_gray))
            primary = ela_maps[len(ela_maps) // 2]
            h, w = primary.shape
            bm = []
            bs = 32
            for i in range(0, h - bs, bs):
                for j in range(0, w - bs, bs):
                    bm.append(np.mean(primary[i:i+bs, j:j+bs]))
            if bm:
                results['ela_block_cv'] = float(np.std(bm) / (np.mean(bm) + 1e-10))
                results['ela_block_range'] = float(np.max(bm) - np.min(bm))
            else:
                results['ela_block_cv'] = 0.0
                results['ela_block_range'] = 0.0
            results['ela_map'] = primary
            return results
        except:
            return {'ela_block_cv': 0.0, 'ela_block_range': 0.0}

# ============================================================================
# CELL 10: SRM Analyzer (ORIGINAL - unchanged)
# ============================================================================

class SRMAnalyzer:
    def __init__(self):
        self.filters = [
            np.array([[0,0,0],[0,-1,1],[0,0,0]], dtype=np.float64),
            np.array([[0,0,0],[0,-1,0],[0,1,0]], dtype=np.float64),
            np.array([[0,0,0],[1,-2,1],[0,0,0]], dtype=np.float64),
            np.array([[0,1,0],[0,-2,0],[0,1,0]], dtype=np.float64),
            np.array([[-1,2,-1],[2,-4,2],[-1,2,-1]], dtype=np.float64),
            np.array([[-1,2,-1],[2,-4,2],[0,0,0]], dtype=np.float64),
        ]
    def analyze(self, frame):
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
            energies, kurts = [], []
            for f in self.filters:
                r = cv2.filter2D(gray, cv2.CV_64F, f)
                energies.append(float(np.mean(r**2)))
                kurts.append(Stats.safe_kurtosis(r.flatten()))
            return {'srm_energy_mean': float(np.mean(energies)), 'srm_energy_std': float(np.std(energies)),
                    'srm_kurtosis_mean': float(np.mean(kurts))}
        except:
            return {'srm_energy_mean': 0.0, 'srm_energy_std': 0.0, 'srm_kurtosis_mean': 0.0}

# ============================================================================
# CELL 11: Color Analyzer (ORIGINAL - unchanged)
# ============================================================================

class ColorAnalyzer:
    def analyze(self, frame):
        try:
            b, g, r = cv2.split(frame.astype(np.float64))
            sz = min(config.min_sample_size, b.size)
            rf, gf, bf = r.flatten()[:sz], g.flatten()[:sz], b.flatten()[:sz]
            rg = Stats.safe_corr(rf, gf)
            rb = Stats.safe_corr(rf, bf)
            gb = Stats.safe_corr(gf, bf)
            entropies = [Stats.safe_entropy(ch.flatten(), 256, (0, 256)) for ch in [b, g, r]]
            color_entropy = float(np.mean(entropies))
            edges = {}
            for nm, ch in [('r', r), ('g', g), ('b', b)]:
                edges[nm] = cv2.Canny(ch.astype(np.uint8), 50, 150).flatten()[:sz]
            ec = [Stats.safe_corr(edges[a].astype(float), edges[b_].astype(float))
                  for a, b_ in [('r','g'), ('r','b'), ('g','b')]]
            edge_alignment = float(np.mean(ec))
            roughness = []
            for ch in [b, g, r]:
                hist, _ = np.histogram(ch, bins=256, range=(0, 256))
                hd = np.diff(hist.astype(float))
                roughness.append(float(np.std(hd) / (np.mean(np.abs(hd)) + 1e-10)))
            sub = frame.reshape(-1, 3)
            if len(sub) > 10000:
                sub = sub[np.random.RandomState(42).choice(len(sub), 10000, replace=False)]
            uniq = len(np.unique(sub, axis=0)) / len(sub)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float64)
            sat = hsv[:, :, 1]
            return {
                'rg_correlation': float(rg), 'rb_correlation': float(rb), 'gb_correlation': float(gb),
                'color_entropy': color_entropy, 'edge_alignment': edge_alignment,
                'color_hist_roughness': float(np.mean(roughness)), 'color_uniqueness': float(uniq),
                'saturation_mean': float(np.mean(sat)), 'saturation_std': float(np.std(sat)),
            }
        except:
            return {k: 0.0 for k in ['rg_correlation', 'rb_correlation', 'gb_correlation',
                'color_entropy', 'edge_alignment', 'color_hist_roughness', 'color_uniqueness',
                'saturation_mean', 'saturation_std']}

# ============================================================================
# CELL 12: Texture Analyzer (ORIGINAL - unchanged)
# ============================================================================

class TextureAnalyzer:
    def analyze(self, frame):
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
            gu8 = gray.astype(np.uint8)
            et = cv2.Canny(gu8, 100, 200)
            el = cv2.Canny(gu8, 30, 60)
            edt = float(np.mean(et > 0))
            edl = float(np.mean(el > 0))
            gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gmag = np.sqrt(gx**2 + gy**2)
            gdir = np.arctan2(gy, gx)
            de = Stats.safe_entropy(gdir.flatten(), 36, (-np.pi, np.pi))
            h, w = gu8.shape
            pad = np.pad(gu8, 1, mode='edge').astype(np.int16)
            center = pad[1:-1, 1:-1]
            offsets = [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]
            lbp = np.zeros((h, w), dtype=np.uint8)
            for bit, (dy, dx) in enumerate(offsets):
                nb = pad[1+dy:h+1+dy, 1+dx:w+1+dx]
                lbp |= ((nb >= center).astype(np.uint8) << (7 - bit))
            lbp_e = Stats.safe_entropy(lbp.flatten(), 256, (0, 256))
            edge_pix = gmag[et > 0]
            ecv = float(np.std(edge_pix) / (np.mean(edge_pix) + 1e-10)) if len(edge_pix) > 50 else 0.0
            levels = 32
            q = np.clip((gray / 256.0 * levels).astype(int), 0, levels - 1)
            left = q[:, :-1].flatten()
            right = q[:, 1:].flatten()
            mp = min(100000, len(left))
            if len(left) > mp:
                idx = np.random.RandomState(42).choice(len(left), mp, replace=False)
                left, right = left[idx], right[idx]
            glcm = np.zeros((levels, levels), dtype=float)
            np.add.at(glcm, (left, right), 1)
            glcm /= (glcm.sum() + 1e-10)
            ii, jj = np.meshgrid(range(levels), range(levels), indexing='ij')
            return {
                'edge_density': edt, 'edge_ratio': float(edt / (edl + 1e-10)),
                'gradient_kurtosis': Stats.safe_kurtosis(gmag.flatten()),
                'direction_uniformity': float(de / 5.17), 'lbp_entropy': float(lbp_e),
                'edge_sharpness_cv': ecv, 'glcm_energy': float(np.sum(glcm**2)),
                'glcm_contrast': float(np.sum((ii - jj)**2 * glcm)),
                'glcm_homogeneity': float(np.sum(glcm / (1 + np.abs(ii - jj)))),
            }
        except:
            return {k: 0.0 for k in ['edge_density', 'edge_ratio', 'gradient_kurtosis',
                'direction_uniformity', 'lbp_entropy', 'edge_sharpness_cv',
                'glcm_energy', 'glcm_contrast', 'glcm_homogeneity']}

# ============================================================================
# CELL 13: Pixel Analyzer (ORIGINAL - unchanged)
# ============================================================================

class PixelAnalyzer:
    def analyze(self, frame):
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
            checker_scores = []
            for s in [1, 2, 4, 8]:
                sz = 2 * s
                k = np.ones((sz, sz), dtype=np.float64)
                k[:s, :s] = 1; k[s:, s:] = 1; k[:s, s:] = -1; k[s:, :s] = -1
                k /= (s * s)
                checker_scores.append(float(np.mean(np.abs(cv2.filter2D(gray, cv2.CV_64F, k)))))
            gh = np.array([[1,1,1],[-2,-2,-2],[1,1,1]], dtype=np.float64) / 6
            gv = np.array([[1,-2,1],[1,-2,1],[1,-2,1]], dtype=np.float64) / 6
            grid = (np.mean(np.abs(cv2.filter2D(gray, cv2.CV_64F, gh))) +
                    np.mean(np.abs(cv2.filter2D(gray, cv2.CV_64F, gv)))) / 2
            hist, _ = np.histogram(gray, 256, (0, 256))
            nz = np.where(hist > 0)[0]
            gaps = 0.0
            if len(nz) > 1:
                ir = hist[nz[0]:nz[-1]+1]
                gaps = float(np.sum(ir == 0) / len(ir))
            pe = Stats.safe_entropy(gray.flatten(), 256, (0, 256))
            hd = np.diff(hist.astype(float))
            hr = float(np.std(hd) / (np.mean(np.abs(hd)) + 1e-10))
            return {'checker_max': float(max(checker_scores)), 'checker_mean': float(np.mean(checker_scores)),
                    'grid_score': float(grid), 'histogram_gaps': gaps, 'pixel_entropy': pe, 'hist_roughness': hr}
        except:
            return {k: 0.0 for k in ['checker_max', 'checker_mean', 'grid_score',
                'histogram_gaps', 'pixel_entropy', 'hist_roughness']}

# ============================================================================
# CELL 14: Compression Analyzer (ORIGINAL - unchanged)
# ============================================================================

class CompressionAnalyzer:
    def analyze(self, frame):
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
            h, w = gray.shape
            boundary, interior = [], []
            lim = min(512, h, w)
            for i in range(1, lim):
                rd = np.mean(np.abs(gray[i, :lim] - gray[i-1, :lim]))
                (boundary if i % 8 == 0 else interior).append(rd)
            for j in range(1, lim):
                cd = np.mean(np.abs(gray[:lim, j] - gray[:lim, j-1]))
                (boundary if j % 8 == 0 else interior).append(cd)
            ratio = float(np.mean(boundary) / (np.mean(interior) + 1e-10)) if boundary and interior else 1.0
            dct_stds = []
            for i in range(0, min(h-8, 256), 8):
                for j in range(0, min(w-8, 256), 8):
                    dct_stds.append(float(np.std(cv2.dct(gray[i:i+8, j:j+8]))))
                    if len(dct_stds) >= 64:
                        break
            dct_cv = float(np.std(dct_stds) / (np.mean(dct_stds) + 1e-10)) if dct_stds else 0.0
            return {'block_boundary_ratio': ratio, 'dct_consistency': dct_cv}
        except:
            return {'block_boundary_ratio': 1.0, 'dct_consistency': 0.0}

# ============================================================================
# CELL 15: Deep Feature Analyzer (ORIGINAL - unchanged)
# ============================================================================

class DeepFeatureAnalyzer:
    def __init__(self):
        self.model = None
        self.transform = None
        self._hooks = []
        self._features = {}
        try:
            base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self._hooks.append(base.layer3.register_forward_hook(self._hook('layer3')))
            self._hooks.append(base.layer4.register_forward_hook(self._hook('layer4')))
            self.model = base
            self.model.eval().to(device)
            self.transform = transforms.Compose([
                transforms.ToPILImage(), transforms.Resize((224, 224)),
                transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        except Exception as e:
            print(f"⚠️ Deep model failed: {e}")

    def _hook(self, name):
        def fn(m, i, o): self._features[name] = o.detach()
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
                    p = F.adaptive_avg_pool2d(self._features[layer], 1).cpu().numpy().flatten()
                    results[f'{layer}_sparsity'] = float(np.mean(np.abs(p) < 0.01))
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

# ============================================================================
# CELL 16: Temporal Analyzer (ORIGINAL - unchanged)
# ============================================================================

class TemporalAnalyzer:
    def analyze(self, frames):
        if len(frames) < 2:
            return self._empty()
        try:
            n = len(frames)
            diffs, nc, cs, fm = [], [], [], []
            for i in range(n - 1):
                f1 = frames[i].astype(np.float64)
                f2 = frames[i+1].astype(np.float64)
                diffs.append(float(np.mean(np.abs(f1 - f2))))
                g1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY).astype(np.float64)
                g2 = cv2.cvtColor(frames[i+1], cv2.COLOR_BGR2GRAY).astype(np.float64)
                n1 = (g1 - cv2.GaussianBlur(g1, (5,5), 1.0)).flatten()[:10000]
                n2 = (g2 - cv2.GaussianBlur(g2, (5,5), 1.0)).flatten()[:10000]
                nc.append(Stats.safe_corr(n1, n2))
                mc1 = np.mean(f1, axis=(0,1))
                mc2 = np.mean(f2, axis=(0,1))
                cs.append(float(np.linalg.norm(mc1 - mc2)))
                try:
                    flow = cv2.calcOpticalFlowFarneback(g1.astype(np.uint8), g2.astype(np.uint8),
                        None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    fm.append(float(np.mean(np.sqrt(flow[...,0]**2 + flow[...,1]**2))))
                except:
                    fm.append(0.0)
            diffs = np.array(diffs)
            nc = np.array(nc)
            flicker = float(np.std(diffs) / (np.mean(diffs) + 1e-10))
            jerk = float(np.std(np.diff(diffs)) / (np.mean(diffs) + 1e-10)) if len(diffs) > 2 else 0.0
            fcv = float(np.std(fm) / (np.mean(fm) + 1e-10)) if fm and np.mean(fm) > 0.1 else 0.0
            return {
                'flicker_score': flicker, 'motion_jerk': jerk,
                'noise_corr_mean': float(np.mean(nc)), 'noise_corr_abs_mean': float(np.mean(np.abs(nc))),
                'color_shift_mean': float(np.mean(cs)), 'color_shift_std': float(np.std(cs)),
                'flow_mean': float(np.mean(fm)) if fm else 0.0, 'flow_cv': fcv,
                'frame_diff_mean': float(np.mean(diffs)),
            }
        except:
            return self._empty()

    def _empty(self):
        return {k: 0.0 for k in ['flicker_score', 'motion_jerk', 'noise_corr_mean',
            'noise_corr_abs_mean', 'color_shift_mean', 'color_shift_std',
            'flow_mean', 'flow_cv', 'frame_diff_mean']}

# ============================================================================
# CELL 17: Metadata Analyzer (ORIGINAL - unchanged)
# ============================================================================

class MetadataAnalyzer:
    def analyze(self, video_path, info):
        flags = []
        dur = info.get('duration', 0)
        if 0 < dur < 6:
            flags.append(Evidence('metadata', f'Very short ({dur:.1f}s)', 0.25, dur, (10, 300)))
        elif 0 < dur < 15:
            flags.append(Evidence('metadata', f'Short ({dur:.1f}s)', 0.12, dur, (10, 300)))
        w, h = info.get('width', 0), info.get('height', 0)
        std_res = [(1920,1080),(1280,720),(3840,2160),(640,480),
                   (1080,1920),(720,1280),(1080,1080),(720,720),
                   (854,480),(1920,1200),(2560,1440)]
        if w > 0 and h > 0:
            if not any(abs(w-sw)<10 and abs(h-sh)<10 for sw,sh in std_res):
                flags.append(Evidence('metadata', f'Non-standard resolution {w}x{h}', 0.18, 0, (0, 0)))
        return flags

# ============================================================================
# CELL 18: VIDEO Scoring Engine (ORIGINAL - 100% unchanged)
# ============================================================================

class VideoScoringEngine:
    """ORIGINAL scoring engine for videos. Not modified at all."""

    def score(self, frame_metrics, temporal_metrics, metadata_flags, deep_metrics):
        all_evidence = []
        component_scores = {}

        def avg(key):
            vals = frame_metrics.get(key, [])
            clean = [v for v in vals if np.isfinite(v)]
            return float(np.mean(clean)) if clean else None

        ev = self._eval_frequency(avg); all_evidence.extend(ev); component_scores['frequency'] = self._category_score(ev)
        ev = self._eval_noise(avg); all_evidence.extend(ev); component_scores['noise'] = self._category_score(ev)
        ev = self._eval_ela(avg); all_evidence.extend(ev); component_scores['ela'] = self._category_score(ev)
        ev = self._eval_srm(avg); all_evidence.extend(ev); component_scores['srm'] = self._category_score(ev)
        ev = self._eval_color(avg); all_evidence.extend(ev); component_scores['color'] = self._category_score(ev)
        ev = self._eval_texture(avg); all_evidence.extend(ev); component_scores['texture'] = self._category_score(ev)
        ev = self._eval_pixels(avg); all_evidence.extend(ev); component_scores['pixels'] = self._category_score(ev)
        ev = self._eval_temporal(temporal_metrics); all_evidence.extend(ev); component_scores['temporal'] = self._category_score(ev)
        all_evidence.extend(metadata_flags); component_scores['metadata'] = self._category_score(metadata_flags)
        ev = self._eval_deep(deep_metrics); all_evidence.extend(ev); component_scores['deep'] = self._category_score(ev)

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
                ev.append(Evidence('frequency', f'Spectrum too flat (slope={slope:.2f}, expect -1 to -2)', strength, slope, (-2.0, -1.0)))
            elif slope > -0.8:
                ev.append(Evidence('frequency', f'Spectrum somewhat flat ({slope:.2f})', 0.4, slope, (-2.0, -1.0)))
            elif slope < -3.0:
                ev.append(Evidence('frequency', f'Spectrum too steep ({slope:.2f})', 0.35, slope, (-2.0, -1.0)))
        hf = avg('hf_energy_ratio')
        if hf is not None:
            if hf < 0.005: ev.append(Evidence('frequency', f'Extremely low HF energy ({hf:.4f})', 0.7, hf, (0.02, 0.15)))
            elif hf < 0.015: ev.append(Evidence('frequency', f'Very low HF energy ({hf:.4f})', 0.55, hf, (0.02, 0.15)))
            elif hf < 0.03: ev.append(Evidence('frequency', f'Low HF energy ({hf:.4f})', 0.35, hf, (0.02, 0.15)))
        anom = avg('spectral_anomalies')
        if anom is not None and anom > 10:
            ev.append(Evidence('frequency', f'Spectral anomalies ({anom:.0f} peaks)', min(0.6, anom / 25), anom, (0, 5)))
        return ev

    def _eval_noise(self, avg):
        ev = []
        nu = avg('noise_uniformity')
        if nu is not None and nu > 0:
            if nu < 0.12: ev.append(Evidence('noise', f'Extremely uniform noise (CV={nu:.3f})', 0.7, nu, (0.25, 0.7)))
            elif nu < 0.2: ev.append(Evidence('noise', f'Very uniform noise (CV={nu:.3f})', 0.5, nu, (0.25, 0.7)))
            elif nu < 0.28: ev.append(Evidence('noise', f'Somewhat uniform noise (CV={nu:.3f})', 0.3, nu, (0.25, 0.7)))
        nb = avg('noise_brightness_corr')
        if nb is not None:
            if abs(nb) < 0.03: ev.append(Evidence('noise', f'Noise independent of brightness (r={nb:.3f})', 0.55, nb, (0.1, 0.5)))
            elif abs(nb) < 0.08: ev.append(Evidence('noise', f'Weak noise-brightness correlation ({nb:.3f})', 0.35, nb, (0.1, 0.5)))
            elif nb < -0.1: ev.append(Evidence('noise', f'Inverse noise-brightness ({nb:.3f})', 0.6, nb, (0.1, 0.5)))
        scv = avg('noise_spatial_cv')
        if scv is not None and 0 < scv < 0.12:
            ev.append(Evidence('noise', f'Spatially uniform noise (CV={scv:.3f})', 0.45, scv, (0.2, 0.6)))
        cc = avg('cross_channel_corr')
        if cc is not None:
            if cc > 0.85: ev.append(Evidence('noise', f'Very high cross-channel noise corr ({cc:.3f})', 0.55, cc, (0.1, 0.5)))
            elif cc > 0.7: ev.append(Evidence('noise', f'High cross-channel noise corr ({cc:.3f})', 0.4, cc, (0.1, 0.5)))
        ac = avg('noise_autocorr')
        if ac is not None and abs(ac) > 0.15:
            ev.append(Evidence('noise', f'High noise autocorrelation ({ac:.3f})', min(0.5, abs(ac) / 0.4), abs(ac), (0.0, 0.1)))
        return ev

    def _eval_ela(self, avg):
        ev = []
        ecv = avg('ela_block_cv')
        if ecv is not None and ecv > 0:
            if ecv < 0.2: ev.append(Evidence('ela', f'Very uniform ELA (CV={ecv:.3f})', 0.55, ecv, (0.35, 0.8)))
            elif ecv < 0.3: ev.append(Evidence('ela', f'Somewhat uniform ELA (CV={ecv:.3f})', 0.35, ecv, (0.35, 0.8)))
        er = avg('ela_block_range')
        if er is not None and 0 < er < 1.5:
            ev.append(Evidence('ela', f'Low ELA range ({er:.2f})', 0.4, er, (3.0, 15.0)))
        return ev

    def _eval_srm(self, avg):
        ev = []
        se = avg('srm_energy_mean')
        if se is not None and se > 0:
            if se < 8: ev.append(Evidence('srm', f'Very low SRM energy ({se:.1f})', 0.55, se, (15, 80)))
            elif se < 15: ev.append(Evidence('srm', f'Low SRM energy ({se:.1f})', 0.35, se, (15, 80)))
            elif se < 22: ev.append(Evidence('srm', f'Below-average SRM ({se:.1f})', 0.2, se, (15, 80)))
        return ev

    def _eval_color(self, avg):
        ev = []
        ea = avg('edge_alignment')
        if ea is not None:
            if ea > 0.97: ev.append(Evidence('color', f'No chromatic aberration ({ea:.4f})', 0.5, ea, (0.6, 0.93)))
            elif ea > 0.94: ev.append(Evidence('color', f'Very low chromatic aberration ({ea:.4f})', 0.3, ea, (0.6, 0.93)))
        ce = avg('color_entropy')
        if ce is not None and 0 < ce < 5.5:
            ev.append(Evidence('color', f'Low color entropy ({ce:.2f})', 0.4, ce, (6.5, 7.8)))
        hr = avg('color_hist_roughness')
        if hr is not None and 0 < hr < 1.0:
            ev.append(Evidence('color', f'Smooth color histogram ({hr:.3f})', 0.3, hr, (1.5, 4.0)))
        return ev

    def _eval_texture(self, avg):
        ev = []
        le = avg('lbp_entropy')
        if le is not None and 0 < le < 5.5: ev.append(Evidence('texture', f'Low texture diversity (LBP={le:.2f})', 0.4, le, (6.0, 7.5)))
        du = avg('direction_uniformity')
        if du is not None and 0 < du < 0.75: ev.append(Evidence('texture', f'Limited edge directions ({du:.3f})', 0.3, du, (0.85, 0.98)))
        ge = avg('glcm_energy')
        if ge is not None and ge > 0.03: ev.append(Evidence('texture', f'Repetitive texture (GLCM={ge:.4f})', min(0.6, 0.25 + (ge - 0.03) * 5), ge, (0.002, 0.02)))
        ecv = avg('edge_sharpness_cv')
        if ecv is not None and 0 < ecv < 0.3: ev.append(Evidence('texture', f'Uniform edge sharpness (CV={ecv:.3f})', 0.3, ecv, (0.4, 0.8)))
        return ev

    def _eval_pixels(self, avg):
        ev = []
        cm = avg('checker_max')
        if cm is not None:
            if cm > 10: ev.append(Evidence('pixels', f'Strong checkerboard ({cm:.2f})', 0.65, cm, (0, 4)))
            elif cm > 7: ev.append(Evidence('pixels', f'Checkerboard artifacts ({cm:.2f})', 0.5, cm, (0, 4)))
            elif cm > 5: ev.append(Evidence('pixels', f'Mild checkerboard ({cm:.2f})', 0.3, cm, (0, 4)))
        return ev

    def _eval_temporal(self, tm):
        ev = []
        if not tm: return ev
        nca = tm.get('noise_corr_abs_mean', 0)
        if nca > 0.4: ev.append(Evidence('temporal', f'Noise too consistent (|r|={nca:.3f})', 0.5, nca, (0.0, 0.2)))
        elif nca > 0.25: ev.append(Evidence('temporal', f'Somewhat consistent noise ({nca:.3f})', 0.3, nca, (0.0, 0.2)))
        fcv = tm.get('flow_cv', 0); fmean = tm.get('flow_mean', 0)
        if fmean > 0.5:
            if fcv > 2.0: ev.append(Evidence('temporal', f'Erratic flow (CV={fcv:.2f})', 0.4, fcv, (0.2, 1.0)))
            elif fcv < 0.05 and fmean > 2.0: ev.append(Evidence('temporal', f'Unnaturally smooth motion', 0.3, fcv, (0.2, 1.0)))
        fl = tm.get('flicker_score', 0)
        if fl > 1.0: ev.append(Evidence('temporal', f'High flickering ({fl:.2f})', 0.4, fl, (0.1, 0.6)))
        cs = tm.get('color_shift_std', 0)
        if cs > 3.0: ev.append(Evidence('temporal', f'Color instability (σ={cs:.2f})', 0.3, cs, (0, 2.0)))
        return ev

    def _eval_deep(self, dm):
        ev = []
        l4s = dm.get('layer4_sparsity', [])
        if l4s:
            m = float(np.mean(l4s))
            if m > 0.5: ev.append(Evidence('deep', f'High feature sparsity ({m:.3f})', 0.3, m, (0.1, 0.4)))
        return ev

    def _category_score(self, evidence_list):
        if not evidence_list: return 0.0
        ai_ev = [e for e in evidence_list if e.direction == 'ai']
        if not ai_ev: return 0.0
        max_strength = max(e.strength for e in ai_ev)
        boost = 1.2 if len(ai_ev) >= 3 else 1.1 if len(ai_ev) >= 2 else 1.0
        max_dev_ev = max(ai_ev, key=lambda e: e.deviation_ratio)
        dev_factor = min(1.5, 1.0 + max_dev_ev.deviation_ratio * 0.3)
        raw = max_strength * boost * dev_factor * 100
        return min(100.0, max(0.0, raw))

    def _aggregate(self, all_evidence, component_scores):
        weights = {'frequency': 0.15, 'noise': 0.14, 'ela': 0.10, 'srm': 0.09,
                   'color': 0.08, 'texture': 0.09, 'pixels': 0.09,
                   'temporal': 0.13, 'metadata': 0.06, 'deep': 0.07}
        weighted = sum(component_scores.get(k, 0) * w for k, w in weights.items())
        ai_ev = [e for e in all_evidence if e.direction == 'ai']
        strong = [e for e in ai_ev if e.strength >= 0.4]
        indep_cats = len(set(e.category for e in strong))
        all_cats_with_signal = len(set(e.category for e in ai_ev if e.strength >= 0.25))
        if indep_cats >= 5: weighted = min(100, weighted * 1.4)
        elif indep_cats >= 4: weighted = min(100, weighted * 1.25)
        elif indep_cats >= 3: weighted = min(100, weighted * 1.15)
        elif indep_cats >= 2: weighted = min(100, weighted * 1.05)
        if ai_ev:
            avg_deviation = np.mean([e.deviation_ratio for e in ai_ev])
            if avg_deviation > 2.0: weighted = min(100, weighted * 1.15)
            elif avg_deviation > 1.0: weighted = min(100, weighted * 1.08)
        if len(ai_ev) == 0: weighted = min(weighted, 15)
        elif len(ai_ev) == 1 and ai_ev[0].strength < 0.3: weighted = min(weighted, 30)
        if not ai_ev: confidence = 25.0
        elif indep_cats >= 4: confidence = min(85, 60 + indep_cats * 5)
        elif indep_cats >= 3: confidence = min(80, 55 + indep_cats * 5)
        elif all_cats_with_signal >= 3: confidence = min(70, 45 + all_cats_with_signal * 5)
        else: confidence = min(60, 35 + len(ai_ev) * 5)
        return float(min(100, max(0, weighted))), float(min(90, max(15, confidence)))

    def _verdict(self, score, confidence):
        ub = config.uncertainty_band
        if confidence < 30: return f"🔍 INSUFFICIENT EVIDENCE ({score:.0f}±{ub:.0f})"
        elif score >= 60: return f"🤖 STRONG AI INDICATORS ({score:.0f}±{ub:.0f})"
        elif score >= 45: return f"⚠️ PROBABLE AI ({score:.0f}±{ub:.0f})"
        elif score >= 30: return f"🤔 SUSPICIOUS — possible AI ({score:.0f}±{ub:.0f})"
        elif score >= 18: return f"🔍 MOSTLY REAL with minor anomalies ({score:.0f}±{ub:.0f})"
        else: return f"✅ CONSISTENT WITH REAL ({score:.0f}±{ub:.0f})"

# ============================================================================
# CELL 19: IMAGE Scoring Engine (NEW - JPEG-aware, separate from video)
# ============================================================================

class ImageScoringEngine:
    """Separate scoring for images with JPEG/compression awareness."""

    def __init__(self, context=None):
        self.ctx = context or MediaContext()

    def score(self, frame_metrics, deep_metrics):
        all_evidence = []
        component_scores = {}

        def avg(key):
            vals = frame_metrics.get(key, [])
            clean = [v for v in vals if np.isfinite(v)]
            return float(np.mean(clean)) if clean else None

        ctx = self.ctx

        # FREQUENCY - adjusted for compression
        ev = []
        slope = avg('spectral_slope')
        if slope is not None:
            if ctx.is_heavily_compressed or ctx.is_small:
                if slope > 0.2:
                    ev.append(Evidence('frequency', f'Spectrum very flat even for compressed ({slope:.2f})', min(0.5, 0.2 + abs(slope) * 0.2), slope, (-2.0, -0.5)))
            else:
                if slope > -0.5:
                    ev.append(Evidence('frequency', f'Spectrum too flat (slope={slope:.2f})', min(0.9, 0.4 + abs(slope - (-1.5)) * 0.3), slope, (-2.0, -1.0)))
                elif slope > -0.8:
                    ev.append(Evidence('frequency', f'Spectrum somewhat flat ({slope:.2f})', 0.4, slope, (-2.0, -1.0)))
        hf = avg('hf_energy_ratio')
        if hf is not None:
            if ctx.is_heavily_compressed or ctx.is_small:
                if hf < 0.001:
                    ev.append(Evidence('frequency', f'Abnormally low HF ({hf:.4f})', 0.4, hf, (0.005, 0.15)))
            else:
                if hf < 0.005: ev.append(Evidence('frequency', f'Extremely low HF ({hf:.4f})', 0.7, hf, (0.02, 0.15)))
                elif hf < 0.015: ev.append(Evidence('frequency', f'Very low HF ({hf:.4f})', 0.55, hf, (0.02, 0.15)))
                elif hf < 0.03: ev.append(Evidence('frequency', f'Low HF ({hf:.4f})', 0.35, hf, (0.02, 0.15)))
        all_evidence.extend(ev); component_scores['frequency'] = self._cat(ev)

        # NOISE - skip most checks for JPEG
        ev = []
        if ctx.is_heavily_compressed:
            nu = avg('noise_uniformity')
            if nu is not None and nu < 0.05:
                ev.append(Evidence('noise', f'Extremely uniform noise for JPEG ({nu:.3f})', 0.4, nu, (0.1, 0.7)))
        else:
            nu = avg('noise_uniformity')
            if nu is not None and nu > 0:
                if nu < 0.12: ev.append(Evidence('noise', f'Extremely uniform noise ({nu:.3f})', 0.7, nu, (0.25, 0.7)))
                elif nu < 0.2: ev.append(Evidence('noise', f'Very uniform noise ({nu:.3f})', 0.5, nu, (0.25, 0.7)))
            nb = avg('noise_brightness_corr')
            if nb is not None and abs(nb) < 0.03:
                ev.append(Evidence('noise', f'Noise independent of brightness ({nb:.3f})', 0.55, nb, (0.1, 0.5)))
            if not ctx.has_jpeg_artifacts:
                cc = avg('cross_channel_corr')
                if cc is not None and cc > 0.85:
                    ev.append(Evidence('noise', f'High cross-channel corr ({cc:.3f})', 0.55, cc, (0.1, 0.5)))
                ac = avg('noise_autocorr')
                if ac is not None and abs(ac) > 0.15:
                    ev.append(Evidence('noise', f'High autocorrelation ({ac:.3f})', min(0.5, abs(ac) / 0.4), abs(ac), (0.0, 0.1)))
        all_evidence.extend(ev); component_scores['noise'] = self._cat(ev)

        # ELA
        ev = []
        ecv = avg('ela_block_cv')
        if ecv is not None and 0 < ecv < 0.2: ev.append(Evidence('ela', f'Very uniform ELA ({ecv:.3f})', 0.55, ecv, (0.35, 0.8)))
        elif ecv is not None and 0 < ecv < 0.3: ev.append(Evidence('ela', f'Somewhat uniform ELA ({ecv:.3f})', 0.35, ecv, (0.35, 0.8)))
        all_evidence.extend(ev); component_scores['ela'] = self._cat(ev)

        # SRM
        ev = []
        se = avg('srm_energy_mean')
        if se is not None and 0 < se < 8: ev.append(Evidence('srm', f'Very low SRM ({se:.1f})', 0.55, se, (15, 80)))
        elif se is not None and 0 < se < 15: ev.append(Evidence('srm', f'Low SRM ({se:.1f})', 0.35, se, (15, 80)))
        all_evidence.extend(ev); component_scores['srm'] = self._cat(ev)

        # COLOR
        ev = []
        ea = avg('edge_alignment')
        if ea is not None and ea > 0.97: ev.append(Evidence('color', f'No chromatic aberration ({ea:.4f})', 0.5, ea, (0.6, 0.93)))
        ce = avg('color_entropy')
        if ce is not None and 0 < ce < 5.5: ev.append(Evidence('color', f'Low color entropy ({ce:.2f})', 0.4, ce, (6.5, 7.8)))
        all_evidence.extend(ev); component_scores['color'] = self._cat(ev)

        # TEXTURE
        ev = []
        le = avg('lbp_entropy')
        if le is not None and 0 < le < 5.5: ev.append(Evidence('texture', f'Low texture diversity ({le:.2f})', 0.4, le, (6.0, 7.5)))
        ge = avg('glcm_energy')
        if ge is not None and ge > 0.03: ev.append(Evidence('texture', f'Repetitive texture ({ge:.4f})', min(0.6, 0.25 + (ge - 0.03) * 5), ge, (0.002, 0.02)))
        all_evidence.extend(ev); component_scores['texture'] = self._cat(ev)

        # PIXELS - JPEG aware
        ev = []
        cm = avg('checker_max')
        if cm is not None:
            if ctx.has_jpeg_artifacts:
                if cm > 25: ev.append(Evidence('pixels', f'Pattern beyond JPEG ({cm:.2f})', 0.4, cm, (0, 15)))
            else:
                if cm > 10: ev.append(Evidence('pixels', f'Strong checkerboard ({cm:.2f})', 0.65, cm, (0, 4)))
                elif cm > 7: ev.append(Evidence('pixels', f'Checkerboard ({cm:.2f})', 0.5, cm, (0, 4)))
        all_evidence.extend(ev); component_scores['pixels'] = self._cat(ev)

        component_scores['temporal'] = None  # N/A for images
        meta_ev = []
        w, h = ctx.width, ctx.height
        if w == h and w in [512, 768, 1024, 2048]:
            meta_ev.append(Evidence('metadata', 
                f'AI-typical square resolution {w}x{h}', 0.35, 0, (0, 0)))
        # Check file name hints (optional)
        all_evidence.extend(meta_ev)
        component_scores['metadata'] = self._cat(meta_ev)
        # DEEP
        ev = []
        l4s = deep_metrics.get('layer4_sparsity', [])
        if l4s:
            m = float(np.mean(l4s))
            if m > 0.5: ev.append(Evidence('deep', f'High sparsity ({m:.3f})', 0.3, m, (0.1, 0.4)))
        all_evidence.extend(ev); component_scores['deep'] = self._cat(ev)

        # AGGREGATE with context adjustments
        weights = {'frequency': 0.20, 'noise': 0.18, 'ela': 0.12, 'srm': 0.10,
                   'color': 0.10, 'texture': 0.12, 'pixels': 0.10, 'deep': 0.08}
        weighted = sum(component_scores.get(k, 0) * w for k, w in weights.items() if component_scores.get(k) is not None)

        ai_ev = [e for e in all_evidence if e.direction == 'ai']
        strong = [e for e in ai_ev if e.strength >= 0.4]
        indep_cats = len(set(e.category for e in strong))

        if indep_cats >= 4: weighted = min(100, weighted * 1.25)
        elif indep_cats >= 3: weighted = min(100, weighted * 1.15)
        elif indep_cats >= 2: weighted = min(100, weighted * 1.05)

        if len(ai_ev) == 0: weighted = min(weighted, 15)

        # Context reductions
        if ctx.is_heavily_compressed: weighted *= 0.7
        if ctx.is_small: weighted *= 0.8
        if ctx.is_document_photo: weighted *= 0.75

        # Confidence
        if not ai_ev: confidence = 25.0
        elif indep_cats >= 4: confidence = min(75, 55 + indep_cats * 5)
        elif indep_cats >= 3: confidence = min(70, 50 + indep_cats * 5)
        else: confidence = min(60, 35 + len(ai_ev) * 5)
        confidence = min(confidence, 80)
        if ctx.is_heavily_compressed: confidence = min(confidence, 55)
        if ctx.is_small: confidence = min(confidence, 60)
        if ctx.is_document_photo: confidence = min(confidence, 50)

        score = float(min(100, max(0, weighted)))
        confidence = float(min(90, max(15, confidence)))
        verdict = self._verdict(score, confidence)
        interp = self._interpret(score, confidence)

        return score, confidence, verdict, component_scores, all_evidence, interp

    def _cat(self, ev_list):
        if not ev_list: return 0.0
        ai = [e for e in ev_list if e.direction == 'ai']
        if not ai: return 0.0
        mx = max(e.strength for e in ai)
        boost = 1.2 if len(ai) >= 3 else 1.1 if len(ai) >= 2 else 1.0
        dev = max(ai, key=lambda e: e.deviation_ratio)
        df = min(1.5, 1.0 + dev.deviation_ratio * 0.3)
        return min(100.0, max(0.0, mx * boost * df * 100))

    def _verdict(self, score, confidence):
        ub = config.uncertainty_band
        ctx = self.ctx
        tag = " [JPEG]" if ctx.is_heavily_compressed else " [Small]" if ctx.is_small else ""
        if confidence < 35: return f"🔍 LOW CONFIDENCE{tag} ({score:.0f}±{ub:.0f})"
        elif score >= 60: return f"🤖 LIKELY AI ({score:.0f}±{ub:.0f})"
        elif score >= 45: return f"⚠️ PROBABLY AI ({score:.0f}±{ub:.0f})"
        elif score >= 30: return f"🤔 SUSPICIOUS{tag} ({score:.0f}±{ub:.0f})"
        elif score >= 15: return f"✅ LIKELY REAL{tag} ({score:.0f}±{ub:.0f})"
        else: return f"✅ REAL{tag} ({score:.0f}±{ub:.0f})"

    def _interpret(self, score, confidence):
        ctx = self.ctx
        issues = []
        if ctx.is_heavily_compressed: issues.append("heavy JPEG compression")
        if ctx.is_small: issues.append("small image size")
        if ctx.is_document_photo: issues.append("document/ID style")
        note = f" (affected by: {', '.join(issues)})" if issues else ""

        if confidence < 35:
            if issues:
                return {'conclusion': 'LIKELY REAL', 'icon': '✅',
                    'explanation': f"No AI indicators found, but quality limits analysis{note}.",
                    'recommendation': "Image appears genuine. Low confidence is due to image quality."}
            return {'conclusion': 'INCONCLUSIVE', 'icon': '❓',
                'explanation': "Insufficient data for reliable determination.",
                'recommendation': "Try with a higher quality version."}
        elif score >= 60:
            return {'conclusion': 'LIKELY AI-GENERATED', 'icon': '🤖',
                'explanation': "Multiple strong AI indicators detected.",
                'recommendation': "This media shows significant signs of AI generation."}
        elif score >= 45:
            return {'conclusion': 'PROBABLY AI', 'icon': '⚠️',
                'explanation': f"Several AI indicators detected{note}.",
                'recommendation': "Treat with caution."}
        elif score >= 30:
            return {'conclusion': 'SUSPICIOUS', 'icon': '🤔',
                'explanation': f"Some unusual patterns{note}. Could be AI or edited.",
                'recommendation': "Cannot confirm authenticity."}
        elif score >= 15:
            return {'conclusion': 'LIKELY REAL', 'icon': '✅',
                'explanation': f"Only minor anomalies{note}, consistent with authentic media.",
                'recommendation': "Appears genuine."}
        else:
            return {'conclusion': 'REAL', 'icon': '✅',
                'explanation': f"No significant AI indicators{note}.",
                'recommendation': "This media appears authentic."}

# ============================================================================
# CELL 20: Upload Media
# ============================================================================
from google.colab import files as colab_files
print("📁 Upload IMAGE or VIDEO:")
print("   Images: JPG, PNG, WebP, BMP, TIFF, GIF")
print("   Videos: MP4, AVI, MOV, MKV, WebM")
uploaded = colab_files.upload()
media_path = list(uploaded.keys())[0]
file_type = detect_file_type(media_path)
print(f"✅ Uploaded: {media_path} ({file_type.upper()})")

# ============================================================================
# CELL 21: Main Detector (Routes video/image to separate engines)
# ============================================================================

class Detector:
    def __init__(self):
        self.freq = FrequencyAnalyzer()
        self.noise = NoiseAnalyzer()
        self.ela = ELAAnalyzer()
        self.srm = SRMAnalyzer()
        self.color = ColorAnalyzer()
        self.texture = TextureAnalyzer()
        self.pixel = PixelAnalyzer()
        self.compress = CompressionAnalyzer()
        self.deep = DeepFeatureAnalyzer()
        self.temporal = TemporalAnalyzer()
        self.meta = MetadataAnalyzer()
        self.img_compress = ImageCompressionDetector()
        self.result = None

    def run(self, file_path):
        ftype = detect_file_type(file_path)
        if ftype == 'video':
            return self._run_video(file_path)
        elif ftype == 'image':
            return self._run_image(file_path)
        else:
            print("❌ Unsupported file type!")
            return None, None

    # ===== VIDEO PATH (ORIGINAL - unchanged) =====
    def _run_video(self, video_path):
        t0 = time.time()
        print("=" * 60)
        print("🎬 AI VIDEO FORENSIC ANALYSIS")
        print("=" * 60)

        print("\n📦 Extracting frames...")
        frames, info = extract_frames(video_path, config.max_frames, config.max_dimension)
        print(f"   {info['width']}x{info['height']} | {info['fps']:.1f}fps | {info['duration']:.1f}s | {len(frames)} frames")

        if len(frames) < 2:
            print("❌ Not enough frames!")
            return None, None

        print("\n🔬 Per-frame analysis...")
        metrics = defaultdict(list)
        analyzers = [self.freq, self.noise, self.ela, self.srm, self.color, self.texture, self.pixel, self.compress]
        for frame in tqdm(frames, desc="Analyzing"):
            for a in analyzers:
                try:
                    r = a.analyze(frame)
                    for k, v in r.items():
                        if isinstance(v, (int, float, np.floating)) and np.isfinite(v):
                            metrics[k].append(float(v))
                except: pass

        print("\n🧠 Deep features...")
        deep_metrics = defaultdict(list)
        step = max(1, len(frames) // config.deep_feature_frames)
        for i in range(0, len(frames), step):
            try:
                r = self.deep.analyze(frames[i])
                for k, v in r.items():
                    if isinstance(v, (int, float, np.floating)) and np.isfinite(v):
                        deep_metrics[k].append(float(v))
            except: pass

        print("\n⏱️ Temporal analysis...")
        temp = self.temporal.analyze(frames)
        meta_flags = self.meta.analyze(video_path, info)

        print("\n📊 Scoring...")
        scorer = VideoScoringEngine()
        score, conf, verdict, comp, evidence = scorer.score(dict(metrics), temp, meta_flags, dict(deep_metrics))

        elapsed = time.time() - t0
        self.result = {
            'score': score, 'confidence': conf, 'verdict': verdict,
            'components': comp, 'evidence': evidence, 'info': info,
            'elapsed': elapsed, 'is_image': False,
            'ub_low': max(0, score - config.uncertainty_band),
            'ub_high': min(100, score + config.uncertainty_band),
        }
        self._frames = frames
        self._report_video()
        self._visualize()
        return score, verdict

    # ===== IMAGE PATH (NEW - JPEG-aware) =====
    def _run_image(self, image_path):
        t0 = time.time()
        print("=" * 60)
        print("🖼️ AI IMAGE FORENSIC ANALYSIS")
        print("=" * 60)

        print("\n📦 Loading image...")
        frames, info = load_image(image_path, config.max_dimension)
        print(f"   {info['width']}x{info['height']} | Single image")

        print("\n🔍 Analyzing compression...")
        comp_info = self.img_compress.analyze(frames[0], image_path)
        ctx = build_image_context(image_path, info, comp_info)

        notes = []
        if ctx.is_heavily_compressed: notes.append(f"Heavy JPEG (quality ~{ctx.jpeg_quality_estimate:.0f})")
        if ctx.is_small: notes.append(f"Small ({ctx.width}x{ctx.height})")
        if ctx.is_document_photo: notes.append("Document photo")
        if notes: print(f"   ⚠️ Context: {', '.join(notes)}")
        else: print(f"   ✓ Good quality")

        print("\n🔬 Analyzing...")
        metrics = defaultdict(list)
        analyzers = [self.freq, self.noise, self.ela, self.srm, self.color, self.texture, self.pixel, self.compress]
        for a in analyzers:
            try:
                r = a.analyze(frames[0])
                for k, v in r.items():
                    if isinstance(v, (int, float, np.floating)) and np.isfinite(v):
                        metrics[k].append(float(v))
            except: pass

        print("\n🧠 Deep features...")
        deep_metrics = defaultdict(list)
        try:
            r = self.deep.analyze(frames[0])
            for k, v in r.items():
                if isinstance(v, (int, float, np.floating)) and np.isfinite(v):
                    deep_metrics[k].append(float(v))
        except: pass

        print("\n📊 Scoring...")
        scorer = ImageScoringEngine(ctx)
        score, conf, verdict, comp, evidence, interp = scorer.score(dict(metrics), dict(deep_metrics))

        elapsed = time.time() - t0
        self.result = {
            'score': score, 'confidence': conf, 'verdict': verdict,
            'components': comp, 'evidence': evidence, 'info': info,
            'elapsed': elapsed, 'is_image': True, 'interpretation': interp,
            'ub_low': max(0, score - config.uncertainty_band),
            'ub_high': min(100, score + config.uncertainty_band),
            'context': {'is_small': ctx.is_small, 'is_heavily_compressed': ctx.is_heavily_compressed,
                        'is_document_photo': ctx.is_document_photo, 'jpeg_quality': ctx.jpeg_quality_estimate}
        }
        self._frames = frames
        self._report_image(interp, ctx)
        self._visualize()
        return score, verdict

    def _report_video(self):
        """ORIGINAL video report - unchanged."""
        r = self.result
        print("\n" + "=" * 60)
        print("📋 FORENSIC REPORT")
        print("=" * 60)
        print(f"\n🎬 {r['info']['width']}x{r['info']['height']} | {r['info']['fps']:.1f}fps | {r['info']['duration']:.1f}s")
        print(f"⏱️ {r['elapsed']:.1f}s")
        print(f"\n{'='*50}")
        print(f"  🎯 SCORE: {r['score']:.1f}/100 (range: {r['ub_low']:.0f}–{r['ub_high']:.0f})")
        print(f"  🎲 CONFIDENCE: {r['confidence']:.0f}%")
        print(f"  📌 {r['verdict']}")
        print(f"{'='*50}")
        print("\n📊 Components:")
        labels = {'frequency': '🔊 Frequency', 'noise': '📡 Noise', 'ela': '🔬 ELA', 'srm': '🧪 SRM',
                  'color': '🎨 Color', 'texture': '📐 Texture', 'pixels': '🔲 Pixels', 'temporal': '⏱️ Temporal',
                  'metadata': '📋 Metadata', 'deep': '🧠 Deep'}
        for key, label in labels.items():
            s = r['components'].get(key, 0)
            bar = '█' * int(s / 5) + '░' * (20 - int(s / 5))
            icon = '🔴' if s >= 50 else '🟡' if s >= 25 else '🟢'
            print(f"  {icon} {label:16s} [{bar}] {s:.0f}")
        ai_ev = [e for e in r['evidence'] if e.direction == 'ai' and e.strength >= 0.25]
        if ai_ev:
            print(f"\n🔑 KEY EVIDENCE:")
            ai_ev.sort(key=lambda e: e.strength, reverse=True)
            for ev in ai_ev:
                dots = '●' * int(ev.strength * 5) + '○' * (5 - int(ev.strength * 5))
                print(f"  [{dots}] {ev.description}")
                if ev.expected_range != (0, 0):
                    print(f"         Measured: {ev.value:.4f} (real: {ev.expected_range[0]:.3f}–{ev.expected_range[1]:.3f})")
            devs = [ev.deviation_ratio for ev in ai_ev]
            print(f"\n  📐 Average deviation: {np.mean(devs):.2f}x outside range")
            print(f"  🎯 Categories with AI signals: {len(set(ev.category for ev in ai_ev))}")
        else:
            print("\n  ✅ No strong AI indicators found")
        print(f"\n{'─'*50}")
        print("⚠️ Probabilistic analysis, not definitive proof")

    def _report_image(self, interp, ctx):
        """Image-specific report with clear interpretation."""
        r = self.result
        print("\n" + "=" * 60)
        print("📋 IMAGE FORENSIC REPORT")
        print("=" * 60)

        # Clear conclusion first
        print(f"\n╔{'═'*56}╗")
        print(f"║  {interp['icon']}  {interp['conclusion']:^51} ║")
        print(f"╚{'═'*56}╝")
        print(f"\n📝 {interp['explanation']}")
        print(f"💡 {interp['recommendation']}")

        # Technical details
        print(f"\n{'─'*60}")
        print(f"🖼️ {r['info']['width']}x{r['info']['height']} | ⏱️ {r['elapsed']:.1f}s")
        if ctx.is_heavily_compressed: print(f"   ⚠️ JPEG quality ~{ctx.jpeg_quality_estimate:.0f}")
        if ctx.is_small: print(f"   ⚠️ Small image")
        if ctx.is_document_photo: print(f"   ⚠️ Document photo")
        print(f"🎯 Score: {r['score']:.1f}/100 ({r['ub_low']:.0f}–{r['ub_high']:.0f}) | Confidence: {r['confidence']:.0f}%")

        print("\n📊 Components:")
        labels = {'frequency': '🔊 Freq', 'noise': '📡 Noise', 'ela': '🔬 ELA', 'srm': '🧪 SRM',
                  'color': '🎨 Color', 'texture': '📐 Tex', 'pixels': '🔲 Pix',
                  'temporal': '⏱️ Temp', 'metadata': '📋 Meta', 'deep': '🧠 Deep'}
        for key, label in labels.items():
            s = r['components'].get(key)
            if s is None:
                print(f"  ⚪ {label:12s} [{'─'*20}] N/A")
            else:
                bar = '█' * int(s / 5) + '░' * (20 - int(s / 5))
                icon = '🔴' if s >= 50 else '🟡' if s >= 25 else '🟢'
                print(f"  {icon} {label:12s} [{bar}] {s:.0f}")

        ai_ev = [e for e in r['evidence'] if e.direction == 'ai' and e.strength >= 0.25]
        if ai_ev:
            print(f"\n🔑 Evidence:")
            ai_ev.sort(key=lambda e: e.strength, reverse=True)
            for ev in ai_ev[:5]:
                dots = '●' * int(ev.strength * 5) + '○' * (5 - int(ev.strength * 5))
                print(f"  [{dots}] {ev.description}")
        else:
            print("\n  ✅ No AI indicators detected")

    def _visualize(self):
        r = self.result
        frames = self._frames
        is_image = r.get('is_image', False)
        try:
            fig = plt.figure(figsize=(22, 14))
            title = f"[{'IMAGE' if is_image else 'VIDEO'}] Score: {r['score']:.1f}/100 | Conf: {r['confidence']:.0f}% | {r['verdict']}"
            fig.suptitle(title, fontsize=11, fontweight='bold', y=0.99)

            valid_cats = [(k, v) for k, v in r['components'].items() if v is not None]
            cats = [k for k, v in valid_cats]
            vals = [v for k, v in valid_cats]
            lbl_map = {'frequency':'Freq','noise':'Noise','ela':'ELA','srm':'SRM','color':'Color',
                       'texture':'Tex','pixels':'Pix','temporal':'Temp','metadata':'Meta','deep':'Deep'}
            lbls = [lbl_map.get(c, c) for c in cats]

            ax = fig.add_subplot(2, 3, 1, polar=True)
            angles = np.linspace(0, 2*np.pi, len(lbls), endpoint=False).tolist()
            vp = vals + [vals[0]]; ap = angles + [angles[0]]
            ax.set_theta_offset(np.pi/2); ax.set_theta_direction(-1)
            ax.fill(ap, vp, alpha=0.25, color='red'); ax.plot(ap, vp, 'o-', color='red', lw=2)
            ax.set_xticks(angles); ax.set_xticklabels(lbls, size=7); ax.set_ylim(0, 100)
            ax.set_title('Scores', fontweight='bold', pad=20)

            ax = fig.add_subplot(2, 3, 2)
            colors = ['#ff4444' if v>=50 else '#ffaa00' if v>=25 else '#44bb44' for v in vals]
            bars = ax.barh(lbls, vals, color=colors, edgecolor='gray', lw=0.5)
            ax.set_xlim(0, 100); ax.axvline(50, color='red', ls='--', alpha=0.3)
            ax.set_title('Components', fontweight='bold')
            for bar, v in zip(bars, vals):
                ax.text(min(v+1, 90), bar.get_y()+bar.get_height()/2, f'{v:.0f}', va='center', fontsize=8)

            ax = fig.add_subplot(2, 3, 3)
            ax.imshow(cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB))
            ax.set_title('Input' if is_image else 'Sample Frame', fontweight='bold'); ax.axis('off')

            gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY).astype(np.float64)

            ax = fig.add_subplot(2, 3, 4)
            ax.imshow(np.log1p(np.abs(fftpack.fftshift(fftpack.fft2(gray)))), cmap='hot')
            ax.set_title('FFT', fontweight='bold'); ax.axis('off')

            ax = fig.add_subplot(2, 3, 5)
            try:
                ela_r = self.ela.analyze(frames[0])
                ax.imshow(ela_r.get('ela_map', np.zeros_like(gray)), cmap='jet')
                ax.set_title('ELA', fontweight='bold')
            except: ax.set_title('ELA (failed)')
            ax.axis('off')

            ax = fig.add_subplot(2, 3, 6)
            noise = gray - cv2.GaussianBlur(gray, (5,5), 1.0)
            vm = max(abs(np.percentile(noise, 2)), abs(np.percentile(noise, 98)), 1)
            ax.imshow(noise, cmap='RdBu', vmin=-vm, vmax=vm)
            ax.set_title(f'Noise (σ={np.std(noise):.2f})', fontweight='bold'); ax.axis('off')

            plt.tight_layout(rect=[0,0,1,0.92])
            plt.savefig('detection_overview.png', dpi=150, bbox_inches='tight'); plt.show()
            print("   ✅ Saved: detection_overview.png")
        except Exception as e:
            print(f"   ⚠️ Viz failed: {e}")

        # Evidence plot
        try:
            ai_ev = [e for e in r['evidence'] if e.direction == 'ai' and e.strength >= 0.2]
            if ai_ev:
                fig, axes = plt.subplots(1, 2, figsize=(16, 6))
                fig.suptitle('Evidence', fontweight='bold')
                cat_d = defaultdict(float)
                for e in ai_ev: cat_d[e.category] = max(cat_d[e.category], e.strength)
                names = list(cat_d.keys()); strs = [cat_d[n] for n in names]
                cols = ['#ff4444' if s>=0.5 else '#ffaa00' if s>=0.3 else '#44bb44' for s in strs]
                axes[0].barh(names, strs, color=cols, alpha=0.8); axes[0].set_xlim(0, 1); axes[0].set_title('By Category')
                ai_ev.sort(key=lambda e: e.strength, reverse=True)
                top = ai_ev[:8]
                descs = [e.description[:45] for e in top]; ts = [e.strength for e in top]
                ec = ['#ff4444' if s>=0.5 else '#ffaa00' for s in ts]
                axes[1].barh(range(len(descs)), ts, color=ec, alpha=0.8)
                axes[1].set_yticks(range(len(descs))); axes[1].set_yticklabels(descs, fontsize=7)
                axes[1].set_xlim(0, 1); axes[1].set_title('Top Evidence'); axes[1].invert_yaxis()
                plt.tight_layout(); plt.savefig('evidence.png', dpi=150, bbox_inches='tight'); plt.show()
                print("   ✅ Saved: evidence.png")
        except Exception as e:
            print(f"   ⚠️ Evidence plot failed: {e}")

    def cleanup(self):
        self.deep.cleanup()
        if hasattr(self, '_frames'): del self._frames
        gc.collect()

# ============================================================================
# RUN
# ============================================================================
detector = Detector()
try:
    score, verdict = detector.run(media_path)
    if score is not None:
        print("\n" + "🔮" * 30)
        r = detector.result
        if r.get('is_image') and 'interpretation' in r:
            interp = r['interpretation']
            print(f"\n  {interp['icon']}  {interp['conclusion']}")
            print(f"  📊 Score: {score:.0f}/100")
            print(f"  💡 {interp['recommendation']}")
        else:
            print(f"\n  🎯 SCORE: {score:.1f}/100")
            print(f"  📌 {verdict}")
        print(f"\n" + "🔮" * 30)
except Exception as e:
    print(f"\n❌ Failed: {e}")
    traceback.print_exc()
finally:
    detector.cleanup()

print("\n✅ Analysis Complete!")