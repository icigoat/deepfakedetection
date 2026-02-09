import cv2
import numpy as np
from scipy import fftpack
from scipy.stats import kurtosis
from scipy.signal import medfilt
from PIL import Image
import io
import os
from dataclasses import dataclass


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
                'spectral_slope': spectral_slope,
                'spectral_fit_r2': fit_r2,
                'spectral_residual_std': residual_std,
                'hf_energy_ratio': hf_energy_ratio,
                'spectral_flatness': flatness,
                'spectral_anomalies': float(anomalies),
                'radial_profile': profile,
                'magnitude_spectrum': mag_log
            }
        except:
            return {
                'spectral_slope': 0.0, 'spectral_fit_r2': 0.0,
                'spectral_residual_std': 0.0, 'hf_energy_ratio': 0.0,
                'spectral_flatness': 0.0, 'spectral_anomalies': 0.0
            }


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
            
            flat = noise.flatten()[:10000]  # min_sample_size
            autocorrs = []
            for lag in [1, 2, 4, 8]:
                if len(flat) > lag + 10:
                    autocorrs.append(Stats.safe_corr(flat[:-lag], flat[lag:]))
            noise_ac = float(np.mean(autocorrs)) if autocorrs else 0.0
            
            b_ch, g_ch, r_ch = cv2.split(frame.astype(np.float64))
            sz = min(10000, b_ch.size)
            nr = (r_ch - cv2.GaussianBlur(r_ch, (5,5), 1.0)).flatten()[:sz]
            ng = (g_ch - cv2.GaussianBlur(g_ch, (5,5), 1.0)).flatten()[:sz]
            nb = (b_ch - cv2.GaussianBlur(b_ch, (5,5), 1.0)).flatten()[:sz]
            cc = float(np.mean([abs(Stats.safe_corr(nr, ng)), abs(Stats.safe_corr(nr, nb)), abs(Stats.safe_corr(ng, nb))]))
            scale_ratio = float(np.std(noise_fine)) / (float(np.std(noise_coarse)) + 1e-10)
            
            return {
                'noise_std': float(np.std(noise)),
                'noise_uniformity': noise_uniformity,
                'noise_brightness_corr': float(nb_corr),
                'noise_spatial_cv': spatial_cv,
                'noise_autocorr': float(noise_ac),
                'cross_channel_corr': cc,
                'noise_scale_ratio': float(scale_ratio),
                'noise_kurtosis': Stats.safe_kurtosis(noise.flatten()),
                'noise_map': noise
            }
        except:
            return {
                'noise_std': 0.0, 'noise_uniformity': 0.0, 'noise_brightness_corr': 0.0,
                'noise_spatial_cv': 0.0, 'noise_autocorr': 0.0, 'cross_channel_corr': 0.0,
                'noise_scale_ratio': 0.0, 'noise_kurtosis': 0.0
            }


class ELAAnalyzer:
    def analyze(self, frame, qualities=None):
        if qualities is None:
            qualities = [75, 85, 95]  # config.ela_qualities
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


class ColorAnalyzer:
    def analyze(self, frame):
        try:
            b, g, r = cv2.split(frame.astype(np.float64))
            sz = min(10000, b.size)
            rf, gf, bf = r.flatten()[:sz], g.flatten()[:sz], b.flatten()[:sz]
            
            # Channel correlations
            rg = Stats.safe_corr(rf, gf)
            rb = Stats.safe_corr(rf, bf)
            gb = Stats.safe_corr(gf, bf)
            
            # Color entropy
            entropies = [Stats.safe_entropy(ch.flatten(), 256, (0, 256))
                        for ch in [b, g, r]]
            color_entropy = float(np.mean(entropies))
            
            # Edge alignment across channels
            edges = {}
            for nm, ch in [('r', r), ('g', g), ('b', b)]:
                edges[nm] = cv2.Canny(ch.astype(np.uint8), 50, 150).flatten()[:sz]
            ec = [Stats.safe_corr(edges[a].astype(float), edges[b_].astype(float))
                  for a, b_ in [('r','g'), ('r','b'), ('g','b')]]
            edge_alignment = float(np.mean(ec))
            
            # Histogram roughness
            roughness = []
            for ch in [b, g, r]:
                hist, _ = np.histogram(ch, bins=256, range=(0, 256))
                hd = np.diff(hist.astype(float))
                roughness.append(float(np.std(hd) / (np.mean(np.abs(hd)) + 1e-10)))
            
            # Color uniqueness
            sub = frame.reshape(-1, 3)
            if len(sub) > 10000:
                sub = sub[np.random.RandomState(42).choice(len(sub), 10000, replace=False)]
            uniq = len(np.unique(sub, axis=0)) / len(sub)
            
            # Saturation
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float64)
            sat = hsv[:, :, 1]
            
            return {
                'rg_correlation': float(rg),
                'rb_correlation': float(rb),
                'gb_correlation': float(gb),
                'color_entropy': color_entropy,
                'edge_alignment': edge_alignment,
                'color_hist_roughness': float(np.mean(roughness)),
                'color_uniqueness': float(uniq),
                'saturation_mean': float(np.mean(sat)),
                'saturation_std': float(np.std(sat)),
            }
        except:
            return {
                'rg_correlation': 0.0, 'rb_correlation': 0.0, 'gb_correlation': 0.0,
                'color_entropy': 0.0, 'edge_alignment': 0.0, 'color_hist_roughness': 0.0,
                'color_uniqueness': 0.0, 'saturation_mean': 0.0, 'saturation_std': 0.0
            }


class TextureAnalyzer:
    def analyze(self, frame):
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
            gu8 = gray.astype(np.uint8)
            
            # Edge detection at two thresholds
            et = cv2.Canny(gu8, 100, 200)
            el = cv2.Canny(gu8, 30, 60)
            edt = float(np.mean(et > 0))
            edl = float(np.mean(el > 0))
            
            # Gradient analysis
            gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gmag = np.sqrt(gx**2 + gy**2)
            gdir = np.arctan2(gy, gx)
            de = Stats.safe_entropy(gdir.flatten(), 36, (-np.pi, np.pi))
            
            # Local Binary Pattern (LBP)
            h, w = gu8.shape
            pad = np.pad(gu8, 1, mode='edge').astype(np.int16)
            center = pad[1:-1, 1:-1]
            offsets = [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]
            lbp = np.zeros((h, w), dtype=np.uint8)
            for bit, (dy, dx) in enumerate(offsets):
                nb = pad[1+dy:h+1+dy, 1+dx:w+1+dx]
                lbp |= ((nb >= center).astype(np.uint8) << (7 - bit))
            lbp_e = Stats.safe_entropy(lbp.flatten(), 256, (0, 256))
            
            # Edge sharpness
            edge_pix = gmag[et > 0]
            ecv = float(np.std(edge_pix) / (np.mean(edge_pix) + 1e-10)) if len(edge_pix) > 50 else 0.0
            
            # GLCM (Gray Level Co-occurrence Matrix)
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
                'edge_density': edt,
                'edge_ratio': float(edt / (edl + 1e-10)),
                'gradient_kurtosis': Stats.safe_kurtosis(gmag.flatten()),
                'direction_uniformity': float(de / 5.17),
                'lbp_entropy': float(lbp_e),
                'edge_sharpness_cv': ecv,
                'glcm_energy': float(np.sum(glcm**2)),
                'glcm_contrast': float(np.sum((ii - jj)**2 * glcm)),
                'glcm_homogeneity': float(np.sum(glcm / (1 + np.abs(ii - jj)))),
            }
        except:
            return {
                'edge_density': 0.0, 'edge_ratio': 0.0, 'gradient_kurtosis': 0.0,
                'direction_uniformity': 0.0, 'lbp_entropy': 0.0, 'edge_sharpness_cv': 0.0,
                'glcm_energy': 0.0, 'glcm_contrast': 0.0, 'glcm_homogeneity': 0.0
            }


class SRMAnalyzer:
    """Spatial Rich Model - detects manipulation artifacts"""
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
            return {
                'srm_energy_mean': float(np.mean(energies)),
                'srm_energy_std': float(np.std(energies)),
                'srm_kurtosis_mean': float(np.mean(kurts))
            }
        except:
            return {'srm_energy_mean': 0.0, 'srm_energy_std': 0.0, 'srm_kurtosis_mean': 0.0}


class PixelAnalyzer:
    """Detects pixel-level artifacts like checkerboard patterns"""
    def analyze(self, frame):
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
            
            # Checkerboard detection at multiple scales
            checker_scores = []
            for s in [1, 2, 4, 8]:
                sz = 2 * s
                k = np.ones((sz, sz), dtype=np.float64)
                k[:s, :s] = 1
                k[s:, s:] = 1
                k[:s, s:] = -1
                k[s:, :s] = -1
                k /= (s * s)
                checker_scores.append(float(np.mean(np.abs(cv2.filter2D(gray, cv2.CV_64F, k)))))
            
            # Grid detection
            gh = np.array([[1,1,1],[-2,-2,-2],[1,1,1]], dtype=np.float64) / 6
            gv = np.array([[1,-2,1],[1,-2,1],[1,-2,1]], dtype=np.float64) / 6
            grid = (np.mean(np.abs(cv2.filter2D(gray, cv2.CV_64F, gh))) +
                    np.mean(np.abs(cv2.filter2D(gray, cv2.CV_64F, gv)))) / 2
            
            # Histogram analysis
            hist, _ = np.histogram(gray, 256, (0, 256))
            nz = np.where(hist > 0)[0]
            gaps = 0.0
            if len(nz) > 1:
                ir = hist[nz[0]:nz[-1]+1]
                gaps = float(np.sum(ir == 0) / len(ir))
            
            pe = Stats.safe_entropy(gray.flatten(), 256, (0, 256))
            hd = np.diff(hist.astype(float))
            hr = float(np.std(hd) / (np.mean(np.abs(hd)) + 1e-10))
            
            return {
                'checker_max': float(max(checker_scores)),
                'checker_mean': float(np.mean(checker_scores)),
                'grid_score': float(grid),
                'histogram_gaps': gaps,
                'pixel_entropy': pe,
                'hist_roughness': hr
            }
        except:
            return {
                'checker_max': 0.0, 'checker_mean': 0.0, 'grid_score': 0.0,
                'histogram_gaps': 0.0, 'pixel_entropy': 0.0, 'hist_roughness': 0.0
            }


class CompressionAnalyzer:
    """Analyzes compression artifacts (for videos and general use)"""
    def analyze(self, frame):
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
            h, w = gray.shape
            
            # Block boundary detection
            boundary, interior = [], []
            lim = min(512, h, w)
            for i in range(1, lim):
                rd = np.mean(np.abs(gray[i, :lim] - gray[i-1, :lim]))
                (boundary if i % 8 == 0 else interior).append(rd)
            for j in range(1, lim):
                cd = np.mean(np.abs(gray[:lim, j] - gray[:lim, j-1]))
                (boundary if j % 8 == 0 else interior).append(cd)
            
            ratio = float(np.mean(boundary) / (np.mean(interior) + 1e-10)) if boundary and interior else 1.0
            
            # DCT consistency
            dct_stds = []
            for i in range(0, min(h-8, 256), 8):
                for j in range(0, min(w-8, 256), 8):
                    dct_stds.append(float(np.std(cv2.dct(gray[i:i+8, j:j+8]))))
                    if len(dct_stds) >= 64:
                        break
            
            dct_cv = float(np.std(dct_stds) / (np.mean(dct_stds) + 1e-10)) if dct_stds else 0.0
            
            return {
                'block_boundary_ratio': ratio,
                'dct_consistency': dct_cv
            }
        except:
            return {'block_boundary_ratio': 1.0, 'dct_consistency': 0.0}


class CompressionDetector:
    """Detects JPEG compression for IMAGES only (alias for backward compatibility)"""
    def analyze(self, frame, file_path=None):
        result = {
            'jpeg_quality_estimate': 100.0,
            'has_jpeg_artifacts': False,
            'is_heavily_compressed': False,
            'compression_type': 'unknown'
        }
        
        # FIX 1: Check file extension FIRST — PNG/BMP/TIFF are lossless
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
            
            # Detect 8x8 block boundaries
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
            
            # Estimate JPEG quality
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
    """Build context for IMAGE only"""
    ctx = MediaContext()
    ctx.width = info.get('width', 0)
    ctx.height = info.get('height', 0)
    ctx.is_small = max(ctx.width, ctx.height) < 512
    ctx.jpeg_quality_estimate = compression_info.get('jpeg_quality_estimate', 100)
    ctx.has_jpeg_artifacts = compression_info.get('has_jpeg_artifacts', False)
    ctx.is_heavily_compressed = compression_info.get('is_heavily_compressed', False)
    
    # FIX 3: Lossless formats are NEVER heavily compressed
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
