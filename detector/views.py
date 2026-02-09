from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.core.files.base import ContentFile
from django.views.decorators.csrf import csrf_exempt
from .models import Detection
from .detector import AIDetector
import cv2
import numpy as np
import json
import os
import tempfile
import traceback


def _make_json_safe(obj):
    """Recursively convert numpy types to Python native types for JSON storage.
    Skips numpy arrays that are images (2D/3D) — only converts scalars."""
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_safe(item) for item in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        val = float(obj)
        # Handle NaN/Inf which JSON also can't serialize
        if np.isnan(val) or np.isinf(val):
            return 0.0
        return val
    elif isinstance(obj, np.ndarray):
        # Don't convert image arrays — skip them
        return obj.tolist()
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return 0.0
    return obj


def index(request):
    return render(request, 'detector/index.html')


@csrf_exempt
def analyze(request):
    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = request.FILES['file']

        suffix = os.path.splitext(uploaded_file.name)[1]
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)

        try:
            for chunk in uploaded_file.chunks():
                temp_file.write(chunk)
            temp_file.close()

            # Check if it's a video and validate duration (20 second limit)
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v']
            if suffix.lower() in video_extensions:
                cap = cv2.VideoCapture(temp_file.name)
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    duration = frame_count / fps if fps > 0 else 0
                    cap.release()
                    
                    if duration > 20:
                        os.unlink(temp_file.name)
                        return JsonResponse({
                            'success': False,
                            'error': f'Video duration ({duration:.1f}s) exceeds 20 second limit. Please upload a shorter video.'
                        }, status=400)

            detector = AIDetector()
            result = detector.analyze(temp_file.name)

            # ── Save visualization images FIRST (before any conversion) ──
            viz = result.get('visualization', {})
            
            # Create detection record
            # Convert evidence objects to plain dicts
            evidence_list = []
            for e in result.get('evidence', []):
                if hasattr(e, 'category'):
                    # It's an Evidence object
                    evidence_list.append({
                        'category': str(e.category),
                        'description': str(e.description),
                        'strength': float(e.strength),
                        'value': float(e.value) if isinstance(e.value, (int, float, np.integer, np.floating)) else 0.0,
                    })
                elif isinstance(e, dict):
                    # Already a dict
                    evidence_list.append({
                        'category': str(e.get('category', '')),
                        'description': str(e.get('description', '')),
                        'strength': float(e.get('strength', 0)),
                        'value': float(e.get('value', 0)),
                    })

            # Make components JSON-safe (convert numpy types)
            components = _make_json_safe(result.get('components', {}))
            
            # Make info JSON-safe
            info_data = _make_json_safe(result.get('info', {}))
            
            # Make context JSON-safe
            context_data = _make_json_safe(result.get('context', {}))
            
            # Make interpretation JSON-safe
            user_interp = _make_json_safe(result.get('user_interpretation', {}))

            # Build final info dict with everything
            final_info = {
                **info_data,
                'user_interpretation': user_interp,
                'context': context_data,
            }

            detection = Detection(
                file=uploaded_file,
                file_type='image' if info_data.get('is_image') else 'video',
                score=float(result['score']),
                confidence=float(result['confidence']),
                verdict=str(result['verdict']),
                components=components,
                evidence=evidence_list,
                info=final_info,
            )
            detection.save()

            # ── Save visualization images (raw numpy arrays) ──
            for key in ['fft', 'ela', 'noise']:
                img_data = viz.get(key)
                if img_data is None:
                    continue

                # Ensure it's a numpy array
                if not isinstance(img_data, np.ndarray):
                    img_data = np.array(img_data, dtype=np.uint8)

                # Ensure correct dtype
                if img_data.dtype != np.uint8:
                    img_data = img_data.astype(np.uint8)

                success, buffer = cv2.imencode('.png', img_data)
                if not success:
                    print(f"Failed to encode {key} image")
                    continue

                img_file = ContentFile(buffer.tobytes())

                if key == 'fft':
                    detection.fft_image.save(f'fft_{detection.id}.png', img_file, save=False)
                elif key == 'ela':
                    detection.ela_image.save(f'ela_{detection.id}.png', img_file, save=False)
                elif key == 'noise':
                    detection.noise_image.save(f'noise_{detection.id}.png', img_file, save=False)

            detection.save()

            # Cleanup
            os.unlink(temp_file.name)
            detector.cleanup()

            return JsonResponse({
                'success': True,
                'detection_id': detection.id
            })

        except Exception as e:
            traceback.print_exc()
            try:
                os.unlink(temp_file.name)
            except:
                pass

            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=400)

    return JsonResponse({'success': False, 'error': 'Invalid request'}, status=400)


def result(request, detection_id):
    detection = get_object_or_404(Detection, id=detection_id)

    user_interp = None
    context_info = {}

    if isinstance(detection.info, dict):
        user_interp = detection.info.get('user_interpretation')
        context_info = detection.info.get('context', {})

    context = {
        'detection': detection,
        'evidence_list': detection.evidence or [],
        'components': detection.components or {},
        'user_interpretation': user_interp,
        'context_info': context_info,
    }

    return render(request, 'detector/result.html', context)