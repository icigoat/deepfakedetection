from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse, HttpResponse
from django.core.files.base import ContentFile
from django.views.decorators.csrf import csrf_exempt
from .models import Detection, AnonymousUser
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

            # Get or create anonymous user
            from .user_tracking import get_or_create_anonymous_user
            anonymous_user = get_or_create_anonymous_user(request)

            detection = Detection(
                file=uploaded_file,
                file_type='image' if info_data.get('is_image') else 'video',
                score=float(result['score']),
                confidence=float(result['confidence']),
                verdict=str(result['verdict']),
                components=components,
                evidence=evidence_list,
                info=final_info,
                anonymous_user=anonymous_user,
            )
            detection.save()
            
            # Update user's detection count
            anonymous_user.detection_count += 1
            anonymous_user.save(update_fields=['detection_count'])

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

            # Check if anime slugs are enabled
            from .models import SiteSettings
            settings = SiteSettings.get_settings()
            
            if settings.use_anime_slugs and detection.slug:
                return JsonResponse({
                    'success': True,
                    'use_slugs': True,
                    'detection_slug': detection.slug
                })
            else:
                return JsonResponse({
                    'success': True,
                    'use_slugs': False,
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


def result_by_slug(request, slug):
    """Result view using anime slug"""
    detection = get_object_or_404(Detection, slug=slug)
    
    # Increment view count
    detection.increment_views()
    
    # Get anonymous user
    from .user_tracking import get_or_create_anonymous_user
    anonymous_user = get_or_create_anonymous_user(request)
    
    # Parse evidence
    evidence_list = []
    if detection.evidence:
        if isinstance(detection.evidence, dict):
            # If evidence is a dict
            for category, items in detection.evidence.items():
                for item in items:
                    evidence_list.append({
                        'category': category,
                        'description': item.get('description', ''),
                        'strength': item.get('strength', 0)
                    })
        elif isinstance(detection.evidence, list):
            # If evidence is already a list
            evidence_list = detection.evidence
    
    # Generate share URL
    share_url = request.build_absolute_uri()
    
    context = {
        'detection': detection,
        'components': detection.components,
        'evidence_list': evidence_list,
        'share_url': share_url,
    }
    
    response = render(request, 'detector/result.html', context)
    
    # Set cookie for user tracking
    from .user_tracking import set_anonymous_user_cookie
    response = set_anonymous_user_cookie(response, anonymous_user)
    
    return response


def result_by_id(request, detection_id):
    """Result view using numeric ID (fallback)"""
    detection = get_object_or_404(Detection, id=detection_id)
    
    # Parse evidence
    evidence_list = []
    if detection.evidence:
        if isinstance(detection.evidence, dict):
            # If evidence is a dict
            for category, items in detection.evidence.items():
                for item in items:
                    evidence_list.append({
                        'category': category,
                        'description': item.get('description', ''),
                        'strength': item.get('strength', 0)
                    })
        elif isinstance(detection.evidence, list):
            # If evidence is already a list
            evidence_list = detection.evidence
    
    context = {
        'detection': detection,
        'components': detection.components,
        'evidence_list': evidence_list,
    }
    return render(request, 'detector/result.html', context)

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



def my_detections(request):
    """Show user's detection history"""
    from .user_tracking import get_or_create_anonymous_user
    
    anonymous_user = get_or_create_anonymous_user(request)
    detections = Detection.objects.filter(anonymous_user=anonymous_user).order_by('-created_at')
    
    context = {
        'detections': detections,
        'user': anonymous_user,
    }
    
    response = render(request, 'detector/my_detections.html', context)
    
    # Set cookie for user tracking
    from .user_tracking import set_anonymous_user_cookie
    response = set_anonymous_user_cookie(response, anonymous_user)
    
    return response


def share_detection(request, slug):
    """Track share and redirect"""
    detection = get_object_or_404(Detection, slug=slug)
    detection.increment_shares()
    
    # Redirect to result page
    return redirect('detector:result', slug=slug)


def stats_dashboard(request):
    """Admin statistics dashboard"""
    from django.db.models import Count, Avg
    from django.utils import timezone
    from datetime import timedelta
    
    # Check if user is admin
    if not request.user.is_staff:
        return redirect('detector:index')
    
    now = timezone.now()
    today = now.date()
    week_ago = now - timedelta(days=7)
    month_ago = now - timedelta(days=30)
    
    # Statistics
    total_detections = Detection.objects.count()
    total_users = AnonymousUser.objects.count()
    
    detections_today = Detection.objects.filter(created_at__date=today).count()
    detections_week = Detection.objects.filter(created_at__gte=week_ago).count()
    detections_month = Detection.objects.filter(created_at__gte=month_ago).count()
    
    users_today = AnonymousUser.objects.filter(first_visit__date=today).count()
    users_week = AnonymousUser.objects.filter(first_visit__gte=week_ago).count()
    users_month = AnonymousUser.objects.filter(first_visit__gte=month_ago).count()
    
    # Verdict breakdown
    verdict_stats = Detection.objects.values('verdict').annotate(count=Count('id')).order_by('-count')
    
    # Average scores
    avg_score = Detection.objects.aggregate(Avg('score'))['score__avg'] or 0
    avg_confidence = Detection.objects.aggregate(Avg('confidence'))['confidence__avg'] or 0
    
    # Most active users
    top_users = AnonymousUser.objects.order_by('-detection_count')[:10]
    
    # Recent detections
    recent_detections = Detection.objects.select_related('anonymous_user').order_by('-created_at')[:20]
    
    context = {
        'total_detections': total_detections,
        'total_users': total_users,
        'detections_today': detections_today,
        'detections_week': detections_week,
        'detections_month': detections_month,
        'users_today': users_today,
        'users_week': users_week,
        'users_month': users_month,
        'verdict_stats': verdict_stats,
        'avg_score': avg_score,
        'avg_confidence': avg_confidence,
        'top_users': top_users,
        'recent_detections': recent_detections,
    }
    
    return render(request, 'detector/stats_dashboard.html', context)


def manifest(request):
    """Serve manifest.json with proper MIME type"""
    manifest_path = os.path.join(os.path.dirname(__file__), 'static', 'manifest.json')
    with open(manifest_path, 'r') as f:
        manifest_data = f.read()
    return HttpResponse(manifest_data, content_type='application/manifest+json')


def service_worker(request):
    """Serve service-worker.js with proper MIME type"""
    sw_path = os.path.join(os.path.dirname(__file__), 'static', 'service-worker.js')
    with open(sw_path, 'r') as f:
        sw_data = f.read()
    return HttpResponse(sw_data, content_type='application/javascript')
