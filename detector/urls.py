from django.urls import path
from . import views

app_name = 'detector'

urlpatterns = [
    path('', views.index, name='index'),
    path('analyze/', views.analyze, name='analyze'),
    path('results/<path:slug>/', views.result_by_slug, name='result'),  # Use path: to allow slashes
    
    # New features
    path('my-detections/', views.my_detections, name='my_detections'),
    path('share/<path:slug>/', views.share_detection, name='share'),
    path('delete/<path:slug>/', views.delete_detection, name='delete'),
    path('stats/', views.stats_dashboard, name='stats'),
    
    # PWA files with proper MIME types
    path('manifest.json', views.manifest, name='manifest'),
    path('service-worker.js', views.service_worker, name='service_worker'),
]
