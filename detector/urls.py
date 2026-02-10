from django.urls import path
from . import views

app_name = 'detector'

urlpatterns = [
    path('', views.index, name='index'),
    # Both old and new URLs supported
    path('analyze/', views.analyze, name='analyze_old'),
    path('detection-working/analyze/', views.analyze, name='analyze'),
    path('result/<int:detection_id>/', views.result_by_id, name='result_old'),
    path('results/<path:slug>/', views.result_by_slug, name='result'),  # Use path: to allow slashes
    
    # New features
    path('my-detections/', views.my_detections, name='my_detections'),
    path('share/<path:slug>/', views.share_detection, name='share'),
    path('stats/', views.stats_dashboard, name='stats'),
]
