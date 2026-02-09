from django.contrib import admin
from .models import Detection


@admin.register(Detection)
class DetectionAdmin(admin.ModelAdmin):
    list_display = ['id', 'file_type', 'score', 'confidence', 'verdict', 'created_at']
    list_filter = ['file_type', 'created_at']
    search_fields = ['verdict']
    readonly_fields = ['created_at']
    ordering = ['-created_at']
    
    fieldsets = (
        ('File Information', {
            'fields': ('file', 'file_type', 'info')
        }),
        ('Analysis Results', {
            'fields': ('score', 'confidence', 'verdict', 'components', 'evidence')
        }),
        ('Visualizations', {
            'fields': ('fft_image', 'ela_image', 'noise_image')
        }),
        ('Metadata', {
            'fields': ('created_at',)
        }),
    )
