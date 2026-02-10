from django.contrib import admin
from django.utils.html import format_html
from django.urls import reverse, path
from django.shortcuts import redirect
from .models import Detection, SiteSettings, AnonymousUser


class CustomAdminSite(admin.AdminSite):
    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path('stats/', self.admin_view(self.stats_view), name='stats_dashboard'),
        ]
        return custom_urls + urls
    
    def stats_view(self, request):
        from django.shortcuts import redirect
        return redirect('detector:stats')


@admin.register(SiteSettings)
class SiteSettingsAdmin(admin.ModelAdmin):
    list_display = ['use_anime_slugs']
    fieldsets = (
        ('URL Settings', {
            'fields': ('use_anime_slugs',),
            'description': 'Toggle anime/motivational quote URLs on or off'
        }),
    )
    
    def has_add_permission(self, request):
        # Only allow one instance
        return not SiteSettings.objects.exists()
    
    def has_delete_permission(self, request, obj=None):
        # Don't allow deletion
        return False


@admin.register(AnonymousUser)
class AnonymousUserAdmin(admin.ModelAdmin):
    list_display = ['user_id_short', 'detection_count', 'ip_address', 'first_visit', 'last_visit', 'view_detections_link']
    list_filter = ['first_visit', 'last_visit']
    search_fields = ['user_id', 'ip_address', 'fingerprint']
    readonly_fields = ['user_id', 'fingerprint', 'first_visit', 'last_visit', 'detection_count', 'user_agent_display']
    
    fieldsets = (
        ('User Info', {
            'fields': ('user_id', 'fingerprint', 'ip_address')
        }),
        ('Activity', {
            'fields': ('detection_count', 'first_visit', 'last_visit')
        }),
        ('Browser Info', {
            'fields': ('user_agent_display',)
        }),
    )
    
    def user_id_short(self, obj):
        return f"{obj.user_id[:16]}..."
    user_id_short.short_description = "User ID"
    
    def user_agent_display(self, obj):
        return obj.user_agent or "Unknown"
    user_agent_display.short_description = "User Agent"
    
    def view_detections_link(self, obj):
        url = reverse('admin:detector_detection_changelist') + f'?anonymous_user__id__exact={obj.id}'
        return format_html('<a href="{}">View {} detections</a>', url, obj.detection_count)
    view_detections_link.short_description = "Detections"


@admin.register(Detection)
class DetectionAdmin(admin.ModelAdmin):
    list_display = ['id', 'slug_short', 'anonymous_user_link', 'verdict', 'score', 'confidence', 'view_count', 'share_count', 'created_at']
    list_filter = ['verdict', 'file_type', 'created_at', 'anonymous_user']
    search_fields = ['slug', 'verdict']
    readonly_fields = ['slug', 'created_at', 'view_count', 'share_count', 'anonymous_user']
    
    fieldsets = (
        ('Basic Info', {
            'fields': ('slug', 'anonymous_user', 'file', 'file_type', 'created_at')
        }),
        ('Analysis Results', {
            'fields': ('score', 'confidence', 'verdict', 'components', 'evidence', 'info')
        }),
        ('Visualizations', {
            'fields': ('fft_image', 'ela_image', 'noise_image')
        }),
        ('Statistics', {
            'fields': ('view_count', 'share_count')
        }),
    )
    
    def slug_short(self, obj):
        if obj.slug and len(obj.slug) > 30:
            return f"{obj.slug[:30]}..."
        return obj.slug or f"#{obj.id}"
    slug_short.short_description = "Slug"
    
    def anonymous_user_link(self, obj):
        if obj.anonymous_user:
            url = reverse('admin:detector_anonymoususer_change', args=[obj.anonymous_user.id])
            return format_html('<a href="{}">{}</a>', url, obj.anonymous_user.user_id[:16] + "...")
        return "No user"
    anonymous_user_link.short_description = "User"
