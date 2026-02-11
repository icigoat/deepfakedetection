from django.db import models
from .slug_generator import generate_unique_slug
from django.utils import timezone


class AnonymousUser(models.Model):
    """Track anonymous users without registration"""
    user_id = models.CharField(max_length=64, unique=True, db_index=True)
    fingerprint = models.CharField(max_length=255, blank=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.TextField(blank=True)
    first_visit = models.DateTimeField(auto_now_add=True)
    last_visit = models.DateTimeField(auto_now=True)
    detection_count = models.IntegerField(default=0)
    
    class Meta:
        ordering = ['-last_visit']
        verbose_name = "Anonymous User"
        verbose_name_plural = "Anonymous Users"
    
    def __str__(self):
        return f"User {self.user_id[:8]}... ({self.detection_count} detections)"


class SiteSettings(models.Model):
    """Global site settings - only one instance should exist"""
    
    class Meta:
        verbose_name = "Site Settings"
        verbose_name_plural = "Site Settings"
    
    def __str__(self):
        return "Site Settings"
    
    def save(self, *args, **kwargs):
        # Ensure only one instance exists
        self.pk = 1
        super().save(*args, **kwargs)
    
    @classmethod
    def get_settings(cls):
        """Get or create the single settings instance"""
        obj, created = cls.objects.get_or_create(pk=1)
        return obj


class Detection(models.Model):
    slug = models.SlugField(max_length=200, unique=True, db_index=True, null=True, blank=True)
    anonymous_user = models.ForeignKey(AnonymousUser, on_delete=models.SET_NULL, null=True, blank=True, related_name='detections')
    file = models.FileField(upload_to='uploads/')
    file_type = models.CharField(max_length=10)
    score = models.FloatField()
    confidence = models.FloatField()
    verdict = models.CharField(max_length=200)
    components = models.JSONField()
    evidence = models.JSONField()
    info = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    # Visualization images
    fft_image = models.ImageField(upload_to='visualizations/', null=True, blank=True)
    ela_image = models.ImageField(upload_to='visualizations/', null=True, blank=True)
    noise_image = models.ImageField(upload_to='visualizations/', null=True, blank=True)
    
    # Share tracking
    share_count = models.IntegerField(default=0)
    view_count = models.IntegerField(default=0)
    
    def save(self, *args, **kwargs):
        # Always generate slug if not present
        if not self.slug:
            # Get existing slugs to avoid duplicates
            existing_slugs = set(Detection.objects.values_list('slug', flat=True))
            self.slug = generate_unique_slug(existing_slugs)
        super().save(*args, **kwargs)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        if self.slug:
            return f"Detection: {self.slug}"
        return f"Detection #{self.id}"
    
    def increment_views(self):
        """Increment view count"""
        self.view_count += 1
        self.save(update_fields=['view_count'])
    
    def increment_shares(self):
        """Increment share count"""
        self.share_count += 1
        self.save(update_fields=['share_count'])
