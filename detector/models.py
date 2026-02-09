from django.db import models


class Detection(models.Model):
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
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Detection {self.id} - {self.verdict}"
