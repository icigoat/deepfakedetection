from django.core.management.base import BaseCommand
from detector.models import Detection
from detector.slug_generator import generate_unique_slug


class Command(BaseCommand):
    help = 'Generate slugs for existing detections that do not have one'

    def handle(self, *args, **options):
        detections_without_slugs = Detection.objects.filter(slug__isnull=True) | Detection.objects.filter(slug='')
        count = detections_without_slugs.count()
        
        if count == 0:
            self.stdout.write(self.style.SUCCESS('All detections already have slugs!'))
            return
        
        self.stdout.write(f'Found {count} detections without slugs. Generating...')
        
        existing_slugs = set(Detection.objects.exclude(slug__isnull=True).exclude(slug='').values_list('slug', flat=True))
        
        for detection in detections_without_slugs:
            detection.slug = generate_unique_slug(existing_slugs)
            existing_slugs.add(detection.slug)
            detection.save(update_fields=['slug'])
            self.stdout.write(f'Generated slug for Detection #{detection.id}: {detection.slug}')
        
        self.stdout.write(self.style.SUCCESS(f'Successfully generated {count} slugs!'))
