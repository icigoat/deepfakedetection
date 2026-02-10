"""
Generate PWA icons from the custom modern_app.png icon
Run this script to create all required icon sizes
"""

from PIL import Image
import os

# Try multiple possible locations for source icon
possible_sources = [
    'staticfiles/icons/modern_app.png',
    'static/icons/modern_app.png',
    'detector/static/icons/modern_app.png',
]

source_icon = None
for path in possible_sources:
    if os.path.exists(path):
        source_icon = path
        break

if not source_icon:
    print('❌ Source icon not found in any of these locations:')
    for path in possible_sources:
        print(f'   - {path}')
    print('\n⚠️  Skipping icon generation')
    exit(0)  # Exit gracefully, don't fail the build

# Create icons directory
icons_dir = 'detector/static/icons'
os.makedirs(icons_dir, exist_ok=True)

# Icon sizes needed for PWA
sizes = [72, 96, 128, 144, 152, 192, 384, 512]

def create_icon(size):
    """Resize the source icon to the target size"""
    try:
        # Open the source image
        img = Image.open(source_icon)
        
        # Convert to RGBA if not already
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        # Resize with high-quality resampling
        img_resized = img.resize((size, size), Image.Resampling.LANCZOS)
        
        # Save icon
        filename = f'{icons_dir}/icon-{size}x{size}.png'
        img_resized.save(filename, 'PNG', optimize=True)
        print(f'✅ Created: {filename}')
        
    except Exception as e:
        print(f'❌ Error creating {size}x{size}: {e}')

print(f'📦 Using source icon: {source_icon}')
print('🎨 Generating PWA icons...\n')

# Generate all icon sizes
for size in sizes:
    create_icon(size)

# Also create favicon
try:
    img = Image.open(source_icon)
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    
    # Create 32x32 favicon
    favicon = img.resize((32, 32), Image.Resampling.LANCZOS)
    favicon.save('detector/static/favicon.ico', format='ICO')
    print(f'\n✅ Created: detector/static/favicon.ico')
except Exception as e:
    print(f'\n❌ Error creating favicon: {e}')

print(f'\n✅ All icons generated in {icons_dir}/')
print('🚀 Icons are ready for PWA deployment!')
