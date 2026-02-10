"""
Anonymous User Tracking
Tracks users without registration using browser fingerprinting
"""

import hashlib
import uuid
from .models import AnonymousUser


def get_client_ip(request):
    """Get client IP address from request"""
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip


def generate_fingerprint(request):
    """Generate browser fingerprint from request headers"""
    user_agent = request.META.get('HTTP_USER_AGENT', '')
    accept_language = request.META.get('HTTP_ACCEPT_LANGUAGE', '')
    accept_encoding = request.META.get('HTTP_ACCEPT_ENCODING', '')
    
    # Combine to create fingerprint
    fingerprint_string = f"{user_agent}|{accept_language}|{accept_encoding}"
    fingerprint = hashlib.sha256(fingerprint_string.encode()).hexdigest()
    
    return fingerprint


def get_or_create_anonymous_user(request):
    """
    Get or create anonymous user from request
    Uses cookie/session + fingerprint for identification
    """
    # Try to get user_id from cookie/session
    user_id = request.COOKIES.get('anonymous_user_id')
    if not user_id:
        user_id = request.session.get('anonymous_user_id')
    
    # Generate fingerprint
    fingerprint = generate_fingerprint(request)
    ip_address = get_client_ip(request)
    user_agent = request.META.get('HTTP_USER_AGENT', '')
    
    if user_id:
        # Try to find existing user
        try:
            user = AnonymousUser.objects.get(user_id=user_id)
            # Update last visit and IP
            user.ip_address = ip_address
            user.user_agent = user_agent
            user.save(update_fields=['last_visit', 'ip_address', 'user_agent'])
            return user
        except AnonymousUser.DoesNotExist:
            pass
    
    # Try to find by fingerprint
    try:
        user = AnonymousUser.objects.get(fingerprint=fingerprint, ip_address=ip_address)
        return user
    except AnonymousUser.DoesNotExist:
        pass
    
    # Create new anonymous user
    user_id = str(uuid.uuid4())
    user = AnonymousUser.objects.create(
        user_id=user_id,
        fingerprint=fingerprint,
        ip_address=ip_address,
        user_agent=user_agent
    )
    
    # Store in session
    request.session['anonymous_user_id'] = user_id
    
    return user


def set_anonymous_user_cookie(response, user):
    """Set cookie for anonymous user tracking"""
    response.set_cookie(
        'anonymous_user_id',
        user.user_id,
        max_age=365*24*60*60,  # 1 year
        httponly=True,
        samesite='Lax'
    )
    return response
