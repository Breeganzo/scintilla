"""Production settings.

Deliberately strict: anything that could be insecurely defaulted is asserted
rather than defaulted. Failing to boot is a better outcome than booting
insecurely.
"""

import os

from .base import *  # noqa: F403
from .base import SECRET_KEY, env_list

DEBUG = False

if not SECRET_KEY:
    raise RuntimeError(
        "DJANGO_SECRET_KEY is not set. Refusing to start rather than fall back "
        "to a predictable default."
    )

ALLOWED_HOSTS = env_list("DJANGO_ALLOWED_HOSTS")
if not ALLOWED_HOSTS:
    raise RuntimeError("DJANGO_ALLOWED_HOSTS must be set in production.")

CORS_ALLOWED_ORIGINS = env_list("CORS_ALLOWED_ORIGINS")
if not CORS_ALLOWED_ORIGINS:
    raise RuntimeError("CORS_ALLOWED_ORIGINS must be set in production.")

# ---------------------------------------------------------------------------
# HTTPS and security headers
# ---------------------------------------------------------------------------

# TLS terminates at Cloudflare, which forwards the original scheme in this
# header. Without it Django believes every request is plain HTTP and
# redirects forever.
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")

SECURE_SSL_REDIRECT = True

# Start low and raise once confident. A long max-age is hard to undo: browsers
# remember it, and a misconfiguration becomes a site nobody can reach.
SECURE_HSTS_SECONDS = int(os.getenv("SECURE_HSTS_SECONDS", "3600"))
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_HSTS_PRELOAD = False

SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
SESSION_COOKIE_HTTPONLY = True

SECURE_CONTENT_TYPE_NOSNIFF = True
SECURE_REFERRER_POLICY = "strict-origin-when-cross-origin"
X_FRAME_OPTIONS = "DENY"

CSRF_TRUSTED_ORIGINS = env_list("CSRF_TRUSTED_ORIGINS")

# ---------------------------------------------------------------------------
# Static files
# ---------------------------------------------------------------------------

STORAGES = {
    "default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
    "staticfiles": {"BACKEND": "django.contrib.staticfiles.storage.ManifestStaticFilesStorage"},
}
