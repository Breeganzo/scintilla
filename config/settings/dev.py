"""Local development settings."""

from .base import *  # noqa: F403
from .base import SECRET_KEY, env_bool

DEBUG = env_bool("DJANGO_DEBUG", True)

ALLOWED_HOSTS = ["localhost", "127.0.0.1", "0.0.0.0", "api"]  # noqa: S104

# Convenience only. Never reached in production, which asserts a real key.
if not SECRET_KEY:
    SECRET_KEY = "dev-insecure-key-do-not-use-outside-local-development"  # noqa: S105

# The Vite dev server, plus the Django dev server itself.
CORS_ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8000",
]

# Browsable API is genuinely useful while developing endpoints by hand.
REST_FRAMEWORK["DEFAULT_RENDERER_CLASSES"] = [  # noqa: F405
    "rest_framework.renderers.JSONRenderer",
    "rest_framework.renderers.BrowsableAPIRenderer",
]

# Throttling gets in the way when clicking around locally, so raise the ceiling -
# but raise it for every scope base declares rather than writing out a new dict.
# Replacing the dict wholesale is how /api/search/ came to return 500 for every
# request in development: the "search" scope vanished, ScopedRateThrottle raised
# ImproperlyConfigured, and no test noticed because tests use test.py. Deriving
# the keys from base means a scope added later cannot be dropped here by
# omission.
REST_FRAMEWORK["DEFAULT_THROTTLE_RATES"] = dict.fromkeys(  # noqa: F405
    REST_FRAMEWORK["DEFAULT_THROTTLE_RATES"],  # noqa: F405
    "10000/hour",
)
