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

# Throttling gets in the way when clicking around locally.
REST_FRAMEWORK["DEFAULT_THROTTLE_RATES"] = {"anon": "10000/hour"}  # noqa: F405
