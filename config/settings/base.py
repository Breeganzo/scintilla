"""Shared settings. Never used directly - always via dev, test or prod."""

import os
from pathlib import Path

import dj_database_url
from dotenv import load_dotenv

# config/settings/base.py -> config/settings -> config -> repository root
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Load .env if present. Absent in CI and production, where real environment
# variables are used instead.
load_dotenv(BASE_DIR / ".env")


def env_bool(name: str, default: bool = False) -> bool:
    """Read a boolean from the environment.

    Everything in the environment is a string, so `bool(os.getenv("DJANGO_DEBUG"))`
    is True for the string "False". This is a common and expensive mistake.
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def env_list(name: str, default: str = "") -> list[str]:
    """Read a comma-separated list, dropping empty entries."""
    return [item.strip() for item in os.getenv(name, default).split(",") if item.strip()]


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

# No fallback value. dev.py supplies one for convenience; prod.py asserts that
# a real key is present. A hardcoded default here would silently ship.
SECRET_KEY = os.getenv("DJANGO_SECRET_KEY", "")

DEBUG = False

ALLOWED_HOSTS = env_list("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1")

ROOT_URLCONF = "config.urls"
WSGI_APPLICATION = "config.wsgi.application"
ASGI_APPLICATION = "config.asgi.application"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# ---------------------------------------------------------------------------
# Applications
# ---------------------------------------------------------------------------

DJANGO_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
]

THIRD_PARTY_APPS = [
    "rest_framework",
    "drf_spectacular",
    "corsheaders",
]

LOCAL_APPS = [
    "papers",
    "search",
    "evaluation",
]

INSTALLED_APPS = DJANGO_APPS + THIRD_PARTY_APPS + LOCAL_APPS

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    # CORS must sit above CommonMiddleware so that its headers are attached
    # even to responses that CommonMiddleware short-circuits.
    "corsheaders.middleware.CorsMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

DATABASES = {
    "default": dj_database_url.parse(
        os.getenv("DATABASE_URL", "postgres://scintilla:scintilla@localhost:5432/scintilla"),
        # Reuse connections for 10 minutes instead of opening a new one per
        # request. Postgres connections are relatively expensive to establish.
        conn_max_age=600,
        conn_health_checks=True,
    )
}

# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

# ---------------------------------------------------------------------------
# Internationalisation
# ---------------------------------------------------------------------------

LANGUAGE_CODE = "en-gb"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

# ---------------------------------------------------------------------------
# Static files
# ---------------------------------------------------------------------------

STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"

# ---------------------------------------------------------------------------
# Django REST Framework
# ---------------------------------------------------------------------------

# Search gets its own throttle bucket. It is the endpoint the evaluation harness
# drives, and 50 queries across 3 modes is 150 requests in under a minute - the
# shared 100/hour allowance would have returned 429 for a third of an ablation
# run, which the harness would have recorded as "no results" rather than
# "refused". A named constant so the test suite can assert the shipped default
# is large enough, which it cannot do through REST_FRAMEWORK: test.py mutates
# that dict in place and would be asserting against its own override.
SEARCH_THROTTLE_RATE = os.getenv("SEARCH_THROTTLE_RATE", "60/minute")

REST_FRAMEWORK = {
    "DEFAULT_SCHEMA_CLASS": "drf_spectacular.openapi.AutoSchema",
    "DEFAULT_PAGINATION_CLASS": "rest_framework.pagination.PageNumberPagination",
    "PAGE_SIZE": 20,
    "DEFAULT_RENDERER_CLASSES": [
        "rest_framework.renderers.JSONRenderer",
    ],
    # Corpus data is public arXiv metadata, so reads are open. Rate limiting
    # rather than authentication is the control that matters here - the risk
    # is resource abuse, not disclosure.
    "DEFAULT_PERMISSION_CLASSES": [
        "rest_framework.permissions.AllowAny",
    ],
    "DEFAULT_THROTTLE_CLASSES": [
        "rest_framework.throttling.AnonRateThrottle",
    ],
    "DEFAULT_THROTTLE_RATES": {
        "anon": "100/hour",
        "search": SEARCH_THROTTLE_RATE,
    },
}

SPECTACULAR_SETTINGS = {
    "TITLE": "Scintilla API",
    "DESCRIPTION": ("Hybrid search and grounded question answering over scientific preprints."),
    "VERSION": "0.1.0",
    "SERVE_INCLUDE_SCHEMA": False,
    "COMPONENT_SPLIT_REQUEST": True,
}

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

# An explicit allowlist, never a wildcard. See docs/SECURITY.md.
CORS_ALLOWED_ORIGINS = env_list("CORS_ALLOWED_ORIGINS", "http://localhost:5173")

# ---------------------------------------------------------------------------
# Project settings
# ---------------------------------------------------------------------------

OPENSEARCH_URL = os.getenv("OPENSEARCH_URL", "http://localhost:9200")
OPENSEARCH_INDEX = os.getenv("OPENSEARCH_INDEX", "papers")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
# The default was llama-3.3-70b-versatile until Groq decommissioned it, which
# returned a 404 that reads like an authentication problem ("you do not have
# access to it") and is not. Hosted model names are not a stable interface, so
# this is configurable and the provider surfaces the API's own message rather
# than collapsing every failure into "the LLM is down".
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")

ARXIV_CATEGORIES = env_list("ARXIV_CATEGORIES", "hep-ex,hep-th")
ARXIV_RATE_LIMIT_SECONDS = float(os.getenv("ARXIV_RATE_LIMIT_SECONDS", "3"))
ARXIV_MAX_RESULTS = int(os.getenv("ARXIV_MAX_RESULTS", "3000"))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "verbose": {
            "format": "{levelname} {asctime} {name} {message}",
            "style": "{",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "verbose",
        },
    },
    "root": {
        "handlers": ["console"],
        "level": "INFO",
    },
    "loggers": {
        "django.db.backends": {
            # Set to DEBUG to see every SQL query. Very loud - useful when
            # hunting an N+1 problem, unbearable otherwise.
            "level": "INFO",
            "handlers": ["console"],
            "propagate": False,
        },
    },
}
