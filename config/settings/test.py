"""Test settings, used by pytest and CI.

Database strategy: if DATABASE_URL is set, use it. CI sets it to a real
PostgreSQL service container, so the test suite exercises the same engine
production uses.

If it is not set, fall back to SQLite. That lets the unit suite run on a
machine with no Postgres installed - useful, but it means a green local run is
weaker evidence than a green CI run. CI is the authority.
"""

import os

from .base import *  # noqa: F403

DEBUG = False

SECRET_KEY = "test-only-key-never-used-outside-the-test-suite"  # noqa: S105

ALLOWED_HOSTS = ["*"]

if not os.getenv("DATABASE_URL"):
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": ":memory:",
        }
    }

# Hashing passwords properly is slow by design. Tests create many users and do
# not care about hash strength, so use the cheapest hasher available.
PASSWORD_HASHERS = ["django.contrib.auth.hashers.MD5PasswordHasher"]

# Throttling would make the suite non-deterministic. A rate of None disables a
# scope without unwiring it, so the search view keeps its ScopedRateThrottle -
# tests that care about throttling switch it back on with override_settings.
REST_FRAMEWORK["DEFAULT_THROTTLE_CLASSES"] = []  # noqa: F405
REST_FRAMEWORK["DEFAULT_THROTTLE_RATES"] = {"search": None}  # noqa: F405

# Keep test output readable.
LOGGING["root"]["level"] = "WARNING"  # noqa: F405

# No migrations run against a live search cluster during unit tests.
OPENSEARCH_URL = "http://localhost:9200"
OPENSEARCH_INDEX = "papers_test"
