#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""

import os
import sys


def main() -> None:
    # Development is the default so that a bare `python manage.py runserver`
    # can never accidentally load production settings. CI and production set
    # DJANGO_SETTINGS_MODULE explicitly.
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings.dev")

    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Could not import Django. Is it installed and is your virtual "
            "environment active? Try: source .venv/bin/activate"
        ) from exc

    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
