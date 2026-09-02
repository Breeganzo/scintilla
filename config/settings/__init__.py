"""Settings package.

Split rather than a single module with `if DEBUG:` branches, because that
pattern is how production ends up running with a development default nobody
noticed. Here, production imports base and overrides explicitly.

    base.py   shared configuration
    dev.py    local development
    test.py   pytest and CI
    prod.py   deployed
"""
