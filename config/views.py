"""Project-level views.

Only cross-cutting endpoints belong here. Domain endpoints live in their app.
"""

from django.db import connection
from drf_spectacular.utils import extend_schema
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.request import Request
from rest_framework.response import Response


@extend_schema(
    summary="Liveness and readiness probe",
    description=(
        "Returns 200 when the service can reach its database, 503 otherwise. "
        "Used by the container health check and by uptime monitoring."
    ),
    responses={200: None, 503: None},
    tags=["system"],
)
@api_view(["GET"])
@permission_classes([AllowAny])
def health(request: Request) -> Response:
    """Report service health.

    This deliberately touches the database rather than returning a bare 200.
    A health endpoint that reports healthy while the database is unreachable
    is worse than no health endpoint, because it suppresses the alert that
    would otherwise fire.
    """
    checks: dict[str, str] = {}
    healthy = True

    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
            cursor.fetchone()
        checks["database"] = "ok"
    except Exception as exc:  # noqa: BLE001 - any failure means unhealthy
        checks["database"] = f"error: {type(exc).__name__}"
        healthy = False

    return Response(
        {"status": "ok" if healthy else "degraded", "checks": checks},
        status=status.HTTP_200_OK if healthy else status.HTTP_503_SERVICE_UNAVAILABLE,
    )
