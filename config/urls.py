"""Root URL configuration."""

from django.conf import settings
from django.contrib import admin
from django.urls import include, path
from drf_spectacular.views import (
    SpectacularAPIView,
    SpectacularSwaggerView,
)

from config.views import health

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/health/", health, name="health"),
    path("api/", include("papers.urls")),
    # The schema is the contract. The frontend generates its TypeScript types
    # from this endpoint, so a serializer change that breaks the client is
    # caught at build time rather than at runtime.
    path("api/schema/", SpectacularAPIView.as_view(), name="schema"),
    path(
        "api/docs/",
        SpectacularSwaggerView.as_view(url_name="schema"),
        name="swagger-ui",
    ),
]

if settings.DEBUG:
    urlpatterns += [
        path("api-auth/", include("rest_framework.urls")),
    ]
