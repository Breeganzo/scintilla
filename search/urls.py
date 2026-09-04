"""URL routing for search.

A plain path rather than a router: this is one action, not a resource with list
and detail views, and forcing it into a ViewSet would obscure that.
"""

from django.urls import path

from search.views import SearchView

urlpatterns = [
    path("search/", SearchView.as_view(), name="search"),
]
