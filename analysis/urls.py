from django.urls import path
from .views import predict_engagement_view, demo_view

urlpatterns = [
    path("", predict_engagement_view, name="predict"),
    path("demo/", demo_view, name="demo"),
]
