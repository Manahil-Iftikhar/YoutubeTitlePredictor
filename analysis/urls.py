from django.urls import path
from .views import predict_engagement_view

urlpatterns = [path("", predict_engagement_view, name="predict")]
