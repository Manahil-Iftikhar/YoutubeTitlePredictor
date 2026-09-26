from django.shortcuts import render
from django.views.decorators.http import require_http_methods
from .forms import VideoTitleForm
from .services import VideoSearchError, search_videos


@require_http_methods(["GET", "POST"])
def predict_engagement_view(request):
    form = VideoTitleForm(request.POST if request.method == "POST" else None)
    if request.method == "POST" and form.is_valid():
        query = form.cleaned_data["video_title"]
        try:
            videos = search_videos(query)
        except VideoSearchError as exc:
            form.add_error(None, str(exc))
        else:
            return render(request, "results.html", {"query": query, "videos": videos})
    return render(request, "predict.html", {"form": form})


@require_http_methods(["GET"])
def demo_view(request):
    from .demo import DEMO_VIDEOS
    return render(request, "results.html", {
        "query": "Offline sample collection",
        "videos": DEMO_VIDEOS,
        "is_demo": True,
    })
