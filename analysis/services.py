"""Request-time YouTube metadata lookup; no training or network activity on import."""
import os
import re
import requests


class VideoSearchError(Exception):
    """Safe, user-facing search failure."""


def search_videos(query):
    key = os.environ.get("YOUTUBE_API_KEY", "").strip()
    if not key:
        raise VideoSearchError("YouTube search is not configured. Set YOUTUBE_API_KEY locally.")
    try:
        response = requests.get(
            "https://www.googleapis.com/youtube/v3/search",
            params={"part": "snippet", "type": "video", "q": query,
                    "maxResults": 10, "key": key}, timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
        items = payload["items"]
        if not isinstance(items, list):
            raise ValueError("Invalid items")
        videos = []
        for item in items:
            video_id = item["id"]["videoId"]
            snippet = item["snippet"]
            if not isinstance(video_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
                continue
            videos.append({
                "title": str(snippet.get("title", "Untitled video")),
                "description": str(snippet.get("description", "")),
                "channel": str(snippet.get("channelTitle", "")),
                "url": "https://www.youtube.com/watch?v=" + video_id,
            })
        return videos
    except (requests.RequestException, ValueError, KeyError, TypeError, AttributeError):
        # Do not expose API keys, request URLs, or provider response bodies.
        raise VideoSearchError("YouTube search is unavailable. Check your API configuration and quota, then try again.") from None
