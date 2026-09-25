import os
from unittest.mock import patch, Mock
import requests
from django.test import SimpleTestCase, Client
from .services import search_videos, VideoSearchError


class WebTests(SimpleTestCase):
    @patch("analysis.views.search_videos")
    def test_get_and_invalid_post_never_search(self, search):
        self.assertEqual(self.client.get("/").status_code, 200)
        for data in ({}, {"video_title": "   "}, {"video_title": "x" * 256}):
            self.assertEqual(self.client.post("/", data).status_code, 200)
        search.assert_not_called()

    @patch("analysis.views.search_videos")
    def test_results_escape_provider_text(self, search):
        search.return_value = [{"title": "<script>alert(1)</script>", "channel": "Example", "description": "Demo", "url": "https://www.youtube.com/watch?v=abcdefghijk"}]
        response = self.client.post("/", {"video_title": " Python "})
        search.assert_called_once_with("Python")
        self.assertContains(response, "&lt;script&gt;")
        self.assertNotContains(response, "<script>")
        self.assertContains(response, "Example")

    @patch("analysis.views.search_videos", return_value=[])
    def test_empty_results(self, search):
        self.assertContains(self.client.post("/", {"video_title": "Python"}), "No matching videos")

    @patch("analysis.views.search_videos", side_effect=VideoSearchError("Search unavailable"))
    def test_error_preserves_form(self, search):
        response = self.client.post("/", {"video_title": "Python"})
        self.assertContains(response, "Search unavailable")
        self.assertContains(response, 'value="Python"')

    def test_csrf_and_methods(self):
        self.assertEqual(Client(enforce_csrf_checks=True).post("/", {"video_title": "Python"}).status_code, 403)
        self.assertEqual(self.client.put("/").status_code, 405)


class ServiceTests(SimpleTestCase):
    @patch.dict(os.environ, {}, clear=True)
    @patch("analysis.services.requests.get")
    def test_missing_key_never_calls_network(self, get):
        with self.assertRaisesRegex(VideoSearchError, "not configured"):
            search_videos("Python")
        get.assert_not_called()

    @patch.dict(os.environ, {"YOUTUBE_API_KEY": "test-only"})
    @patch("analysis.services.requests.get")
    def test_success_and_safe_video_links(self, get):
        response = Mock()
        response.json.return_value = {"items": [
            {"id": {"videoId": "abcdefghijk"}, "snippet": {"title": "Python"}},
            {"id": {"videoId": "javascript:bad"}, "snippet": {}},
        ]}
        get.return_value = response
        videos = search_videos("Python")
        self.assertEqual(len(videos), 1)
        self.assertEqual(videos[0]["url"], "https://www.youtube.com/watch?v=abcdefghijk")
        self.assertEqual(get.call_args.kwargs["timeout"], 15)
        self.assertEqual(get.call_args.kwargs["params"]["q"], "Python")

    @patch.dict(os.environ, {"YOUTUBE_API_KEY": "test-only"})
    @patch("analysis.services.requests.get")
    def test_provider_failures_are_sanitized(self, get):
        for error in (requests.Timeout("secret-url"), requests.HTTPError("secret-url")):
            get.side_effect = error
            with self.assertRaises(VideoSearchError) as caught:
                search_videos("Python")
            self.assertNotIn("secret-url", str(caught.exception))
        get.side_effect = None
        for payload in ({}, {"items": None}, {"items": [{}]}, None):
            get.return_value = Mock()
            get.return_value.json.return_value = payload
            with self.assertRaises(VideoSearchError):
                search_videos("Python")
