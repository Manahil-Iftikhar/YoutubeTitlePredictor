from django import forms


class VideoTitleForm(forms.Form):
    video_title = forms.CharField(
        label="Topic or video title", max_length=255, strip=True,
        widget=forms.TextInput(attrs={"placeholder": "e.g. Python data analysis", "autofocus": True}),
    )
