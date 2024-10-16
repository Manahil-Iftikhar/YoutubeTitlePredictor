from django import forms

class VideoTitleForm(forms.Form):
    video_title = forms.CharField(label='Video Title', max_length=255)
    competition = forms.IntegerField(label='Competition')
    search_volume = forms.IntegerField(label='Search Volume')
