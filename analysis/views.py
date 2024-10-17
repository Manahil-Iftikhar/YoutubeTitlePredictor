import joblib  # Import joblib to load the model
from django.shortcuts import render
from .forms import VideoTitleForm
# from .seo_tool import *
from .youtube_seo_tool import *

# Load the model at the top of your views.py file
# model = joblib.load('path/to/save/model.pkl')  # Adjust the path as necessary

def predict_engagement_view(request):
    form = VideoTitleForm(request.POST or None)  # Initialize form with POST data if available

    if request.method == 'POST' and form.is_valid():
        title = form.cleaned_data['video_title']
        competition = form.cleaned_data['competition']  # Ensure this field exists in the form
        search_volume = form.cleaned_data['search_volume']  # Ensure this field exists in the form

        result = search_videos(query=title)

        return render(request, 'results.html', {
            'form': form,
            # 'prediction': predicted_engagement_score,
            'video_data' : result
            # 'video_data': video_data_df.to_dict(orient='records')  # Convert DataFrame to a list of dictionaries for rendering
        })

    return render(request, 'predict.html', {'form': form})
