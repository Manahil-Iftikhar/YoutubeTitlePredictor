from django.shortcuts import render

import pickle
from django.shortcuts import render
from .forms import VideoTitleForm

# Load your trained machine learning model (replace 'model.pkl' with your model's file)
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

def predict_engagement(request):
    if request.method == 'POST':
        form = VideoTitleForm(request.POST)
        if form.is_valid():
            video_title = form.cleaned_data['video_title']
            competition = form.cleaned_data['competition']
            search_volume = form.cleaned_data['search_volume']

            # Feature extraction (title length in this case)
            title_length = len(video_title)
            features = [[title_length, competition, search_volume]]

            # Predict engagement
            predicted_engagement = model.predict(features)[0]

            return render(request, 'template/result.html', {
                'form': form,
                'prediction': predicted_engagement,
                'video_title': video_title
            })
    else:
        form = VideoTitleForm()
    
    return render(request, 'template/predict.html', {'form': form})

