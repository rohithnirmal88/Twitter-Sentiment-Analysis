from django.shortcuts import render
from .forms import PredictionForm

import pandas as pd
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = joblib.load(
    os.path.join(BASE_DIR, "ml_project", "model.pkl")
)
def home(request):
    return render(request, "predictor/home.html")


def about(request):
    return render(request, "predictor/about.html")


def contact(request):
    return render(request, "predictor/contact.html")

def predict(request):
    

    prediction = None

    if request.method == "POST":

        form = PredictionForm(request.POST)

        if form.is_valid():

            text = form.cleaned_data["text"]
            city = form.cleaned_data["city"]

            df = pd.DataFrame({
                "text": [text],
                "city": [city]
            })

            prediction = model.predict(df)[0]

    else:
        form = PredictionForm()

    return render(
        request,
        "predictor/predict.html",
        {
            "form": form,
            "prediction": prediction
        }
    )