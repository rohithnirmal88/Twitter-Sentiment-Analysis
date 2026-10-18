from django import forms

class PredictionForm(forms.Form):

    text = forms.CharField(
        label="Tweet Text",
        widget=forms.Textarea(attrs={
            "class": "form-control",
            "rows": 5,
            "placeholder": "Enter tweet text"
        })
    )

    city = forms.CharField(
        label="City",
        widget=forms.TextInput(attrs={
            "class": "form-control",
            "placeholder": "Enter city"
        })
    )