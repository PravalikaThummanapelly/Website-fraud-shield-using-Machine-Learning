import numpy as np
import pickle
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .feature import FeatureExtraction

# Load the model when the application starts
try:
    with open("C:/Users/Sahithya/Downloads/Phishing website/core/model.pkl", "rb") as file:
        gbc = pickle.load(file)
except Exception as e:
    print(f"Error loading model: {e}")
    gbc = None

@api_view(['POST'])
def predict(request):
    if gbc is None:
        return Response({'error': 'Model could not be loaded'}, status=500)

    url = request.data.get('url', '')
    if not url:
        return Response({'error': 'URL is required'}, status=400)

    try:
        obj = FeatureExtraction(url)
        x = np.array(obj.getFeaturesList()).reshape(1, 30)

        y_pred = gbc.predict(x)[0]
        y_pro_phishing = gbc.predict_proba(x)[0, 0]
        y_pro_non_phishing = gbc.predict_proba(x)[0, 1]

        response_data = {
            'url': url,
            'prediction': y_pred,
            'probability_phishing': y_pro_phishing,
            'probability_non_phishing': y_pro_non_phishing
        }
        return Response(response_data)
    except Exception as e:
        return Response({'error': str(e)}, status=500)