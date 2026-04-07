from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os

app = Flask(__name__)
CORS(app)

# Load the pipeline
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
try:
    pipeline = joblib.load(model_path)
    # Extract the random forest model from inside the pipeline to get importances
    rf_model = pipeline.named_steps['classifier']
except FileNotFoundError:
    pipeline = None
    rf_model = None

@app.route('/predict', methods=['POST'])
def predict():
    if not pipeline:
        return jsonify({'error': 'Model not loaded.'}), 500

    try:
        data = request.json
        
        # 1. Convert JSON directly to Pandas DataFrame
        # The pipeline will automatically handle the text-to-number encoding!
        input_df = pd.DataFrame([data])
        
        # 2. Predict Rating
        prediction = pipeline.predict(input_df)[0]
        rating = int(prediction)
        
        # 3. Extract Feature Importance
        # Tells us mathematically what the AI cared about most
        importances = rf_model.feature_importances_
        features = ['Buying Price', 'Maintenance', 'Doors', 'Capacity', 'Boot Size', 'Safety']
        
        # Create a sorted dictionary of feature importances
        importance_dict = {feat: round(float(imp) * 100, 1) for feat, imp in zip(features, importances)}
        # Sort by most important descending
        sorted_importance = dict(sorted(importance_dict.items(), key=lambda item: item[1], reverse=True))

        # 4. Generate dynamic explanation
        top_feature = list(sorted_importance.keys())[0]
        if rating <= 3:
            explanation = f"This car received a {rating}-star rating. The AI relied heavily on its '{top_feature}' to make this decision."
        else:
            explanation = f"Great car! The {rating}-star score was strongly driven by its excellent '{top_feature}'."

        return jsonify({
            'rating': rating,
            'explanation': explanation,
            'feature_importance': sorted_importance
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)