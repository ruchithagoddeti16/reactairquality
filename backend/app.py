from flask import Flask, request, jsonify, send_from_directory #render_template
import os
from flask_cors import CORS
from backend.preprocess import process_data, process_features, make_predictions

app = Flask(__name__, static_folder='./backend/frontend/dist')
CORS(app)

@app.route('/')
@app.route('/<path:path>')
def serve_react(path=''):
    if path != '' and os.path.exists(os.path.join(app.static_folder,path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')
    
@app.route('/health')
def health_check():
    return jsonify({'status':'ok'}), 200

@app.route('/predict', methods=['POST'])
def predict():
    try:            
        # Extract data from the JSON request
        data = request.json
        region = data['region']
        temperature = data['temperature']
        humidity = data['humidity']
        pm25 = data['pm25']
        pm10 = data['pm10']
        no2 = data['no2']
        so2 = data['so2']
        co = data['co']
        proximity_to_industrial_areas = data['proximityToIndustrialAreas']
        population_density = data['populationDensity']

        # Prepare the feature array for prediction
        features = [[region, temperature, humidity, pm25, pm10, no2, so2, co,
                    proximity_to_industrial_areas, population_density]]

        scaled_data = process_data(data)

        preprocessed_data = process_features(scaled_data)

        prediction = make_predictions(preprocessed_data) 

        # Return the prediction result as JSON
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error':str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
