from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('car_price_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form  # Fetch data from the form

        # Validate all input fields
        if not all(key in data and data[key].strip() != "" for key in ["year", "kms_driven", "company", "fuel_type"]):
            return jsonify({'error': 'All fields are required. Please fill in all values.'})

        # Convert inputs to correct format
        year = int(data['year'])
        kms_driven = float(data['kms_driven'])
        company = int(data['company'])
        fuel_type = int(data['fuel_type'])

        # Prepare input for model
        features = np.array([[year, kms_driven, company, fuel_type]])

        # Make prediction
        prediction = model.predict(features)[0]

        return jsonify({'price': round(prediction, 2)})

    except ValueError:
        return jsonify({'error': 'Invalid data format. Ensure all inputs are numbers.'})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
