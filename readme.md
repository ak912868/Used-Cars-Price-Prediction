🚗 Used Cars Price Prediction

A Flask-based web application that predicts the resale price of used cars based on features such as manufacturing year, kilometers driven, company, and fuel type. The app uses a trained machine learning model to provide accurate predictions through a simple and user-friendly web interface.

📌 Features

✅ Predicts used car prices based on input parameters

✅ Flask backend with machine learning integration

✅ Input validation and error handling

✅ REST API endpoint (/predict) for JSON-based predictions

✅ Simple, responsive frontend using HTML (index.html)

🛠️ Tech Stack

Python 3.x

Flask – Web framework

NumPy – Numerical computations

Pickle – Model serialization

HTML/CSS/JS – Frontend UI

📂 Project Structure
Used-Car-Price-Prediction/
│
├── static/                 # Static files (CSS, JS, images)
├── templates/              # HTML templates (frontend UI)
│   └── index.html
│
├── car_price_model.pkl     # Trained ML model (serialized)
├── app.py                  # Main Flask application
├── requirements.txt        # Required dependencies
└── README.md               # Project documentation

⚙️ Installation & Setup

Clone the repository

git clone https://github.com/your-username/used-car-price-prediction.git
cd used-car-price-prediction


Create & activate virtual environment (optional but recommended)

python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows


Install dependencies

pip install -r requirements.txt


Run the application

python app.py


Access the app in your browser

http://127.0.0.1:5000/

📊 Usage

Open the app in your browser.

Enter the required details:

Year of the car

Kilometers Driven

Company (as encoded value)

Fuel Type (as encoded value)

Click Predict.

The app will display the predicted car price.

🔌 API Endpoint
POST /predict

Request (Form Data):

{
  "year": 2018,
  "kms_driven": 45000,
  "company": 2,
  "fuel_type": 1
}


Response (JSON):

{
  "price": 450000.75
}


Error Example:

{
  "error": "All fields are required. Please fill in all values."
}

📦 Dependencies

Add these to requirements.txt:

Flask
numpy
pickle-mixin

🚀 Deployment

You can deploy this app on:

Heroku

PythonAnywhere

Render

AWS / Azure / GCP

📜 License

This project is licensed under the MIT License.

👨‍💻 Author

Anuj Kumar
📧 ak9128685898@gmail.com
