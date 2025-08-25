from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Initialize the Flask application
application = Flask(__name__)

app = application

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html') 

# Route for the prediction form
@app.route('/home')
def home():
    return render_template('home.html')

# Route for handling the prediction form submission
@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        # POST request: Extract data from form (input features)
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))
        )

        # Convert data to dataframe
        pred_df = data.get_data_as_data_frame()
        print("User Data:\n", pred_df)
        
        # Prediction
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        print("Prediction results:", results)

        return render_template('home.html', results=results[0])

# Run the Flask application
if __name__=="__main__":
    app.run(host="0.0.0.0")