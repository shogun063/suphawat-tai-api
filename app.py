from flask import Flask, request
import joblib
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# Load the model
model = joblib.load("tai.pkl")

@app.route('/api/tai', methods=['POST'])
def house():
    BloodPressure = int(request.form.get('1')) 
    SpecificGravity = float(request.form.get('2')) 
    BloodUrea = int(request.form.get('3'))
    Sodium = float(request.form.get('4'))
    Pottasium = float(request.form.get('5'))
    
    # Prepare the input for the model
    x = np.array([[BloodPressure, SpecificGravity, BloodUrea,Sodium,Pottasium]])

    # Predict using the model
    prediction = model.predict(x)

    # Return the result
    return {'tai': prediction[0].tolist()}, 200    

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000)