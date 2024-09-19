from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load the model
model = pickle.load(open('rf.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html', form_data=None, prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    form_data = request.form.to_dict()
    
    # Extract form values
    int_features = [float(form_data[key]) for key in form_data]
    
    # Convert inputs to the format expected by the model
    final_features = [np.array(int_features)]
    
    # Perform prediction
    prediction = model.predict(final_features)[0]
    
    # Translate prediction to the human-readable result
    if prediction == 0:
        prediction_text = "The person has No Heart Disease"
    else:
        prediction_text = "The person has Heart Disease"
    
    # Return the prediction and form data to the template
    return render_template('index.html', form_data=form_data, prediction=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)
