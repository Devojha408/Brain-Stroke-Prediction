from flask import Flask, render_template, request
import numpy as np
import pickle
app = Flask("Stroke_Prediction")

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/result', methods=['POST','GET'])
def result():
    gender=int(request.form['gender'])
    age=int(request.form['age'])
    hypertension=int(request.form['hypertension'])
    heart_disease = int(request.form['heart_disease'])
    ever_married = int(request.form['ever_married'])
    work_type = int(request.form['work_type'])
    Residence_type = int(request.form['Residence_type'])
    avg_glucose_level = float(request.form['avg_glucose_level'])
    bmi = float(request.form['bmi'])
    smoking_status = int(request.form['smoking_status'])
    
    X=np.array([gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,
                avg_glucose_level,bmi,smoking_status]).reshape(1,-1)

    with open('model.pkl', 'rb') as f:
        kn=pickle.load(f)
    pred=kn.predict(X)


    # for No Stroke Risk
    if pred==0:
        return render_template('nostroke.html')
    else:
        return render_template('stroke.html')


if __name__ == "__main__":
    app.run(debug=True)