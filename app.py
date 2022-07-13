from flask import Flask, request, render_template
import pandas as pd
import joblib
import numpy as np

# Declare a Flask app
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def main():

    # If a form is submitted
    if request.method == "POST":

        # Unpickle classifier
        gb = joblib.load("gb.pkl")

        # Get values through input bars
        satisfaction_level = request.form.get("satisfaction_level")
        last_evaluation = request.form.get("last_evaluation")
        number_project = request.form.get("number_project")
        average_montly_hours = request.form.get("average_montly_hours")
        time_spend_company = request.form.get("time_spend_company")
        Work_accident = request.form.get("Work_accident")
        promotion_last_5years = request.form.get("promotion_last_5years")
        Departments_int = request.form.get("Departments_int")
        salary_int = request.form.get("salary_int")

        # Put inputs to dataframe
        X = pd.DataFrame([[satisfaction_level, last_evaluation, number_project, average_montly_hours, time_spend_company, Work_accident, promotion_last_5years, Departments_int, salary_int]],
                         columns=['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company', 'Work_accident', 'promotion_last_5years', 'Departments_int', 'salary_int'])

        # Get prediction
        prediction = gb.predict(X)[0]
        if prediction == 0:
            prediction = "Employee will not leave the job"
        else:
            prediction = "employee will leave the job"

    else:
        prediction = ""

    return render_template("index.html", output=prediction)


if __name__ == '__main__':
    app.run(debug=True)
