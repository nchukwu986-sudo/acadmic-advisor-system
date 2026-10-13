from pathlib import Path

import joblib

import pandas as pd

from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)

# Load the trained model

MODEL_PATH = BASE_DIR / "academic_advisor_model.pkl"

model = joblib.load(MODEL_PATH)

def generate_advice(

    cgpa,

    failed_courses,

    outstanding_courses,

    study_hours,

    attendance

):

    recommendations = []

    if cgpa < 2.5:

        recommendations.append(

            "Meet your academic adviser to discuss your CGPA "

            "and develop an improvement plan."

        )

    if failed_courses > 0:

        recommendations.append(

            "Prioritize your failed courses and consult your "

            "lecturers about revision resources."

        )

    if outstanding_courses >= 3:

        recommendations.append(

            "Create a realistic plan for completing your "

            "outstanding courses."

        )

    if study_hours < 2:

        recommendations.append(

            "Consider establishing a consistent daily study "

            "routine that fits your timetable."

        )

    if attendance < 75:

        recommendations.append(

            "Improve your class attendance and identify any "

            "barriers preventing you from attending."

        )

    if not recommendations:

        recommendations.append(

            "Maintain your current study habits, monitor your "

            "results, and continue setting academic goals."

        )

    return recommendations

@app.route("/", methods=["GET", "POST"])

def index():

    result = None

    recommendations = []

    error = None

    if request.method == "POST":

        try:

            cgpa = float(request.form["cgpa"])

            failed_courses = int(request.form["failed_courses"])

            outstanding_courses = int(

                request.form["outstanding_courses"]

            )

            study_hours = float(request.form["study_hours"])

            attendance = float(request.form["attendance"])

            # Validate student inputs

            if not 0 <= cgpa <= 5:

                raise ValueError("CGPA must be between 0 and 5.")

            if not 0 <= failed_courses <= 100:

                raise ValueError(

                    "Failed courses must be between 0 and 100."

                )

            if not 0 <= outstanding_courses <= 100:

                raise ValueError(

                    "Outstanding courses must be between 0 and 100."

                )

            if not 0 <= study_hours <= 24:

                raise ValueError(

                    "Study hours must be between 0 and 24."

                )

            if not 0 <= attendance <= 100:

                raise ValueError(

                    "Attendance must be between 0 and 100."

                )

            student = pd.DataFrame(

                [[

                    cgpa,

                    failed_courses,

                    outstanding_courses,

                    study_hours,

                    attendance

                ]],

                columns=[

                    "cgpa",

                    "failed_courses",

                    "outstanding_courses",

                    "study_hours",

                    "attendance"

                ]

            )

            # Obtain the machine learning prediction

            result = model.predict(student)[0]

            # Generate personalized recommendations

            recommendations = generate_advice(

                cgpa,

                failed_courses,

                outstanding_courses,

                study_hours,

                attendance

            )

        except (ValueError, KeyError):

            error = (

                "Please enter valid numerical values within "

                "the specified ranges."

            )

    return render_template(

        "index.html",

        result=result,

        recommendations=recommendations,

        error=error

    )

if __name__ == "__main__":

    app.run(debug=True)