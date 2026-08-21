from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

# ==========================================
# PROJECT FOLDER PATH
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# CREATE FLASK APPLICATION
# ==========================================

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates")
)

# ==========================================
# LOAD TRAINED MODEL
# ==========================================

model_path = os.path.join(
    BASE_DIR,
    "house_price_model.pkl"
)

model = joblib.load(model_path)

print("====================================")
print("     MODEL LOADED SUCCESSFULLY")
print("====================================")


# ==========================================
# HOME PAGE
# ==========================================

@app.route("/")
def home():
    return render_template("index.html")


# ==========================================
# PREDICTION
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # ==========================================
        # GET VALUES FROM HTML FORM
        # ==========================================

        bed = float(request.form["bed"])
        bath = float(request.form["bath"])
        house_size = float(request.form["house_size"])
        acre_lot = float(request.form["acre_lot"])

        city = request.form["city"].strip()
        state = request.form["state"].strip()

        zip_code = float(request.form["zip_code"])

        status = request.form["status"]


        # ==========================================
        # PREVIOUS SOLD INFORMATION
        # ==========================================

        sold_month_input = request.form.get(
            "sold_month",
            "never_sold"
        )

        sold_year_input = request.form.get(
            "sold_year",
            "N/A"
        )


        # ==========================================
        # HANDLE NEVER SOLD
        # ==========================================

        if (
            sold_month_input == "never_sold"
            or sold_month_input == ""
            or sold_month_input is None
        ):

            sold_month = 0
            sold_year = 0

        else:

            sold_month = float(sold_month_input)

            if (
                sold_year_input == ""
                or sold_year_input.upper() == "N/A"
            ):

                sold_year = 0

            else:

                sold_year = float(sold_year_input)


        # ==========================================
        # FEATURE ENGINEERING
        # ==========================================

        if sold_year == 0:

            # New house / never sold
            house_age = 0

        else:

            house_age = 2026 - sold_year

            if house_age < 0:
                house_age = 0


        # ==========================================
        # BEDROOM / BATHROOM RATIO
        # ==========================================

        if bath != 0:

            bed_bath_ratio = bed / bath

        else:

            bed_bath_ratio = 0


        # ==========================================
        # CREATE INPUT DATAFRAME
        # ==========================================

        new_house = pd.DataFrame({

            "bed": [bed],

            "bath": [bath],

            "house_size": [house_size],

            "acre_lot": [acre_lot],

            "zip_code": [zip_code],

            "city": [city],

            "state": [state],

            "status": [status],

            "sold_year": [sold_year],

            "sold_month": [sold_month],

            "house_age": [house_age],

            "bed_bath_ratio": [bed_bath_ratio]

        })


        # ==========================================
        # DISPLAY INPUT
        # ==========================================

        print("\n====================================")
        print("         HOUSE INPUT")
        print("====================================")

        print(new_house)

        print("====================================")


        # ==========================================
        # MAKE PREDICTION
        # ==========================================

        prediction = model.predict(new_house)[0]


        # ==========================================
        # DISPLAY RESULT
        # ==========================================

        return render_template(
            "index.html",
            prediction=f"${prediction:,.2f}"
        )


    # ==========================================
    # ERROR HANDLING
    # ==========================================

    except Exception as e:

        print("\nPrediction Error:", str(e))

        return render_template(
            "index.html",
            error=f"Prediction Error: {str(e)}"
        )


# ==========================================
# RUN APPLICATION
# ==========================================

if __name__ == "__main__":

    app.run(
        debug=True
    )
