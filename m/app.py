from flask import Flask, render_template, request, redirect, url_for, session, send_file
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')  # Use this to prevent Tkinter errors
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import plotly.graph_objs as go
import plotly.offline as pyo
from plotly.subplots import make_subplots
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO
app = Flask(__name__)
app.secret_key = 'your_unique_and_secret_key_here'
# Load dataset
df = pd.read_csv("meals.csv")
# Prepare the dataset for machine learning
features = df[['Ages', 'Height', 'Weight', 'Activity Level', 'Dietary Preference', 'Disease']]
target = df['Daily Calorie Target']
# Convert categorical variables to dummy/indicator variables
features = pd.get_dummies(features)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
# Initialize the RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
# Temporary storage for users (use a database in production)
users = {}
# Temporary storage for meal logs
user_meal_log = {}
@app.route("/")
def home():
    return render_template("home.html")
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if username in users:
            return render_template("signup.html", error="User already exists!")
        users[username] = password
        return redirect(url_for("login"))
    return render_template("signup.html")
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if username in users and users[username] == password:
            session["username"] = username
            return redirect(url_for("index"))
        else:
            return render_template("login.html", error="Invalid credentials!")
    return render_template("login.html")
@app.route("/index", methods=["GET", "POST"])
def index():
    if not session.get("username"):
        return redirect(url_for("login"))
    if request.method == "POST":
        age = request.form.get("age")
        gender = request.form.get("gender")
        height = request.form.get("height")
        weight = request.form.get("weight")
        activity_level = request.form.get("activity_level")
        dietary_preference = request.form.get("dietary_preference")
        disease = request.form.get("disease")
        if not age or not height or not weight:
            return render_template("index.html", error="Please fill all required fields!")
        try:
            age = int(age)
            height = int(height)
            weight = int(weight)
        except ValueError:
            return render_template("index.html", error="Invalid numeric input!")

        input_data = pd.DataFrame({
            'Ages': [age],
            'Height': [height],
            'Weight': [weight],
            'Activity Level': [activity_level],
            'Dietary Preference': [dietary_preference],
            'Disease': [disease]
        })
        input_data = pd.get_dummies(input_data)
        input_data = input_data.reindex(columns=features.columns, fill_value=0)
        predicted_calories = model.predict(input_data)[0]
        session["user_input"] = {
            "age": age,
            "gender": gender,
            "height": height,
            "weight": weight,
            "activity_level": activity_level,
            "dietary_preference": dietary_preference,
            "disease": disease,
            "predicted_calories": predicted_calories
        }
        filtered_data = df[
            (df["Ages"] == age) & (df["Gender"] == gender) &
            (df["Dietary Preference"] == dietary_preference) & 
            (df["Disease"].str.contains(disease, na=False))
        ]
        meal_data = filtered_data.iloc[0].to_dict() if not filtered_data.empty else {"error": "No matching meal plan found!"}
        session["meal_data"] = meal_data
        return redirect(url_for("dashboard"))
    return render_template("index.html")
@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if not session.get("username"):
        return redirect(url_for("login"))
    meal_data = session.get("meal_data", {})
    user_input = session.get("user_input", {})
    if request.method == "POST":
        return redirect(url_for("meal-log"))
    return render_template("dashboard.html", meal=meal_data, user_input=user_input, predicted_calories=user_input.get("predicted_calories", 0))
@app.route("/log-meal", methods=["GET", "POST"])
def log_meal():
    username = session.get("username")
    if not username:
        return redirect(url_for("login"))

    # Initialize daily_totals to prevent UndefinedError
    daily_totals = {"calories": 0, "proteins": 0, "fats": 0, "carbs": 0}

    if request.method == "POST":
        meal_name = request.form.get("meal_name")
        portion_size = request.form.get("portion_size")
        if not meal_name or not portion_size:
            return render_template("meal_log.html", error="Please enter both meal name and portion size!", daily_totals=daily_totals)
        
        try:
            portion_size = float(portion_size)
        except ValueError:
            return render_template("meal_log.html", error="Invalid portion size!", daily_totals=daily_totals)

        meal_info = None
        for column in ['Breakfast Suggestion', 'Lunch Suggestion', 'Dinner Suggestion', 'Snacks Suggestion']:
            meal_info = df[df[column].str.lower() == meal_name.lower()]
            if not meal_info.empty:
                break

        if meal_info.empty:
            return render_template("meal_log.html", error="Meal not found in the database!", daily_totals=daily_totals)

        meal_info = meal_info.iloc[0]

        total_calories = round(meal_info["Calories"] * portion_size, 2)
        total_proteins = round(meal_info["Protein"] * portion_size, 2)
        total_fats = round(meal_info["Fat"] * portion_size, 2)
        total_carbs = round(meal_info["Carbohydrates"] * portion_size, 2)

        if username not in user_meal_log:
            user_meal_log[username] = []
        user_meal_log[username].append({
            "meal": meal_name,
            "portion": portion_size,
            "calories": total_calories,
            "proteins": total_proteins,
            "fats": total_fats,
            "carbs": total_carbs
        })
        return redirect(url_for("log_meal"))

    meal_log = user_meal_log.get(username, [])

    if meal_log:
        for entry in meal_log:
            daily_totals["calories"] += entry["calories"]
            daily_totals["proteins"] += entry["proteins"]
            daily_totals["fats"] += entry["fats"]
            daily_totals["carbs"] += entry["carbs"]

    return render_template("meal_log.html", meal_log=meal_log, daily_totals=daily_totals)

@app.route("/graph", methods=["GET"])
def graph():
    username = session.get("username")
    if not username:
        return redirect(url_for("login"))

    meal_log = user_meal_log.get(username, [])
    if not meal_log:
        return redirect(url_for("log_meal"))

    daily_totals = {"calories": 0, "proteins": 0, "fats": 0, "carbs": 0}
    for entry in meal_log:
        daily_totals["calories"] += entry["calories"]
        daily_totals["proteins"] += entry["proteins"]
        daily_totals["fats"] += entry["fats"]
        daily_totals["carbs"] += entry["carbs"]

    calories = daily_totals["calories"]
    proteins = daily_totals["proteins"]
    fats = daily_totals["fats"]
    carbohydrates = daily_totals["carbs"]

    recommended_fats = 70
    recommended_calories = 2500
    recommended_carbohydrates = 300
    recommended_proteins = 100

    # Create subplots with specific types
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Daily Nutrient Intake", "Nutrient Breakdown", 
                        "Calorie & Nutrient Comparison", "Actual vs Recommended Nutrient Intake"),
        specs=[
            [{"type": "xy"}, {"type": "domain"}],  # First row: bar chart and pie chart
            [{"type": "xy"}, {"type": "xy"}]       # Second row: bar chart and line chart
        ]
    )

    # Bar chart for Daily Nutrient Intake
    fig.add_trace(go.Bar(x=["Calories", "Proteins", "Fats", "Carbs"], 
                         y=[calories, proteins, fats, carbohydrates], 
                         marker_color=['blue', 'green', 'orange', 'red']), 
                  row=1, col=1)

    # Pie chart for Nutrient Breakdown
    fig.add_trace(go.Pie(labels=["Fats", "Calories", "Carbohydrates", "Proteins"], 
                         values=[fats, calories, carbohydrates, proteins], 
                         marker_colors=['gold', 'red', 'orange', 'blue']), 
                  row=1, col=2)

    # Bar chart for Calorie & Nutrient Comparison
    fig.add_trace(go.Bar(x=["Calories", "Proteins", "Fats", "Carbs"], 
                         y=[calories, proteins, fats, carbohydrates], 
                         marker_color=['red', 'blue', 'gold', 'orange']), 
                  row=2, col=1)

    # Line chart for Actual vs Recommended Nutrient Intake
    fig.add_trace(go.Scatter(x=["Calories", "Fats", "Proteins", "Carbohydrates"], 
                             y=[calories, fats, proteins, carbohydrates], 
                             mode='lines+markers', name="Actual Intake", line=dict(color='blue')), 
                  row=2, col=2)
    fig.add_trace(go.Scatter(x=["Calories", "Fats", "Proteins", "Carbohydrates"], 
                             y=[recommended_calories, recommended_fats, recommended_proteins, recommended_carbohydrates], 
                             mode='lines+markers', name="Recommended Intake", line=dict(color='red', dash='dash')), 
                  row=2, col=2)

    # Update layout for better visualization
    fig.update_layout(height=800, width=1200, title_text="Nutrient Analysis", showlegend=True)
    fig.update_xaxes(title_text="Nutrients", row=1, col=1)
    fig.update_yaxes(title_text="Amount", row=1, col=1)
    fig.update_xaxes(title_text="Nutrients", row=2, col=1)
    fig.update_yaxes(title_text="Amount", row=2, col=1)
    fig.update_xaxes(title_text="Nutrient Type", row=2, col=2)
    fig.update_yaxes(title_text="Amount", row=2, col=2)

    # Save the plot to a HTML file
    plot_html = pyo.plot(fig, output_type='div')

    return render_template("graph.html", plot_html=plot_html)

@app.route("/model_performance")
def model_performance():
    if not session.get("username"):
        return redirect(url_for("login"))

    # Calculate model performance metrics
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return render_template("model_performance.html", mse=mse, mae=mae, r2=r2)

@app.route("/logout")
def logout():
    session.pop("username", None)
    return redirect(url_for("home"))


@app.route("/download-pdf")
def download_pdf():
    username = session.get("username")
    if not username:
        return redirect(url_for("login"))

    meal_log = user_meal_log.get(username, [])
    if not meal_log:
        return redirect(url_for("log_meal"))

    # Calculate daily totals
    daily_totals = {"calories": 0, "proteins": 0, "fats": 0, "carbs": 0}
    for entry in meal_log:
        daily_totals["calories"] += entry["calories"]
        daily_totals["proteins"] += entry["proteins"]
        daily_totals["fats"] += entry["fats"]
        daily_totals["carbs"] += entry["carbs"]

    # Create a PDF
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    pdf.setTitle("Meal Log Report")

    # Add content to the PDF
    pdf.drawString(100, 750, "Meal Log Report")
    pdf.drawString(100, 730, f"User: {username}")
    pdf.drawString(100, 710, "Daily Nutrient Totals:")
    pdf.drawString(100, 690, f"Calories: {daily_totals['calories']}")
    pdf.drawString(100, 670, f"Proteins: {daily_totals['proteins']}g")
    pdf.drawString(100, 650, f"Fats: {daily_totals['fats']}g")
    pdf.drawString(100, 630, f"Carbs: {daily_totals['carbs']}g")

    # Add meal log details
    pdf.drawString(100, 600, "Logged Meals:")
    y = 580
    for entry in meal_log:
        pdf.drawString(100, y, f"Meal: {entry['meal']}, Portion: {entry['portion']}g")
        pdf.drawString(120, y - 15, f"Calories: {entry['calories']}, Proteins: {entry['proteins']}g, Fats: {entry['fats']}g, Carbs: {entry['carbs']}g")
        y -= 30
        if y < 100:  # Add a new page if content exceeds page height
            pdf.showPage()
            y = 750

    # Save the PDF
    pdf.save()
    buffer.seek(0)

    # Return the PDF as a downloadable file
    return send_file(buffer, as_attachment=True, download_name="meal_log_report.pdf", mimetype="application/pdf")
@app.route("/meal-history")
def meal_history():
    username = session.get("username")
    if not username:
        return redirect(url_for("login"))

    meal_log = user_meal_log.get(username, [])
    return render_template("meal_history.html", meal_log=meal_log)
@app.route("/meal-recommendations", methods=["GET", "POST"])
def meal_recommendations():
    username = session.get("username")
    if not username:
        return redirect(url_for("login"))

    if request.method == "POST":
        # Get user preferences from the form
        dietary_preference = request.form.get("dietary_preference")
        activity_level = request.form.get("activity_level")
        disease = request.form.get("disease")

        # Filter the dataset based on user preferences
        filtered_data = df[
            (df["Dietary Preference"] == dietary_preference) &
            (df["Activity Level"] == activity_level) &
            (df["Disease"].str.contains(disease, na=False))
        ]

        # Get top 3 recommendations
        recommendations = filtered_data.head(3).to_dict("records")

        # Store recommendations in session
        session["recommendations"] = recommendations
        return redirect(url_for("meal_recommendations"))

    # Retrieve recommendations from session
    recommendations = session.get("recommendations", [])
    return render_template("meal_recommendations.html", recommendations=recommendations)
@app.route("/about")
def about():
    return render_template("about.html")
@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        message = request.form.get("message")
        return render_template("contact.html", success="Message sent successfully!")
    return render_template("contact.html")

if __name__ == "__main__":
    app.run(debug=True)