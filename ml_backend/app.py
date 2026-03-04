from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import pickle
import sqlite3
import os
import traceback


PORT = 5001
DB_FILE = "predictions.db"
MODEL_FILE =  os.path.join(os.path.dirname(__file__), "model.pkl")


app = Flask(__name__)
CORS(app)




model = None
if os.path.exists(MODEL_FILE):
    try:
        with open(MODEL_FILE, "rb") as f:
            model = pickle.load(f)
        print(" Loaded model:", MODEL_FILE)
    except Exception as e:
        print(" Failed to load model:", e)
else:
    print(" Model file not found:", MODEL_FILE, " — /predict will fail until model exists.")

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            irradiance REAL,
            temp REAL,
            prevHour REAL,
            prevDay REAL,
            roll3 REAL,
            roll6 REAL,
            prediction REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT,
            message TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()
    print("✅ Database initialized with users table")

init_db()

@app.route("/")
def home():
    return "<h3>Flask ML microservice running. Use POST /predict and POST /contact</h3>"

@app.route("/health")
def health():
    return jsonify({"status": "running", "model_loaded": bool(model)})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        required = ["irradiance", "temp", "prevHour", "prevDay", "roll3", "roll6"]
        for k in required:
            if k not in data:
                return jsonify({"error": f"missing field {k}"}), 400

        features = np.array([[
            float(data["irradiance"]),
            float(data["temp"]),
            float(data["prevHour"]),
            float(data["prevDay"]),
            float(data["roll3"]),
            float(data["roll6"])
        ]])

        if model is None:
            return jsonify({"error": "model not loaded on server"}), 503

        prediction = model.predict(features)[0]
        pred_value = float(prediction)

        
        conn = sqlite3.connect(DB_FILE)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO predictions (irradiance, temp, prevHour, prevDay, roll3, roll6, prediction)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            float(data["irradiance"]), float(data["temp"]), float(data["prevHour"]),
            float(data["prevDay"]), float(data["roll3"]), float(data["roll6"]), pred_value
        ))
        conn.commit()
        conn.close()

        return jsonify({"predictedPower": pred_value})
    except Exception as e:
        print("Predict error:", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/logs", methods=["GET"])
def logs():
    try:
        conn = sqlite3.connect(DB_FILE)
        cur = conn.cursor()
        cur.execute("SELECT id, irradiance, temp, prevHour, prevDay, roll3, roll6, prediction, timestamp FROM predictions ORDER BY timestamp DESC LIMIT 200")
        rows = cur.fetchall()
        conn.close()
        out = []
        for r in rows:
            out.append({
                "id": r[0],
                "irradiance": r[1],
                "temp": r[2],
                "prevHour": r[3],
                "prevDay": r[4],
                "roll3": r[5],
                "roll6": r[6],
                "prediction": r[7],
                "timestamp": r[8],
            })
        return jsonify(out)
    except Exception as e:
        print("Logs error:", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/contact", methods=["POST"])
def contact():
    try:
        data = request.get_json()
        name = data.get("name", "").strip()
        email = data.get("email", "").strip()
        message = data.get("message", "").strip()

        
        if not name or not email or not message:
            return jsonify({"error": "name, email and message are required"}), 400

        
        print("CONTACT MESSAGE:", name, email, message)

       
        conn = sqlite3.connect(DB_FILE)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO messages (name, email, message) VALUES (?, ?, ?)
        """, (name, email, message))
        conn.commit()
        conn.close()

        return jsonify({"status": "ok", "message": "Thanks — message received."})
    except Exception as e:
        print("Contact error:", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
@app.route("/register", methods=["POST"])
def register():
    try:
        data = request.get_json()

        name = data.get("name", "").strip()
        email = data.get("email", "").strip()
        password = data.get("password", "").strip()

        if not name or not email or not password:
            return jsonify({"error": "All fields are required"}), 400

        conn = sqlite3.connect(DB_FILE)
        cur = conn.cursor()

        # check if email already exists
        cur.execute("SELECT id FROM users WHERE email = ?", (email,))
        if cur.fetchone():
            conn.close()
            return jsonify({"error": "Email already registered"}), 409

        cur.execute("""
            INSERT INTO users (name, email, password)
            VALUES (?, ?, ?)
        """, (name, email, password))

        conn.commit()
        conn.close()

        return jsonify({"message": "Registration successful"}), 201

    except Exception as e:
        print("Register error:", e)
        return jsonify({"error": str(e)}), 500
@app.route("/login", methods=["POST"])
def login():
    try:
        data = request.get_json()
        email = data.get("email", "").strip()
        password = data.get("password", "").strip()

        if not email or not password:
            return jsonify({"error": "Email and password required"}), 400

        conn = sqlite3.connect(DB_FILE)
        cur = conn.cursor()

        cur.execute(
            "SELECT id, name, email FROM users WHERE email = ? AND password = ?",
            (email, password)
        )
        user = cur.fetchone()
        conn.close()

        if not user:
            return jsonify({"error": "Invalid email or password"}), 401

        return jsonify({
            "message": "Login successful",
            "user": {
                "id": user[0],
                "name": user[1],
                "email": user[2]
            }
        }), 200

    except Exception as e:
        print("Login error:", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
