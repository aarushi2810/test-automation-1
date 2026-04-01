from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
import os
import traceback
from datetime import datetime
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import DuplicateKeyError
from bson import ObjectId


MODEL_FILE = os.path.join(os.path.dirname(__file__), "model.pkl")
MONGO_URI  = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME    = os.environ.get("MONGO_DB",  "solar_predictions")

app = Flask(__name__)
CORS(app)

client          = MongoClient(MONGO_URI)
db              = client[DB_NAME]
predictions_col = db["predictions"]
messages_col    = db["messages"]
users_col       = db["users"]


def create_indexes():
    predictions_col.create_index([("timestamp",       DESCENDING)], name="idx_timestamp")
    predictions_col.create_index([("input.irradiance", ASCENDING)], name="idx_irradiance")
    predictions_col.create_index([("tags",             ASCENDING)], name="idx_tags")
    users_col.create_index([("email", ASCENDING)], unique=True,     name="idx_email_unique")
    print(" MongoDB indexes created")

create_indexes()


model = None
if os.path.exists(MODEL_FILE):
    try:
        with open(MODEL_FILE, "rb") as f:
            model = pickle.load(f)
        print(" Loaded model")
    except Exception as e:
        print(" Model load failed:", e)


def serialize(doc):
    if doc is None:
        return None
    doc = dict(doc)
    if "_id" in doc:
        doc["_id"] = str(doc["_id"])
    return doc

def tag_from_irradiance(irr):
    if irr >= 800:
        return "peak"
    elif irr >= 400:
        return "normal"
    return "low"


@app.route("/")
def home():
    return "<h3>Flask + MongoDB solar prediction service </h3>"

@app.route("/health")
def health():
    try:
        client.admin.command("ping")
        mongo_ok = "connected"
    except Exception as e:
        mongo_ok = f"error: {e}"
    return jsonify({"status": "running", "model_loaded": bool(model), "mongodb": mongo_ok, "database": DB_NAME})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        for k in ["irradiance","temp","prevHour","prevDay","roll3","roll6"]:
            if k not in data:
                return jsonify({"error": f"Missing field: {k}"}), 400

        if model is None:
            return jsonify({"error": "Model not loaded"}), 503

        irr  = float(data["irradiance"])
        temp = float(data["temp"])
        ph   = float(data["prevHour"])
        pd_  = float(data["prevDay"])
        r3   = float(data["roll3"])
        r6   = float(data["roll6"])

        pred = float(model.predict(np.array([[irr, temp, ph, pd_, r3, r6]]))[0])
        tag  = tag_from_irradiance(irr)

        document = {
            "input":  {"irradiance": irr, "temp": temp, "prevHour": ph,
                       "prevDay": pd_, "roll3": r3, "roll6": r6},   # nested subdocument
            "output": {"predictedPower": pred, "unit": "kW"},        # nested subdocument
            "tags":    [tag, "solar"],                                # array field
            "history": [{"step": "created", "value": pred}],         # array of objects
            "status":    "active",
            "viewCount": 0,
            "timestamp": datetime.utcnow()
        }

        result = predictions_col.insert_one(document)
        return jsonify({"predictedPower": pred, "_id": str(result.inserted_id), "tag": tag})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/logs", methods=["GET"])
def logs():
    try:
        limit = int(request.args.get("limit", 200))
        docs  = predictions_col.find({}, sort=[("timestamp", DESCENDING)]).limit(limit)
        return jsonify([serialize(d) for d in docs])
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/update/<doc_id>", methods=["PUT"])
def update(doc_id):
    try:
        data       = request.get_json() or {}
        new_status = data.get("status", "reviewed")

        result = predictions_col.update_one(
            {"_id": ObjectId(doc_id)},
            {
                "$set":  {"status": new_status},
                "$inc":  {"viewCount": 1},
                "$push": {"history": {"step": "status_updated", "value": new_status,
                                      "at": datetime.utcnow().isoformat()}}
            }
        )

        if result.matched_count == 0:
            return jsonify({"error": "Document not found"}), 404

        return jsonify({"message": "Updated", "modified": result.modified_count})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/delete/<doc_id>", methods=["DELETE"])
def delete(doc_id):
    try:
        result = predictions_col.delete_one({"_id": ObjectId(doc_id)})
        if result.deleted_count == 0:
            return jsonify({"error": "Not found"}), 404
        return jsonify({"message": "Deleted", "deleted": result.deleted_count})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/filter", methods=["GET"])
def filter_predictions():
    try:
        query = {}

        # nested field: input.irradiance
        irr = {}
        if request.args.get("min_irradiance"):
            irr["$gte"] = float(request.args["min_irradiance"])
        if request.args.get("max_irradiance"):
            irr["$lte"] = float(request.args["max_irradiance"])
        if irr:
            query["input.irradiance"] = irr

        # nested field: output.predictedPower
        out = {}
        if request.args.get("min_output"):
            out["$gte"] = float(request.args["min_output"])
        if request.args.get("max_output"):
            out["$lte"] = float(request.args["max_output"])
        if out:
            query["output.predictedPower"] = out

        if request.args.get("tag"):
            query["tags"] = request.args["tag"]        # array element query

        if request.args.get("status"):
            query["status"] = request.args["status"]

        docs   = predictions_col.find(query, sort=[("timestamp", DESCENDING)]).limit(100)
        result = [serialize(d) for d in docs]

        return jsonify({"query": str(query), "count": len(result), "results": result})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/stats", methods=["GET"])
def stats():
    try:
        # Stage 1: overall summary
        overall = list(predictions_col.aggregate([
            {"$group": {
                "_id":        None,
                "totalCount": {"$sum": 1},
                "avgOutput":  {"$avg": "$output.predictedPower"},
                "maxOutput":  {"$max": "$output.predictedPower"},
                "minOutput":  {"$min": "$output.predictedPower"},
                "avgIrr":     {"$avg": "$input.irradiance"}
            }}
        ]))

        # Stage 2: per-tag breakdown (uses $unwind on array)
        by_tag = list(predictions_col.aggregate([
            {"$unwind": "$tags"},
            {"$group": {
                "_id":       "$tags",
                "count":     {"$sum": 1},
                "avgOutput": {"$avg": "$output.predictedPower"}
            }},
            {"$sort": {"count": DESCENDING}}
        ]))

        # Stage 3: time-series (last 7 points)
        ts = list(predictions_col.aggregate([
            {"$sort":  {"timestamp": DESCENDING}},
            {"$limit": 7},
            {"$project": {
                "timestamp":      1,
                "predictedPower": "$output.predictedPower",
                "irradiance":     "$input.irradiance",
                "tag":            {"$arrayElemAt": ["$tags", 0]}
            }},
            {"$sort": {"timestamp": ASCENDING}}
        ]))

        summary = overall[0] if overall else {}
        summary.pop("_id", None)

        return jsonify({
            "overall":    summary,
            "by_tag":     [{"tag": t["_id"], "count": t["count"], "avgOutput": round(t["avgOutput"],3)} for t in by_tag],
            "timeSeries": [serialize(r) for r in ts]
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/collections", methods=["GET"])
def collections():
    try:
        names  = db.list_collection_names()
        result = [{"collection": n, "documentCount": db[n].count_documents({})} for n in names]
        return jsonify({"database": DB_NAME, "collections": result})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/contact", methods=["POST"])
def contact():
    try:
        data    = request.get_json()
        name    = data.get("name","").strip()
        email   = data.get("email","").strip()
        message = data.get("message","").strip()

        if not name or not email or not message:
            return jsonify({"error": "name, email and message are required"}), 400

        messages_col.insert_one({"name": name, "email": email,
                                  "message": message, "timestamp": datetime.utcnow()})
        return jsonify({"status": "ok", "message": "Thanks — message received."})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/register", methods=["POST"])
def register():
    try:
        data     = request.get_json()
        name     = data.get("name","").strip()
        email    = data.get("email","").strip()
        password = data.get("password","").strip()

        if not name or not email or not password:
            return jsonify({"error": "All fields are required"}), 400

        users_col.insert_one({"name": name, "email": email,
                               "password": password, "created_at": datetime.utcnow()})
        return jsonify({"message": "Registration successful"}), 201

    except DuplicateKeyError:
        return jsonify({"error": "Email already registered"}), 409
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/login", methods=["POST"])
def login():
    try:
        data     = request.get_json()
        email    = data.get("email","").strip()
        password = data.get("password","").strip()

        if not email or not password:
            return jsonify({"error": "Email and password required"}), 400

        user = users_col.find_one({"email": email, "password": password})
        if not user:
            return jsonify({"error": "Invalid email or password"}), 401

        return jsonify({"message": "Login successful",
                        "user": {"id": str(user["_id"]),
                                 "name": user["name"], "email": user["email"]}}), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)
