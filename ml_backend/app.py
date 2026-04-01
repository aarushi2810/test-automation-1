from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
import os
import traceback
from datetime import datetime
from pymongo import MongoClient, ASCENDING, DESCENDING, TEXT
from pymongo.errors import DuplicateKeyError, OperationFailure
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
analytics_col   = db["analytics"]          # extra collection for schema variety


def create_indexes():
  
    predictions_col.create_index(
        [("timestamp", DESCENDING)],
        name="idx_timestamp_desc"
    )
    predictions_col.create_index(
        [("input.irradiance", ASCENDING)],
        name="idx_irradiance_asc"
    )
    predictions_col.create_index(
        [("tags", ASCENDING)],
        name="idx_tags"
    )

    # ── Compound index (irradiance + timestamp together) ──────────
    predictions_col.create_index(
        [("input.irradiance", ASCENDING), ("timestamp", DESCENDING)],
        name="idx_compound_irr_time"
    )

   
    predictions_col.create_index(
        [("output.predictedPower", DESCENDING)],
        sparse=True,
        name="idx_sparse_power"
    )

    
    predictions_col.create_index(
        [("input.irradiance", DESCENDING)],
        partialFilterExpression={"tags": "peak"},
        name="idx_partial_peak_irr"
    )

   
    users_col.create_index(
        [("email", ASCENDING)],
        unique=True,
        name="idx_email_unique"
    )

    print(" All MongoDB indexes created")

create_indexes()

model = None
if os.path.exists(MODEL_FILE):
    try:
        with open(MODEL_FILE, "rb") as f:
            model = pickle.load(f)
        print(" Model loaded")
    except Exception as e:
        print("Model load failed:", e)


def serialize(doc):
    """Convert MongoDB document → JSON-safe dict."""
    if doc is None:
        return None
    doc = dict(doc)
    doc["_id"] = str(doc["_id"]) if "_id" in doc else None
    return doc

def tag_from_irradiance(irr: float) -> str:
    """
    Solar domain logic:
      peak   → irradiance ≥ 800  (strong sunlight, max power generation)
      normal → irradiance ≥ 400  (moderate sunlight)
      low    → irradiance < 400  (weak/cloudy conditions)
    """
    if irr >= 800:
        return "peak"
    elif irr >= 400:
        return "normal"
    return "low"

def parse_projection(fields_str: str) -> dict:
    """Turn comma-separated field names into a MongoDB projection dict."""
    if not fields_str:
        return {}
    return {f.strip(): 1 for f in fields_str.split(",") if f.strip()}


@app.route("/")
def home():
    return "<h3>Solar Prediction — Flask + MongoDB </h3>"

@app.route("/health")
def health():
    """Shows MongoDB connection status and loaded model."""
    try:
        client.admin.command("ping")
        mongo_ok = "connected"
    except Exception as e:
        mongo_ok = f"error: {e}"
    return jsonify({
        "status":       "running",
        "model_loaded": bool(model),
        "mongodb":      mongo_ok,
        "database":     DB_NAME
    })


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
            "input": {                          # nested subdocument
                "irradiance": irr,
                "temp":       temp,
                "prevHour":   ph,
                "prevDay":    pd_,
                "roll3":      r3,
                "roll6":      r6
            },
            "output": {                         # nested subdocument
                "predictedPower": pred,
                "unit":           "kW"
            },
            "tags":    [tag, "solar"],          # array field
            "history": [                        # array of embedded objects
                {
                    "step":      "prediction_created",
                    "value":     pred,
                    "timestamp": datetime.utcnow().isoformat()
                }
            ],
            "status":    "active",
            "viewCount": 0,                     # for $inc demo
            "timestamp": datetime.utcnow()
        }
        

        result = predictions_col.insert_one(document)

        return jsonify({
            "predictedPower": pred,
            "_id":            str(result.inserted_id),
            "tag":            tag,
            "description":    f"Irradiance {irr} W/m² → {tag.upper()} solar conditions"
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/logs", methods=["GET"])
def logs():
    """
    MongoDB concepts:
      ✔ find()       — retrieve all documents
      ✔ sort()       — order by timestamp
      ✔ limit()      — cap number of results
      ✔ skip()       — pagination offset
      ✔ count        — total matching documents
      ✔ projection   — return only requested fields
    Query params:
      limit   (default 50)
      skip    (default 0)
      fields  (comma-separated field names for projection)
      sort    (asc | desc)
    """
    try:
        limit      = int(request.args.get("limit", 50))
        skip_val   = int(request.args.get("skip", 0))
        sort_dir   = ASCENDING if request.args.get("sort") == "asc" else DESCENDING
        fields_str = request.args.get("fields", "")
        projection = parse_projection(fields_str) or None

        total = predictions_col.count_documents({})
        cursor = (
            predictions_col
            .find({}, projection)
            .sort("timestamp", sort_dir)
            .skip(skip_val)
            .limit(limit)
        )

        return jsonify({
            "total":      total,
            "skip":       skip_val,
            "limit":      limit,
            "returned":   cursor.retrieved if hasattr(cursor, "retrieved") else limit,
            "results":    [serialize(d) for d in cursor]
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/prediction/<doc_id>", methods=["GET"])
def find_one(doc_id):
    """
    MongoDB concept:
      ✔ find_one() — retrieve a single document by _id
    """
    try:
        doc = predictions_col.find_one({"_id": ObjectId(doc_id)})
        if not doc:
            return jsonify({"error": "Document not found"}), 404
        return jsonify(serialize(doc))
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/search", methods=["GET"])
def search():
    """
    MongoDB concepts:
      ✔ $or  — match documents where ANY condition is true
      ✔ $in  — match documents where field value is in a list
    Query params:
      tags   (comma-separated: peak,normal,low)  → $in on tags array
      status (comma-separated: active,reviewed)  → $in on status
      use_or (true/false) — combine irradiance OR output with $or
    Example:
      /search?tags=peak,normal   → {tags: {$in: ["peak","normal"]}}
      /search?status=active,reviewed
    """
    try:
        query        = {}
        or_clauses   = []

        # $in on tags array
        if request.args.get("tags"):
            tag_list      = [t.strip() for t in request.args["tags"].split(",")]
            query["tags"] = {"$in": tag_list}

        # $in on status field
        if request.args.get("status"):
            status_list      = [s.strip() for s in request.args["status"].split(",")]
            query["status"]  = {"$in": status_list}

        # $or — find predictions with HIGH irradiance OR HIGH output
        if request.args.get("use_or") == "true":
            or_clauses = [
                {"input.irradiance":      {"$gte": 700}},
                {"output.predictedPower": {"$gte": 200}}
            ]
            query["$or"] = or_clauses

        docs   = predictions_col.find(query, sort=[("timestamp", DESCENDING)]).limit(100)
        result = [serialize(d) for d in docs]

        return jsonify({
            "query":   str(query),
            "count":   len(result),
            "results": result
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/filter", methods=["GET"])
def filter_predictions():
    """
    MongoDB concepts:
      ✔ AND query (multiple conditions combined)
      ✔ $gte, $lte range operators on nested fields
      ✔ Dot-notation to query nested subdocuments
      ✔ Array element query {tags: "peak"}
    Query params:
      min_irradiance, max_irradiance
      min_output,     max_output
      tag             (peak | normal | low)
      status          (active | reviewed)
    """
    try:
        query = {}

        # AND condition on nested field input.irradiance
        irr = {}
        if request.args.get("min_irradiance"):
            irr["$gte"] = float(request.args["min_irradiance"])
        if request.args.get("max_irradiance"):
            irr["$lte"] = float(request.args["max_irradiance"])
        if irr:
            query["input.irradiance"] = irr

        # AND condition on nested field output.predictedPower
        out = {}
        if request.args.get("min_output"):
            out["$gte"] = float(request.args["min_output"])
        if request.args.get("max_output"):
            out["$lte"] = float(request.args["max_output"])
        if out:
            query["output.predictedPower"] = out

        # Array element match
        if request.args.get("tag"):
            query["tags"] = request.args["tag"]

        if request.args.get("status"):
            query["status"] = request.args["status"]

        docs   = predictions_col.find(query, sort=[("timestamp", DESCENDING)]).limit(100)
        result = [serialize(d) for d in docs]

        return jsonify({
            "mongo_query": str(query),
            "count":       len(result),
            "results":     result
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/update/<doc_id>", methods=["PUT"])
def update(doc_id):
    """
    MongoDB concepts:
      ✔ update_one
      ✔ $set   — set/overwrite a field
      ✔ $inc   — increment a numeric field
      ✔ $push  — append to an array
      ✔ $mul   — multiply a numeric field (scales output.predictedPower for demo)
      ✔ $unset — remove a field entirely (optional)
      ✔ Dot-notation update of embedded document field
    Body:
      { "status": "reviewed", "note": "optional note", "unset_field": "fieldname" }
    """
    try:
        data       = request.get_json() or {}
        new_status = data.get("status", "reviewed")

        update_ops = {
            "$set": {
                "status":               new_status,
                "output.lastReviewed":  datetime.utcnow().isoformat()  # embedded doc update
            },
            "$inc":  {"viewCount": 1},                                 # increment operator
            "$push": {                                                  # append to array
                "history": {
                    "step":      "status_updated",
                    "value":     new_status,
                    "at":        datetime.utcnow().isoformat()
                }
            }
        }

        # Optional: add a reviewer note via $set
        if data.get("note"):
            update_ops["$set"]["reviewNote"] = data["note"]

        # Optional: remove a field via $unset
        if data.get("unset_field"):
            update_ops["$unset"] = {data["unset_field"]: ""}

        result = predictions_col.update_one(
            {"_id": ObjectId(doc_id)},
            update_ops
        )

        if result.matched_count == 0:
            return jsonify({"error": "Document not found"}), 404

        return jsonify({
            "message":  "Document updated",
            "modified": result.modified_count,
            "operators_used": ["$set", "$inc", "$push"]
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/bulk-update", methods=["PUT"])
def bulk_update():
    """
    MongoDB concept:
      ✔ update_many — update ALL documents matching a filter
    Use case: Mark all "low" irradiance predictions as "needs_review"
    Body:
      { "tag": "low", "new_status": "needs_review" }
    """
    try:
        data       = request.get_json() or {}
        tag        = data.get("tag", "low")
        new_status = data.get("new_status", "needs_review")

        result = predictions_col.update_many(
            {"tags": tag},                          # filter — all docs with this tag
            {
                "$set":  {"status": new_status},
                "$inc":  {"viewCount": 1},
                "$push": {
                    "history": {
                        "step":  "bulk_updated",
                        "value": new_status,
                        "at":    datetime.utcnow().isoformat()
                    }
                }
            }
        )

        return jsonify({
            "message":       f"Bulk update complete for tag='{tag}'",
            "matched":       result.matched_count,
            "modified":      result.modified_count,
            "filter_used":   {"tags": tag},
            "update_applied": {"status": new_status}
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/delete/<doc_id>", methods=["DELETE"])
def delete(doc_id):
    """MongoDB concept: delete_one"""
    try:
        result = predictions_col.delete_one({"_id": ObjectId(doc_id)})
        if result.deleted_count == 0:
            return jsonify({"error": "Document not found"}), 404
        return jsonify({"message": "Deleted", "deleted_count": result.deleted_count})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/stats", methods=["GET"])
def stats():
    """
    MongoDB concepts:
      ✔ Aggregation pipeline
      ✔ $match  — filter documents before grouping
      ✔ $group  — group + compute ($avg, $max, $min, $sum)
      ✔ $unwind — deconstruct array field
      ✔ $project — shape the output
      ✔ $sort   — order results
      ✔ $limit  — cap output
    """
    try:
        
        overall = list(predictions_col.aggregate([
            {
                "$group": {
                    "_id":        None,
                    "totalCount": {"$sum": 1},
                    "avgOutput":  {"$avg": "$output.predictedPower"},
                    "maxOutput":  {"$max": "$output.predictedPower"},
                    "minOutput":  {"$min": "$output.predictedPower"},
                    "avgIrr":     {"$avg": "$input.irradiance"},
                    "maxIrr":     {"$max": "$input.irradiance"}
                }
            }
        ]))

        # ── Pipeline 2: Per-tag breakdown (uses $unwind on array) ─
        by_tag = list(predictions_col.aggregate([
            {"$unwind": "$tags"},                   # flatten tags array
            {
                "$group": {
                    "_id":       "$tags",
                    "count":     {"$sum": 1},
                    "avgOutput": {"$avg": "$output.predictedPower"},
                    "maxOutput": {"$max": "$output.predictedPower"}
                }
            },
            {"$sort": {"count": DESCENDING}},
            {
                "$project": {                       # rename _id → tag
                    "_id":    0,
                    "tag":    "$_id",
                    "count":  1,
                    "avgOutput": {"$round": ["$avgOutput", 2]},
                    "maxOutput": {"$round": ["$maxOutput", 2]}
                }
            }
        ]))

        # ── Pipeline 3: Time-series (last 10 predictions) ────────
        ts = list(predictions_col.aggregate([
            {"$sort":  {"timestamp": DESCENDING}},
            {"$limit": 10},
            {
                "$project": {
                    "timestamp":      1,
                    "predictedPower": "$output.predictedPower",
                    "irradiance":     "$input.irradiance",
                    "tag":            {"$arrayElemAt": ["$tags", 0]}
                }
            },
            {"$sort": {"timestamp": ASCENDING}}
        ]))

        # ── Pipeline 4: Peak hours analysis ──────────────────────
        peak_pipeline = list(predictions_col.aggregate([
            {"$match": {"tags": "peak"}},          # $match stage
            {
                "$group": {
                    "_id":        None,
                    "peakCount":  {"$sum": 1},
                    "avgPower":   {"$avg": "$output.predictedPower"},
                    "maxPower":   {"$max": "$output.predictedPower"},
                    "avgIrr":     {"$avg": "$input.irradiance"}
                }
            }
        ]))

        summary = overall[0] if overall else {}
        summary.pop("_id", None)

        peak_summary = peak_pipeline[0] if peak_pipeline else {}
        peak_summary.pop("_id", None)

        # Round floats for readability
        for k, v in summary.items():
            if isinstance(v, float):
                summary[k] = round(v, 3)

        return jsonify({
            "overall":    summary,
            "by_tag":     by_tag,
            "timeSeries": [serialize(r) for r in ts],
            "peakHours":  peak_summary
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/peak-hours", methods=["GET"])
def peak_hours():
    """
    Solar domain feature:
      Shows all predictions where irradiance ≥ 800 W/m² (peak sunlight).
      These are the moments when the solar panel generates maximum power.
    MongoDB concepts:
      ✔ $match with comparison operators
      ✔ Aggregation for analytics
      ✔ Projection to shape output
    """
    try:
        threshold = float(request.args.get("threshold", 800))

        # All peak predictions
        peak_docs = list(predictions_col.find(
            {"input.irradiance": {"$gte": threshold}},
            {
                "input.irradiance":      1,
                "input.temp":            1,
                "output.predictedPower": 1,
                "tags":                  1,
                "timestamp":             1
            },
            sort=[("input.irradiance", DESCENDING)]
        ).limit(50))

        # Aggregated summary for peak hours
        summary = list(predictions_col.aggregate([
            {"$match": {"input.irradiance": {"$gte": threshold}}},
            {
                "$group": {
                    "_id":       None,
                    "count":     {"$sum": 1},
                    "avgPower":  {"$avg": "$output.predictedPower"},
                    "maxPower":  {"$max": "$output.predictedPower"},
                    "avgIrr":    {"$avg": "$input.irradiance"},
                    "maxIrr":    {"$max": "$input.irradiance"}
                }
            }
        ]))

        s = summary[0] if summary else {}
        s.pop("_id", None)
        for k, v in s.items():
            if isinstance(v, float):
                s[k] = round(v, 2)

        return jsonify({
            "description":  f"Predictions where irradiance ≥ {threshold} W/m² (peak solar conditions)",
            "threshold":    threshold,
            "summary":      s,
            "predictions":  [serialize(d) for d in peak_docs]
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/collections", methods=["GET"])
def collections_info():
    """
    MongoDB concepts:
      ✔ list_collection_names()
      ✔ count_documents()
    """
    try:
        names  = db.list_collection_names()
        result = [
            {"collection": n, "documentCount": db[n].count_documents({})}
            for n in names
        ]
        return jsonify({"database": DB_NAME, "collections": result})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/drop-collection", methods=["DELETE"])
def drop_collection():
    """
    MongoDB concept:
      ✔ drop() — permanently remove a collection
    WARNING: Only allows dropping 'analytics' (safe demo collection).
    Body: { "collection": "analytics", "confirm": true }
    """
    try:
        data       = request.get_json() or {}
        col_name   = data.get("collection", "")
        confirmed  = data.get("confirm", False)

        # Safety: only allow dropping the demo analytics collection
        allowed = ["analytics"]
        if col_name not in allowed:
            return jsonify({
                "error":   f"Cannot drop '{col_name}'. Only allowed: {allowed}",
                "message": "This protects your real data."
            }), 403

        if not confirmed:
            return jsonify({"error": "Set confirm=true to proceed"}), 400

        db[col_name].drop()

        return jsonify({
            "message":    f"Collection '{col_name}' dropped successfully",
            "concept":    "db.collection.drop() removes the entire collection and its indexes"
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/indexes", methods=["GET"])
def get_indexes():
    """
    MongoDB concepts:
      ✔ list_indexes() — get all indexes on a collection
      ✔ Shows: single, compound, sparse, partial, unique indexes
    """
    try:
        col_name = request.args.get("collection", "predictions")
        col      = db[col_name]
        indexes  = list(col.list_indexes())

        result = []
        for idx in indexes:
            idx_info = {
                "name":   idx.get("name"),
                "keys":   dict(idx.get("key", {})),
                "unique": idx.get("unique", False),
                "sparse": idx.get("sparse", False),
            }
            if "partialFilterExpression" in idx:
                idx_info["partial_filter"] = dict(idx["partialFilterExpression"])
            result.append(idx_info)

        return jsonify({
            "collection":   col_name,
            "index_count":  len(result),
            "indexes":      result,
            "concepts": {
                "single_field":  "Index on one field — idx_timestamp_desc, idx_irradiance_asc",
                "compound":      "Index on multiple fields — idx_compound_irr_time",
                "sparse":        "Only indexes docs where the field EXISTS — idx_sparse_power",
                "partial":       "Only indexes docs matching a filter — idx_partial_peak_irr",
                "unique":        "Enforces uniqueness — idx_email_unique on users"
            }
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/db-info", methods=["GET"])
def db_info():
    """
    MongoDB concepts:
      ✔ list_database_names() — list all databases on the server
      ✔ Shows NoSQL vs SQL comparison for viva answers
    """
    try:
        all_dbs = client.list_database_names()
        stats   = db.command("dbStats")

        return jsonify({
            "current_database":  DB_NAME,
            "all_databases":     all_dbs,
            "db_stats": {
                "collections": stats.get("collections"),
                "objects":     stats.get("objects"),
                "dataSize_bytes": stats.get("dataSize"),
                "storageSize_bytes": stats.get("storageSize"),
                "indexes": stats.get("indexes")
            },
            "nosql_vs_sql": {
                "schema":      "MongoDB is schema-less; SQL requires fixed table schema",
                "scaling":     "MongoDB scales horizontally (sharding); SQL scales vertically",
                "data_model":  "MongoDB stores JSON-like BSON documents; SQL stores rows",
                "joins":       "MongoDB uses embedded docs / $lookup; SQL uses JOIN",
                "transactions": "MongoDB supports multi-doc ACID transactions since v4.0"
            }
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/scaling-info", methods=["GET"])
def scaling_info():
    """
    Syllabus topics explained via API (conceptual endpoint):
      ✔ Replication — Master/Slave, Peer-to-Peer, Replica Sets
      ✔ Sharding — horizontal scaling across nodes
      ✔ Database scaling models
    MongoDB Atlas handles this automatically for cloud deployments.
    """
    return jsonify({
        "replication": {
            "what_it_is":       "Copying data across multiple MongoDB servers for availability",
            "replica_set":      "A group of mongod instances that maintain the same dataset",
            "primary":          "Receives all write operations",
            "secondary":        "Replicates data from primary; can serve reads",
            "automatic_failover": "If primary fails, a secondary is elected automatically",
            "master_slave":     "Older model — one master handles writes, slaves replicate",
            "peer_to_peer":     "All nodes are equal; any can become primary (replica sets)"
        },
        "sharding": {
            "what_it_is":       "Distributing data across multiple machines (horizontal scaling)",
            "shard":            "Each shard holds a subset of the data",
            "shard_key":        "The field used to distribute documents across shards",
            "mongos":           "Router — directs queries to the correct shard",
            "config_server":    "Stores metadata and cluster configuration",
            "why_sharding":     "When a single machine cannot handle the data volume or throughput",
            "lookup_strategy":  "Range-based or hash-based partitioning of the shard key"
        },
        "this_project": {
            "atlas_replication": "MongoDB Atlas M0 free tier uses a 3-node replica set automatically",
            "scaling_note":      "For production: upgrade to M10+ and enable sharding via Atlas UI"
        }
    })


@app.route("/contact", methods=["POST"])
def contact():
    try:
        data    = request.get_json()
        name    = data.get("name","").strip()
        email   = data.get("email","").strip()
        message = data.get("message","").strip()

        if not name or not email or not message:
            return jsonify({"error": "name, email and message are required"}), 400

        messages_col.insert_one({
            "name":      name,
            "email":     email,
            "message":   message,
            "timestamp": datetime.utcnow()
        })
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

        users_col.insert_one({
            "name":       name,
            "email":      email,
            "password":   password,
            "created_at": datetime.utcnow()
        })
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

        return jsonify({
            "message": "Login successful",
            "user": {
                "id":    str(user["_id"]),
                "name":  user["name"],
                "email": user["email"]
            }
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)
