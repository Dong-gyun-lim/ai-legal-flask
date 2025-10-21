from flask import Flask, jsonify, request

app = Flask(__name__)

@app.get("/health")
def health():
    return jsonify({"status": "healthy"})

@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    return jsonify({
        "ok": True,
        "received": data,
        "analysis": {"similarity_score": 0.82, "notes": "demo"}
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
