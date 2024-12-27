from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# Directory to store interaction data (as a file)
DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)

DATA_FILE = os.path.join(DATA_DIR, "interactions.json")

# Endpoint to store interactions
@app.route('/api/store', methods=['POST'])
def store_interaction():
    data = request.json
    if not data:
        return jsonify({"status": "error", "message": "No data received"}), 400

    # Append data to the interactions file
    interactions = []
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            try:
                interactions = json.load(f)
            except ValueError:
                # File exists but is empty
                interactions = []

    interactions.append(data)
    with open(DATA_FILE, "w") as f:
        json.dump(interactions, f, indent=2)

    print(f"Received and stored interaction: {data}")
    return jsonify({"status": "success", "message": "Data stored successfully"}), 200

# Endpoint to retrieve stored interactions (for your use)
@app.route('/api/interactions', methods=['GET'])
def get_interactions():
    if not os.path.exists(DATA_FILE):
        return jsonify([])  # Return empty list if no data exists

    with open(DATA_FILE, "r") as f:
        interactions = json.load(f)

    return jsonify(interactions), 200

if __name__ == '__main__':
    app.run(debug=True)
