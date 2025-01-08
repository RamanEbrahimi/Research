from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "No file part"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"status": "error", "message": "No selected file"}), 400
    
    # Save the uploaded file
    file.save(f"./{file.filename}")
    return jsonify({"status": "success", "message": f"File '{file.filename}' uploaded successfully"}), 200

if __name__ == "__main__":
    app.run(debug=True)
