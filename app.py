import os
from flask import Flask, request, send_file, render_template, Response
from voice_cloner import generate_audio
from pathlib import Path
import uuid

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/clone", methods=["POST"])
def clone_voice():
    if "audio" not in request.files:
        return "No audio files uploaded", 400
    
    audio_files = request.files.getlist("audio")
    if not audio_files:
        return "No audio files selected", 400
    
    unique_id = str(uuid.uuid4())
    audio_paths = []
    for audio_file in audio_files:
        if audio_file.filename:
            audio_path = os.path.join(UPLOAD_FOLDER, f"user_audio_{unique_id}_{len(audio_paths)}.wav")
            audio_file.save(audio_path)
            audio_paths.append(audio_path)
    
    if not audio_paths:
        return "No valid audio files uploaded", 400
    
    text = request.form.get("text", "Hello, this is my cloned voice!")
    language = request.form.get("language", "en")
    try:
        output_path = generate_audio(text, audio_paths, f"output_{unique_id}.wav", language=language)
        
        def generate():
            with open(output_path, "rb") as f:
                yield from f
            for path in audio_paths:
                try:
                    os.remove(path)
                except OSError as e:
                    print(f"Failed to remove {path}: {e}")
            try:
                os.remove(output_path)
            except OSError as e:
                print(f"Failed to remove {output_path}: {e}")
        
        return Response(generate(), mimetype="audio/wav")
    except Exception as e:
        print(f"Clone error: {e}")
        return f"Error processing request: {str(e)}", 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use Render's assigned port
    app.run(host="0.0.0.0", port=port, debug=False)  # debug=False for production