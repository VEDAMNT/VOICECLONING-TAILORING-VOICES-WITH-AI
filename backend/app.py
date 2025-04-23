import os
from flask import Flask, request, send_file
from voice_cloner import generate_audio

app = Flask(__name__)

# Ensure the sampled folder exists
SAMPLED_FOLDER = "sampled"
os.makedirs(SAMPLED_FOLDER, exist_ok=True)

# Keep-alive endpoint
@app.route('/ping', methods=['GET'])
def ping():
    return "OK", 200

@app.route('/clone', methods=['POST'])
def clone():
    try:
        text = request.form.get('text', 'Hello, this is a test.')
        language = request.form.get('language', 'en')
        speed = float(request.form.get('speed', 1.0))
        pitch = float(request.form.get('pitch', 0.0))
        voiceProfile = request.form.get('voiceProfile', 'default')

        audio_files = request.files.getlist('audio_files')
        audio_paths = []
        for audio in audio_files:
            audio_path = os.path.join(SAMPLED_FOLDER, audio.filename)
            audio.save(audio_path)
            audio_paths.append(audio_path)

        output_path = os.path.join(SAMPLED_FOLDER, "output.wav")
        progress_generator = generate_audio(
            text, audio_paths, output_path, language=language,
            speed=speed, pitch=pitch, voiceProfile=voiceProfile
        )

        for progress in progress_generator:
            if isinstance(progress, int):
                print(f"Progress: {progress}%")
            else:
                output_path = progress

        return send_file(output_path, as_attachment=True, download_name="cloned_voice.wav")

    except Exception as e:
        return f"Error: {str(e)}", 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)