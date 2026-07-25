import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from phoneme_scorer import score_phoneme

load_dotenv()

app = Flask(__name__)

@app.route('/score-phoneme', methods=['POST'])
def score():
    data = request.json
    result = score_phoneme(
        data.get('transcript', ''),
        data.get('target_word', '')
    )
    return jsonify(result)

if __name__ == '__main__':
    # TASK-26 — debug mode (Werkzeug's interactive debugger + auto-reload)
    # must not be the default for a pilot-facing run; opt in explicitly.
    debug_mode = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(port=5001, debug=debug_mode)