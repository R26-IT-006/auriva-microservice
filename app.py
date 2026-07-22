from flask import Flask, request, jsonify
from phoneme_scorer import score_phoneme

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
    app.run(port=5001, debug=True)