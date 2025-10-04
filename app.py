from flask import Flask, request, jsonify
from flask_cors import CORS
from deepface import DeepFace
import base64
import os
import tempfile

app = Flask(__name__)
CORS(app)

def calculate_contextual_confidence(emotions):
    emotion_entries = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
    dominant_emotion = emotion_entries[0][0] if emotion_entries else 'neutral'
    
    happy_score = emotions.get('happy', 0)
    surprise_score = emotions.get('surprise', 0)
    neutral_score = emotions.get('neutral', 0)
    
    negative_emotions = ['sad', 'fear', 'angry', 'disgust']
    negative_score = sum(emotions.get(emotion, 0) for emotion in negative_emotions)
    
    final_score = happy_score - negative_score - neutral_score
    surprise_contribution = 0
    surprise_reason = 'Nötr Etki'
    
    if dominant_emotion == 'happy':
        surprise_contribution = surprise_score
        surprise_reason = 'Pozitif (Baskın Mutluluk)'
    elif dominant_emotion in negative_emotions:
        surprise_contribution = -surprise_score
        surprise_reason = 'Negatif (Baskın Olumsuz)'
    elif dominant_emotion == 'surprise':
        if happy_score > negative_score:
            surprise_contribution = surprise_score
            surprise_reason = 'Pozitif (Pozitifler Güçlü)'
        else:
            surprise_contribution = -surprise_score
            surprise_reason = 'Negatif (Olumsuzlar Güçlü)'
    
    final_score += surprise_contribution
    normalized_score = max(0, min(100, (final_score + 100) / 2))
    
    return {
        'score': normalized_score,
        'happy_total': happy_score,
        'negative_total': negative_score,
        'neutral_impact': neutral_score,
        'surprise_contribution': surprise_contribution,
        'surprise_reason': surprise_reason,
        'base_calculation': final_score
    }

@app.route('/analyze', methods=['POST'])
def analyze_emotion():
    try:
        # API key kontrolü (opsiyonel güvenlik)
        api_key = request.headers.get('Authorization', '').replace('Bearer ', '')
        expected_key = os.environ.get('API_KEY', '')
        
        if expected_key and api_key != expected_key:
            return jsonify({'error': 'Unauthorized'}), 401
        
        data = request.json
        image_data = data.get('image', '')
        
        if not image_data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Base64'ü decode et
        image_data = image_data.split(',')[1] if ',' in image_data else image_data
        image_bytes = base64.b64decode(image_data)
        
        # Geçici dosyaya yaz
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(image_bytes)
            tmp_path = tmp_file.name
        
        try:
            # DeepFace analizi
            result = DeepFace.analyze(
                img_path=tmp_path,
                actions=['emotion'],
                enforce_detection=False
            )
            
            emotions = result[0]['emotion']
            total = sum(float(v) for v in emotions.values())
            emotions_percent = {k: round(float(v) / total * 100, 1) for k, v in emotions.items()}
            
            confidence_result = calculate_contextual_confidence(emotions_percent)
            
            return jsonify({
                'confidence_score': round(confidence_result['score'], 1),
                'details': confidence_result,
                'emotions': emotions_percent
            })
        
        finally:
            # Geçici dosyayı sil
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'MetaMind Emotion API', 'status': 'running'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
