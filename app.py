from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)  # ✅ 加入 CORS 支援，允許跨網域請求

# 模擬模型預測（實際部署時請載入訓練好的模型）
def mock_predict(features):
    xgb_pred = np.array([0.6, 0.3, 0.1])  # 莊/閒/和
    lgb_pred = np.array([0.5, 0.4, 0.1])
    final_pred = (xgb_pred + lgb_pred) / 2
    return {
        "banker": round(final_pred[0], 2),
        "player": round(final_pred[1], 2),
        "tie": round(final_pred[2], 2)
    }

def extract_features(roadmap):
    features = {
        "banker_streak": 0,
        "player_streak": 0,
        "jump_count": 0,
        "long_banker": 0,
        "long_player": 0
    }
    streak = 0
    last = None
    for item in roadmap:
        if item in ["莊", "閒"]:
            if item == last:
                streak += 1
            else:
                streak = 1
            if item == "莊":
                features["banker_streak"] = max(features["banker_streak"], streak)
            else:
                features["player_streak"] = max(features["player_streak"], streak)
            last = item
        elif item == "跳":
            features["jump_count"] += 1
        elif item == "長莊":
            features["long_banker"] += 1
        elif item == "長閒":
            features["long_player"] += 1
    return np.array(list(features.values())).reshape(1, -1)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    roadmap = data.get("roadmap", [])
    features = extract_features(roadmap)
    prediction = mock_predict(features)
    return jsonify(prediction)

if __name__ == "__main__":
    app.run(debug=True)
