from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)  # ✅ 允許跨網域請求，支援 CodePen 等前端呼叫

# 模擬模型預測（實際部署時請載入訓練好的模型）
def mock_predict(features):
    xgb_pred = np.array([0.6, 0.3, 0.1])  # 莊 / 閒 / 和
    lgb_pred = np.array([0.5, 0.4, 0.1])
    final_pred = (xgb_pred + lgb_pred) / 2
    return {
        "banker": round(final_pred[0], 2),
        "player": round(final_pred[1], 2),
        "tie": round(final_pred[2], 2)
    }

# 特徵提取邏輯：分析 roadmap 中的連莊、連閒、跳牌、長牌等
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

# API 路由：接收 roadmap，回傳預測結果
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    roadmap = data.get("roadmap", [])
    features = extract_features(roadmap)
    prediction = mock_predict(features)
    return jsonify(prediction)

# 啟動 Flask 伺服器（部署時會由 gunicorn 啟動）
if __name__ == "__main__":
    app.run(debug=True)
