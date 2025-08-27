# pip install flask flask-cors scikit-learn joblib

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import joblib
import numpy as np
from sklearn.linear_model import SGDClassifier
from datetime import datetime

app = Flask(__name__)
CORS(app)

MODEL_PATH = "model.pkl"
BACKUP_DIR = "model_backups"

# === 初始化或載入模型 ===
def init_model():
    clf = SGDClassifier(loss="log_loss")
    # 第一次 partial_fit 需要定義類別標籤
    clf.partial_fit(np.zeros((1, 8)), [0], classes=[0, 1])
    return clf

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = init_model()

# 建立備份資料夾
if not os.path.exists(BACKUP_DIR):
    os.makedirs(BACKUP_DIR)

# === 特徵工程：將 roadmap 轉成數值向量 ===
def extract_features(roadmap):
    clean_road = [x for x in roadmap if x in ["莊", "閒"]]
    if not clean_road:
        return np.zeros(8)

    banker_ratio = clean_road.count("莊") / len(clean_road)
    player_ratio = clean_road.count("閒") / len(clean_road)

    # 連勝長度
    streak_len = 1
    for i in range(len(clean_road) - 1, 0, -1):
        if clean_road[i] == clean_road[i - 1]:
            streak_len += 1
        else:
            break

    # 最近 5 局編碼 (莊=1, 閒=0, 不足補 -1)
    recent = [(1 if r == "莊" else 0) for r in clean_road[-5:]]
    if len(recent) < 5:
        recent = [-1] * (5 - len(recent)) + recent

    return np.array([banker_ratio, player_ratio, streak_len] + recent)

# === 預測與即時學習 ===
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    roadmap = data.get("roadmap", [])
    label = data.get("label")  # 可選，0=莊贏, 1=閒贏

    features = extract_features(roadmap).reshape(1, -1)

    # 推論
    probs = model.predict_proba(features)[0]
    predictions = {
        "banker": float(probs[0]),
        "player": float(probs[1]),
        "tie": 0.05
    }

    # 增量學習 + 保存 + 備份
    if label is not None and label in [0, 1]:
        model.partial_fit(features, [label])
        joblib.dump(model, MODEL_PATH)

        # 備份檔案（帶日期時間）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(BACKUP_DIR, f"model_{timestamp}.pkl")
        joblib.dump(model, backup_path)

    return jsonify(predictions)

# === 重置模型（清零重學，可選用） ===
@app.route("/reset_model", methods=["POST"])
def reset_model():
    global model
    model = init_model()
    joblib.dump(model, MODEL_PATH)
    return jsonify({"status": "success", "message": "模型已重置，從零開始學習"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
