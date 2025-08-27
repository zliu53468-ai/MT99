# pip install flask flask-cors scikit-learn joblib

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import joblib
import numpy as np
from sklearn.linear_model import SGDClassifier
from datetime import datetime
from collections import deque

app = Flask(__name__)
CORS(app)

MODEL_PATH = "model.pkl"
BACKUP_DIR = "model_backups"

# 短期視窗：更適合單鞋 50–70 手的節奏
HISTORY_LIMIT = 64
history_features = deque(maxlen=HISTORY_LIMIT)
history_labels = deque(maxlen=HISTORY_LIMIT)

# 用於偵測鞋重置
prev_road_len = 0
RESET_DROP_THRESHOLD = 8  # roadmap 長度相比上一筆大幅回落視為換鞋（可調）

def init_model():
    # 平均化參數提升穩定、log_loss 保留概率輸出
    clf = SGDClassifier(loss="log_loss", average=True)
    clf.partial_fit(np.zeros((1, 8)), [0], classes=[0, 1])
    return clf

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = init_model()

if not os.path.exists(BACKUP_DIR):
    os.makedirs(BACKUP_DIR)

def extract_features(roadmap):
    clean_road = [x for x in roadmap if x in ["莊", "閒"]]
    if not clean_road:
        return np.zeros(8)

    banker_ratio = clean_road.count("莊") / len(clean_road)
    player_ratio = clean_road.count("閒") / len(clean_road)

    streak_len = 1
    for i in range(len(clean_road) - 1, 0, -1):
        if clean_road[i] == clean_road[i - 1]:
            streak_len += 1
        else:
            break

    recent = [(1 if r == "莊" else 0) for r in clean_road[-5:]]
    if len(recent) < 5:
        recent = [-1] * (5 - len(recent)) + recent

    return np.array([banker_ratio, player_ratio, streak_len] + recent)

def count_hands(roadmap):
    return sum(1 for x in roadmap if x in ["莊", "閒"])

@app.route("/predict", methods=["POST"])
def predict():
    global model, prev_road_len
    data = request.get_json()
    roadmap = data.get("roadmap", [])
    label = data.get("label")

    features = extract_features(roadmap).reshape(1, -1)

    # 推論
    probs = model.predict_proba(features)[0]
    banker_prob, player_prob = float(probs[0]), float(probs[1])

    # 輕量糾偏：接近時稍微向近期較強一方傾斜
    diff = abs(banker_prob - player_prob)
    if diff < 0.05:
        banker_ratio, player_ratio = features[0][0], features[0][1]
        shift = 0.03
        if banker_ratio > player_ratio:
            banker_prob = min(1.0, banker_prob + shift)
            player_prob = max(0.0, player_prob - shift)
        else:
            banker_prob = max(0.0, banker_prob - shift)
            player_prob = min(1.0, player_prob + shift)

    predictions = {
        "banker": round(banker_prob, 4),
        "player": round(player_prob, 4),
        "tie": 0.05
    }

    # --- 即時學習 ---
    if label is not None and label in [0, 1]:
        road_len = count_hands(roadmap)

        # 換鞋偵測：roadmap 長度相對前一筆顯著回落
        if road_len + RESET_DROP_THRESHOLD < prev_road_len:
            history_features.clear()
            history_labels.clear()

        prev_road_len = road_len

        # 累積近期資料
        history_features.append(features.flatten())
        history_labels.append(label)

        # 動態有效視窗：跟上當前鞋的步調（至少 16 筆，最多 HISTORY_LIMIT）
        X = np.array(history_features)
        y = np.array(history_labels)
        effective_n = min(len(y), max(16, int(1.5 * max(road_len, 1))))
        X_eff, y_eff = X[-effective_n:], y[-effective_n:]

        # 指數遞減的近期加權（越新權重越大）
        w = np.exp(np.linspace(-1.5, 0, num=len(y_eff)))

        model.partial_fit(X_eff, y_eff, sample_weight=w)
        joblib.dump(model, MODEL_PATH)

        # 備份
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(BACKUP_DIR, f"model_{timestamp}.pkl")
        joblib.dump(model, backup_path)

    return jsonify(predictions)

@app.route("/reset_model", methods=["POST"])
def reset_model():
    global model, history_features, history_labels, prev_road_len
    model = init_model()
    history_features.clear()
    history_labels.clear()
    prev_road_len = 0
    joblib.dump(model, MODEL_PATH)
    return jsonify({"status": "success", "message": "模型已重置，從零開始學習"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
