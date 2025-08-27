# pip install flask flask-cors scikit-learn joblib

from flask import Flask, request, jsonify
from flask_cors import CORS
from collections import deque
from sklearn.linear_model import SGDClassifier
import numpy as np
import joblib
import os

app = Flask(__name__)
CORS(app)

# === 可調參數（針對第 10~20 手入場優化） ===
MODEL_PATH = "model.pkl"
RECENT_MAX = 32              # 近期滑動視窗上限
MIN_STABLE_SAMPLES = 12      # 在樣本 < 12 前，強化保守「收斂」處理
MOMENTUM_WINDOW = 8          # 近期動量視窗（放大最近 8 手的影響）
RECENCY_DECAY = 1.2          # 近新加權的衰減幅度（越大=越看重最新）
SHRINK_MAX = 0.35            # 早期最大收斂強度（往 0.5/0.5 拉的最大比例）
NUDGE = 0.04                 # 當兩邊機率接近時，依近期動量微調的幅度
CLOSE_DIFF = 0.08            # 視為「接近」的門檻（小於此就啟動動量微調）
EXPECTED_FEATURE_LEN = 5     # 請改成你的實際特徵長度（務必和前端一致）

# === 近期資料快取（僅存 label 用來估動量 & 平衡；不改接口） ===
recent_batch = deque(maxlen=RECENT_MAX)  # 存 (features, label)
recent_labels = deque(maxlen=RECENT_MAX) # 只存 label，便於動量與平衡

# === 載入或初始化模型 ===
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = SGDClassifier(loss="log_loss", average=True)
    # 第一次 partial_fit 需定義類別
    model.partial_fit([ [0]*EXPECTED_FEATURE_LEN ], [0], classes=[0, 1])

# === 用近期視窗做增量學習（避免「整條路單重複學」） ===
def update_model_with_new_data(features, label):
    # 緩存
    recent_batch.append((features, label))
    recent_labels.append(label)

    # 準備批次
    X_batch, y_batch = zip(*recent_batch)
    X_batch = np.array(X_batch, dtype=float)
    y_batch = np.array(y_batch, dtype=int)

    # 類別平衡權重（避免早期單邊鎖死）
    unique, counts = np.unique(y_batch, return_counts=True)
    freq = dict(zip(unique.tolist(), counts.tolist()))
    class_w = {c: 1.0 / max(freq.get(c, 1), 1) for c in [0, 1]}
    w_class = np.array([class_w[y] for y in y_batch])

    # 近期指數遞減加權（越新越重）
    w_recency = np.exp(np.linspace(-RECENCY_DECAY, 0, num=len(y_batch)))

    # 合併權重並歸一化到平均 1 附近，避免數值過大/過小
    w = w_class * w_recency
    w = w * (len(w) / (w.sum() + 1e-8))

    model.partial_fit(X_batch, y_batch, classes=[0, 1], sample_weight=w)
    joblib.dump(model, MODEL_PATH)
    return model

# === 機率後處理：樣本少時收斂、接近時依動量微調、保持和局 5% 固定 ===
def postprocess_probs(raw_probs):
    # raw_probs: [p_banker, p_player]
    p_b, p_p = float(raw_probs[0]), float(raw_probs[1])

    k = len(recent_labels)

    # 1) 樣本不足 → 往中性收斂，避免第 20 手入場時過度自信
    if k < MIN_STABLE_SAMPLES:
        # gamma 隨樣本數線性遞減，最高 SHRINK_MAX
        gamma = min(SHRINK_MAX, (MIN_STABLE_SAMPLES - k) / max(MIN_STABLE_SAMPLES, 1))
        p_b = (1 - gamma) * p_b + gamma * 0.5
        p_p = (1 - gamma) * p_p + gamma * 0.5

    # 2) 若兩邊機率很接近 → 依近期動量（最後 MOMENTUM_WINDOW 手）微調
    diff = abs(p_b - p_p)
    if diff < CLOSE_DIFF and k >= 2:
        m = min(MOMENTUM_WINDOW, k)
        last_m = list(recent_labels)[-m:]  # 0=莊, 1=閒
        # 近期莊比例（label=0）
        banker_recent = 1.0 - (sum(last_m) / len(last_m))
        if banker_recent > 0.5:
            p_b = min(1.0, p_b + NUDGE)
            p_p = max(0.0, p_p - NUDGE)
        elif banker_recent < 0.5:
            p_b = max(0.0, p_b - NUDGE)
            p_p = min(1.0, p_p + NUDGE)
        # 平手時不微調

    # 3) 保持 banker+player 的總和為 0.95（和局固定 0.05）
    s = max(p_b + p_p, 1e-8)
    p_b = 0.95 * p_b / s
    p_p = 0.95 * p_p / s

    return round(p_b, 4), round(p_p, 4), 0.05

# === /predict（接口不變，回傳 banker/player/tie 機率） ===
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = data.get("features")
    label = data.get("label")  # 可為 None，0=莊,1=閒

    # 防呆
    if not isinstance(features, list) or len(features) != EXPECTED_FEATURE_LEN:
        return jsonify({"error": "invalid features"}), 400
    if label is not None and label not in [0, 1]:
        return jsonify({"error": "invalid label"}), 400

    # 先學再測（你也可改：先測再學）
    global model
    if label is not None:
        model = update_model_with_new_data(features, label)

    # 推論 + 後處理
    probs = model.predict_proba([features])[0]
    pb, pp, pt = postprocess_probs(probs)
    return jsonify({"banker": pb, "player": pp, "tie": pt})

# === /train（批量學習；接口不變） ===
@app.route("/train", methods=["POST"])
def train():
    data = request.get_json()
    features_list = data.get("features")
    labels_list = data.get("labels")

    if not features_list or not labels_list or len(features_list) != len(labels_list):
        return jsonify({"error": "invalid training data"}), 400

    global model
    for x, y in zip(features_list, labels_list):
        if isinstance(x, list) and len(x) == EXPECTED_FEATURE_LEN and y in [0, 1]:
            model = update_model_with_new_data(x, y)

    return jsonify({"status": "training complete"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
