from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os

app = Flask(__name__)
CORS(app)

MODEL_PATH = "model.pkl"
EXPECTED_FEATURE_LEN = 5  # 修改成你的模型特徵數

# ===== 模型載入 / 初始化 =====
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    from sklearn.linear_model import SGDClassifier
    import numpy as np
    model = SGDClassifier(loss="log_loss")
    # 用假資料初始化，避免首次 predict_proba 出錯
    X_init = np.zeros((1, EXPECTED_FEATURE_LEN))
    y_init = [0]
    model.partial_fit(X_init, y_init, classes=[0, 1])
    joblib.dump(model, MODEL_PATH)

# ===== 增量更新函數 =====
def update_model_with_new_data(features, label):
    import numpy as np
    X_new = np.array([features])
    y_new = [label]
    model.partial_fit(X_new, y_new)
    joblib.dump(model, MODEL_PATH)
    return model

# ===== roadmap -> features 轉換（請替換成你的邏輯） =====
def roadmap_to_features(roadmap):
    # TODO: 根據你的 roadmap 規則轉換成特徵向量
    return [0.4, 0.6, 3, 1, 0]  # 範例假資料

# ===== API 路由 =====
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    # 支援 features 與 roadmap 兩種輸入
    features = data.get("features")
    if not features and "roadmap" in data:
        features = roadmap_to_features(data["roadmap"])

    # 資料驗證
    if not isinstance(features, list) or len(features) != EXPECTED_FEATURE_LEN:
        return jsonify({"error": f"invalid features length, expected {EXPECTED_FEATURE_LEN}"}), 400

    label = data.get("label")

    # 有 label 就進行增量學習
    global model
    if label is not None:
        model = update_model_with_new_data(features, label)

    # 預測機率
    probs = model.predict_proba([features])[0]

    # 回傳三欄位 JSON（tie 暫用固定值）
    return jsonify({
        "banker": float(probs[0]),
        "player": float(probs[1]),
        "tie": 0.05
    })

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
