from flask import Flask, request, jsonify
from collections import deque
from sklearn.linear_model import SGDClassifier
import joblib
import os

app = Flask(__name__)

# === 模型與近期資料緩存 ===
MODEL_PATH = "model.pkl"
recent_data = deque(maxlen=32)  # 近期視窗長度，可依需求調整

# 載入模型或建立新模型
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = SGDClassifier()
    # 初始化類別，避免第一次 partial_fit 出錯
    model.partial_fit([[0]*5], [0], classes=[0, 1])  # 假設 5 維特徵、0/1 兩類

# === 更新模型（使用近期視窗） ===
def update_model_with_new_data(features, label):
    recent_data.append((features, label))
    X_batch, y_batch = zip(*recent_data)
    model.partial_fit(X_batch, y_batch, classes=[0, 1])
    joblib.dump(model, MODEL_PATH)  # 儲存模型
    return model

# === /predict 保留原接口 ===
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = data["features"]
    label = data.get("label")

    global model
    if label is not None:
        model = update_model_with_new_data(features, label)
    
    prediction = model.predict([features])[0]
    return jsonify({"prediction": int(prediction)})

# === /train 也保留原接口（如果有） ===
@app.route("/train", methods=["POST"])
def train():
    data = request.get_json()
    features_list = data["features"]
    labels_list = data["labels"]

    global model
    for features, label in zip(features_list, labels_list):
        model = update_model_with_new_data(features, label)

    return jsonify({"status": "training complete"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
