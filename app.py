# coding=utf-8
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.exceptions import NotFittedError

# 設定應用程式
app = Flask(__name__)
# 允許跨域請求
CORS(app)

# 模型檔案路徑
MODEL_PATH = "model.pkl"
# 預期的特徵數，請根據你的模型來設定
EXPECTED_FEATURE_LEN = 5

# ===== 模型載入 / 初始化 =====
# 檢查模型檔案是否存在
if os.path.exists(MODEL_PATH):
    try:
        # 如果存在，載入已儲存的模型
        model = joblib.load(MODEL_PATH)
        print("模型已成功載入。")
    except Exception as e:
        # 載入失敗時，重新初始化模型
        print(f"載入模型失敗，正在重新初始化。錯誤: {e}")
        model = SGDClassifier(loss="log_loss")
        # 用假資料進行初始化訓練，確保模型可以正確預測
        X_init = np.array([[0.0] * EXPECTED_FEATURE_LEN])
        y_init = np.array([0])
        model.partial_fit(X_init, y_init, classes=[0, 1])
        # 儲存初始化的模型
        joblib.dump(model, MODEL_PATH)
        print("模型已重新初始化並儲存。")
else:
    # 如果不存在，創建新的模型
    print("模型檔案不存在，正在創建新模型並初始化。")
    model = SGDClassifier(loss="log_loss")
    # 用假資料進行初始化訓練，確保模型可以正確預測
    X_init = np.array([[0.0] * EXPECTED_FEATURE_LEN])
    y_init = np.array([0])
    # 重要：partial_fit 需要 classes=[0, 1] 來初始化所有可能的標籤
    model.partial_fit(X_init, y_init, classes=[0, 1])
    # 儲存初始化的模型
    joblib.dump(model, MODEL_PATH)
    print("新模型已創建並儲存。")

# ===== 增量更新函數 =====
def update_model_with_new_data(features, label):
    """
    使用新資料增量更新模型。
    :param features: 包含單一樣本特徵的列表。
    :param label: 單一樣本的標籤 (0 或 1)。
    :return: 更新後的模型。
    """
    X_new = np.array([features])
    y_new = np.array([label])
    model.partial_fit(X_new, y_new)
    joblib.dump(model, MODEL_PATH)
    return model

# ===== roadmap -> features 轉換 (請替換成你的邏輯) =====
def roadmap_to_features(roadmap):
    """
    將百家樂路圖轉換成模型可以使用的特徵向量。
    注意：這裡僅為範例，實際應用需要根據你的策略來實現。
    """
    # 這裡的邏輯需要你自己根據百家樂的規則來設計
    # 例如：
    # - 莊家和閒家連續出現的次數
    # - 大路、大眼仔、小路等路圖的模式
    # - 莊閒總數、和局總數等
    # 假設你從 roadmap 中提取了 5 個數值特徵
    return [0.4, 0.6, 3, 1, 0]

# ===== API 路由 =====
@app.route("/predict", methods=["POST"])
def predict():
    """
    接收資料進行預測或訓練。
    """
    data = request.get_json(force=True)
    
    # 為了幫助偵錯，印出接收到的資料
    print(f"接收到的資料: {data}")

    # 支援 features 與 roadmap 兩種輸入
    features = data.get("features")
    if not features and "roadmap" in data:
        features = roadmap_to_features(data["roadmap"])

    # 資料驗證
    if not isinstance(features, list) or len(features) != EXPECTED_FEATURE_LEN:
        return jsonify({"error": f"無效的特徵值長度，預期為 {EXPECTED_FEATURE_LEN}"}), 400

    label = data.get("label")

    # 如果有 label 就進行增量學習
    global model
    if label is not None:
        try:
            # 訓練模型需要有莊家(0)和閒家(1)兩種類型的資料，模型才能學習其差異
            print(f"正在使用標籤 {label} 訓練模型...")
            model = update_model_with_new_data(features, label)
        except ValueError as e:
            # 處理標籤不正確的錯誤
            return jsonify({"error": f"訓練失敗: {e}"}), 400

    # 預測機率
    try:
        # 使用 numpy 陣列來進行預測
        input_features = np.array([features])
        probs = model.predict_proba(input_features)[0]

        # 由於你的模型只處理莊家(0)和閒家(1)，這裡的機率會是 [prob_banker, prob_player]
        banker_prob = probs[0]
        player_prob = probs[1]
    except NotFittedError:
        # 如果模型未經過任何訓練，會拋出此錯誤
        # 這裡返回一個預設值，避免程式崩潰
        return jsonify({"error": "模型尚未訓練，無法預測。請先進行訓練。"}), 500

    # 回傳三欄位 JSON (和局機率暫用固定值，因為模型未處理)
    return jsonify({
        "banker": float(banker_prob),
        "player": float(player_prob),
        "tie": 0.05  # 和局的機率是固定的，因為模型沒有這方面的訓練
    })

@app.route("/", methods=["GET"])
def health():
    """
    健康檢查路由。
    """
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    # 從環境變數中取得 Port，如果沒有則使用 5000
    port = int(os.environ.get("PORT", 5000))
    # 啟動 Flask 應用程式
    app.run(host="0.0.0.0", port=port)
