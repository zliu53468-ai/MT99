from flask import Flask, request, jsonify
import lightgbm as lgb
import xgboost as xgb
import numpy as np
import joblib

app = Flask(__name__)

# 載入模型
lgb_model = joblib.load("lightgbm_model.pkl")
xgb_model = joblib.load("xgboost_model.pkl")

# 百家樂俚語特徵判斷
def roadmap_to_features(roadmap):
    features = {}
    last4 = roadmap[-4:] if len(roadmap) >= 4 else []

    # 三帶一判斷
    features["is_3_1"] = int(len(last4) == 4 and last4[0] == last4[1] == last4[2] and last4[3] != last4[0])

    # 長龍判斷
    streak = 1
    for i in range(len(roadmap)-2, -1, -1):
        if roadmap[i] == roadmap[i+1]:
            streak += 1
        else:
            break
    features["is_long_dragon"] = int(streak >= 5)

    # 齊腳判斷（莊閒交錯）
    features["is_qijiao"] = int(all(roadmap[i] != roadmap[i+1] for i in range(len(roadmap)-1))) if len(roadmap) >= 4 else 0

    # 破路偵測（前排規律被打斷）
    features["is_break_pattern"] = int(len(roadmap) >= 4 and roadmap[-1] != roadmap[-3])

    # 延續長龍
    features["continue_dragon"] = int(features["is_long_dragon"] and roadmap[-1] == roadmap[-2]) if len(roadmap) >= 2 else 0

    # 前排對照（與前4局一致）
    features["match_previous_pattern"] = int(roadmap[-4:] == roadmap[-8:-4]) if len(roadmap) >= 8 else 0

    return np.array([list(features.values())]), features

# 預測路由
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    roadmap = data.get("roadmap", [])
    next_input = data.get("next_input", "")  # 可選擇性使用

    if not roadmap or len(roadmap) < 4:
        return jsonify({"error": "請提供至少4局牌路"}), 400

    X, feature_flags = roadmap_to_features(roadmap)

    # 模型預測
    lgb_pred = lgb_model.predict_proba(X)[0]
    xgb_pred = xgb_model.predict_proba(X)[0]

    # Soft voting 融合
    final_pred = (np.array(lgb_pred) + np.array(xgb_pred)) / 2

    # 建議下注
    labels = ["banker", "player", "tie"]
    suggestion = labels[np.argmax(final_pred)]

    return jsonify({
        "banker": round(final_pred[0], 4),
        "player": round(final_pred[1], 4),
        "tie": round(final_pred[2], 4),
        "pattern": detect_pattern_name(feature_flags),
        "match_previous": bool(feature_flags["match_previous_pattern"]),
        "suggestion": f"建議下注：{suggestion}",
        "confidence": round(np.max(final_pred), 4)
    })

# 俚語型態命名
def detect_pattern_name(flags):
    if flags["is_3_1"]:
        return "三帶一"
    elif flags["is_long_dragon"]:
        return "長龍"
    elif flags["is_qijiao"]:
        return "齊腳"
    elif flags["is_break_pattern"]:
        return "破路"
    elif flags["continue_dragon"]:
        return "延續長龍"
    elif flags["match_previous_pattern"]:
        return "前排對照"
    else:
        return "無特定型態"

if __name__ == "__main__":
    app.run(debug=True)
