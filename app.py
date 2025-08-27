from flask import Flask, request, jsonify
from flask_cors import CORS
import lightgbm as lgb
import xgboost as xgb
import numpy as np
import joblib
import os

app = Flask(__name__)
CORS(app)

# 模型加载状态
model_loaded = False

# 載入模型
try:
    # 检查模型文件是否存在
    if os.path.exists("lightgbm_model.pkl") and os.path.exists("xgboost_model.pkl"):
        lgb_model = joblib.load("lightgbm_model.pkl")
        xgb_model = joblib.load("xgboost_model.pkl")
        model_loaded = True
        print("模型載入成功")
    else:
        print("模型文件不存在")
        model_loaded = False
        # 創建假模型用於測試
        class DummyModel:
            def predict_proba(self, X):
                return np.array([[0.45, 0.44, 0.11]])  # 模擬預測概率
        lgb_model = DummyModel()
        xgb_model = DummyModel()
except Exception as e:
    print(f"模型載入失敗: {e}")
    model_loaded = False
    # 創建假模型用於測試
    class DummyModel:
        def predict_proba(self, X):
            return np.array([[0.45, 0.44, 0.11]])  # 模擬預測概率
    lgb_model = DummyModel()
    xgb_model = DummyModel()

# 增強版特徵提取
def enhanced_roadmap_to_features(roadmap):
    features = {}
    
    if len(roadmap) < 4:
        return None, features
    
    # 基本統計特徵
    banker_count = roadmap.count('banker')
    player_count = roadmap.count('player')
    tie_count = roadmap.count('tie')
    total = len(roadmap)
    
    features['banker_ratio'] = banker_count / total if total > 0 else 0
    features['player_ratio'] = player_count / total if total > 0 else 0
    features['tie_ratio'] = tie_count / total if total > 0 else 0
    
    # 近期趨勢（最後5局）
    last_5 = roadmap[-5:] if len(roadmap) >= 5 else roadmap
    features['last_5_banker'] = last_5.count('banker') / len(last_5) if last_5 else 0
    features['last_5_player'] = last_5.count('player') / len(last_5) if last_5 else 0
    
    # 連續性特徵
    current_streak = 1
    if len(roadmap) >= 2:
        for i in range(len(roadmap)-1, 0, -1):
            if roadmap[i] == roadmap[i-1]:
                current_streak += 1
            else:
                break
    features['current_streak'] = current_streak
    
    # 原有俚語特徵
    last4 = roadmap[-4:] if len(roadmap) >= 4 else []
    features["is_3_1"] = int(len(last4) == 4 and last4[0] == last4[1] == last4[2] and last4[3] != last4[0])
    features["is_long_dragon"] = int(current_streak >= 5)
    features["is_break_pattern"] = int(len(roadmap) >= 4 and roadmap[-1] != roadmap[-3])
    
    # 轉換為 numpy array
    feature_values = list(features.values())
    return np.array([feature_values]), features

# 俚語型態命名
def detect_pattern_name(flags):
    if flags.get("is_3_1", 0):
        return "三帶一"
    elif flags.get("is_long_dragon", 0):
        return "長龍"
    elif flags.get("is_break_pattern", 0):
        return "破路"
    elif flags.get("current_streak", 0) >= 3:
        return f"連續{flags['current_streak']}局"
    else:
        return "一般模式"

# 主預測路由
@app.route("/predict", methods=["POST", "GET"])
def predict():
    try:
        if request.method == 'GET':
            roadmap_str = request.args.get('roadmap', '')
            roadmap = roadmap_str.split(',') if roadmap_str else []
        else:
            data = request.get_json()
            roadmap = data.get("roadmap", [])
        
        if isinstance(roadmap, str):
            roadmap = roadmap.split(',')
        
        if not roadmap or len(roadmap) < 4:
            return jsonify({
                "error": "請提供至少4局牌路",
                "example": "banker,player,banker,player"
            }), 400
        
        cleaned_roadmap = []
        for item in roadmap:
            if item.lower() in ['b', 'banker', '庄']:
                cleaned_roadmap.append('banker')
            elif item.lower() in ['p', 'player', '闲']:
                cleaned_roadmap.append('player')
            elif item.lower() in ['t', 'tie', '和']:
                cleaned_roadmap.append('tie')
            else:
                cleaned_roadmap.append(item.lower())
        
        X, feature_flags = enhanced_roadmap_to_features(cleaned_roadmap)
        
        if X is None:
            return jsonify({"error": "特徵提取失敗"}), 400
        
        # 模型預測
        lgb_pred = lgb_model.predict_proba(X)[0]
        xgb_pred = xgb_model.predict_proba(X)[0]
        final_pred = (np.array(lgb_pred) + np.array(xgb_pred)) / 2
        
        # 獲取預測概率
        banker_prob = round(final_pred[0], 4)
        player_prob = round(final_pred[1], 4)
        tie_prob = round(final_pred[2], 4)
        
        # 確定最高概率和對應的建議
        max_prob = max(banker_prob, player_prob, tie_prob)
        labels = ["banker", "player", "tie"]
        suggestion_index = np.argmax(final_pred)
        suggestion = labels[suggestion_index]
        
        # 添加观望逻辑 - 如果最高概率不够高，建议观望
        if max_prob < 0.5:  # 可以调整这个阈值
            suggestion = "觀望"
            confidence = 0
        else:
            confidence = round(max_prob, 4)
        
        return jsonify({
            "success": True,
            "model_loaded": model_loaded,  # 添加模型加载状态
            "predictions": {
                "banker": banker_prob,
                "player": player_prob,
                "tie": tie_prob
            },
            "suggestion": suggestion,
            "confidence": confidence,
            "pattern": detect_pattern_name(feature_flags),
            "current_streak": feature_flags.get('current_streak', 0),
            "input_length": len(cleaned_roadmap),
            "roadmap": cleaned_roadmap[-10:]
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "message": "預測過程中發生錯誤"
        }), 500

# 健康檢查路由
@app.route("/")
def health_check():
    return jsonify({
        "status": "active",
        "message": "Baccarat Prediction API is running",
        "model_loaded": model_loaded,  # 添加模型加载状态
        "endpoints": {
            "GET /predict?roadmap=banker,player,banker": "測試預測",
            "POST /predict": "正式預測"
        }
    })

# 測試路由
@app.route("/test")
def test_predict():
    test_roadmap = ['banker', 'player', 'banker', 'player', 'banker']
    X, features = enhanced_roadmap_to_features(test_roadmap)
    
    # 測試模型預測
    lgb_pred = lgb_model.predict_proba(X)[0]
    xgb_pred = xgb_model.predict_proba(X)[0]
    final_pred = (np.array(lgb_pred) + np.array(xgb_pred)) / 2
    
    return jsonify({
        "test_data": test_roadmap,
        "features": features,
        "feature_vector": X.tolist(),
        "lgb_pred": lgb_pred.tolist(),
        "xgb_pred": xgb_pred.tolist(),
        "final_pred": final_pred.tolist(),
        "model_loaded": model_loaded
    })

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
