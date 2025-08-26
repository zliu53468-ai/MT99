from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)

# ✅ 馬可夫轉移矩陣（根據最後一手預測下一手機率）
transition_matrix = {
    "莊": {"莊": 0.6, "閒": 0.3, "和": 0.1},
    "閒": {"莊": 0.4, "閒": 0.5, "和": 0.1},
    "和": {"莊": 0.5, "閒": 0.4, "和": 0.1}
}

# ✅ 特徵抽取函式
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

# ✅ 馬可夫預測函式
def markov_predict(roadmap):
    last = next((x for x in reversed(roadmap) if x in ["莊", "閒", "和"]), None)
    if last and last in transition_matrix:
        return transition_matrix[last]
    else:
        return {"莊": 0.33, "閒": 0.33, "和": 0.34}

# ✅ 混合預測函式（馬可夫 + 特徵加權）
def hybrid_predict(features, roadmap):
    markov = markov_predict(roadmap)
    banker_streak, player_streak, jump_count, long_banker, long_player = features[0]

    feature_score = {
        "莊": 0.5 + 0.05 * banker_streak + 0.03 * long_banker,
        "閒": 0.5 + 0.05 * player_streak + 0.03 * long_player,
        "和": 0.1 + 0.02 * jump_count
    }

    # 加權平均（馬可夫 70%，特徵 30%）
    final = {}
    for key in ["莊", "閒", "和"]:
        final[key] = 0.7 * markov[key] + 0.3 * feature_score[key]

    # 正規化
    total = sum(final.values())
    return {
        "banker": round(final["莊"] / total, 2),
        "player": round(final["閒"] / total, 2),
        "tie": round(final["和"] / total, 2)
    }

# ✅ API 路由
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    roadmap = data.get("roadmap", [])
    features = extract_features(roadmap)
    prediction = hybrid_predict(features, roadmap)
    return jsonify(prediction)

if __name__ == "__main__":
    app.run(debug=True)
