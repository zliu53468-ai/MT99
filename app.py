# 為了處理 CORS (跨來源資源共享)，需要安裝此套件：
# pip install flask-cors

from flask import Flask, request, jsonify
from flask_cors import CORS  # 導入 CORS
import os

app = Flask(__name__)
CORS(app)  # 啟用 CORS，允許跨網域請求

def detect_patterns(roadmap):
    """
    偵測當前牌路模式：
    - single_jump: 單跳
    - double_jump_room: 雙跳一房兩廳
    - long_dragon_break: 長龍破點預警
    - long_dragon_recover: 長龍斷後回補
    """
    patterns = []

    if not roadmap or not isinstance(roadmap, list):
        return patterns  # 防呆：空值或格式錯誤直接返回

    clean_road = [x for x in roadmap if x in ["莊", "閒"]]  # 過濾和局
    if not clean_road:
        return patterns  # 全是和局，直接返回

    # --- 長龍檢測 ---
    if len(clean_road) >= 4:
        streak_len = 1
        for i in range(len(clean_road)-1, 0, -1):
            if clean_road[i] == clean_road[i-1]:
                streak_len += 1
            else:
                break
        if streak_len >= 4:
            patterns.append("long_dragon")
            # 偵測破點（索引安全）
            if len(clean_road) > streak_len:
                prev_index = len(clean_road) - streak_len - 1
                if prev_index >= 0 and clean_road[prev_index] != clean_road[-1]:
                    patterns.append("long_dragon_break")
            # 偵測回補（索引安全）
            if len(clean_road) >= streak_len + 2:
                if (clean_road[-streak_len-1] != clean_road[-1] and
                    clean_road[-1] == clean_road[-streak_len-2]):
                    patterns.append("long_dragon_recover")

    # --- 單跳檢測（最近 4 局交錯）---
    if len(clean_road) >= 4:
        recent = clean_road[-4:]
        if all(recent[i] != recent[i-1] for i in range(1, 4)):
            patterns.append("single_jump")

    # --- 雙跳一房兩廳檢測（放寬條件）---
    if len(clean_road) >= 6:
        last6 = clean_road[-6:]
        if (last6[0] == last6[1] and
            last6[2] == last6[3] and
            last6[4] == last6[5] and
            last6[0] != last6[2]):
            patterns.append("double_jump_room")

    return patterns

# 修正：將端點從 "/detect" 改為 "/predict" 以匹配前端程式碼
@app.route("/predict", methods=["POST"])
def detect():
    data = request.get_json()
    roadmap = data.get("roadmap")
    result = detect_patterns(roadmap)
    
    # 因為你的前端預期回傳的格式是 {banker: ..., player: ..., tie: ...}
    # 但你的後端目前只會回傳 {patterns: ...}
    # 這裡我提供一個暫時的範例回傳，你需要根據你的AI模型調整這段邏輯
    # 假設你用patterns來做預測，這裡只是範例
    
    # 判斷是否有長龍或單跳
    has_long_dragon = "long_dragon" in result
    has_single_jump = "single_jump" in result

    # 根據偵測到的模式返回不同的機率
    if has_long_dragon:
        # 如果有長龍，預測長龍方勝率高
        predictions = {"banker": 0.55, "player": 0.40, "tie": 0.05}
    elif has_single_jump:
        # 如果有單跳，預測跳方勝率高
        predictions = {"banker": 0.40, "player": 0.55, "tie": 0.05}
    else:
        # 預設平均機率
        predictions = {"banker": 0.45, "player": 0.45, "tie": 0.10}

    # 這裡的範例是基於你提供的偵測邏輯，並非一個完整的 AI 預測模型
    return jsonify(predictions)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

