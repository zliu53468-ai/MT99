import os
from flask import Flask, jsonify, request
from flask_cors import CORS

# 創建一個 Flask 實例
# 你的程式碼裡沒有這一行，這是啟動 Web 服務的關鍵
app = Flask(__name__)
# 啟用 CORS，允許跨域請求
CORS(app)

# 你提供的牌路偵測邏輯
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
        return patterns

    clean_road = [x for x in roadmap if x in ["莊", "閒"]]
    if not clean_road:
        return patterns

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
            if len(clean_road) > streak_len:
                prev = clean_road[-streak_len-1]
                if prev != clean_road[-1]:
                    patterns.append("long_dragon_break")
            if len(clean_road) >= streak_len + 2:
                if (clean_road[-streak_len-1] != clean_road[-1] and
                    clean_road[-1] == clean_road[-streak_len-2]):
                    patterns.append("long_dragon_recover")

    # --- 單跳檢測 ---
    if len(clean_road) >= 4 and all(
        clean_road[i] != clean_road[i-1] for i in range(1, len(clean_road))
    ):
        patterns.append("single_jump")

    # --- 雙跳一房兩廳檢測 ---
    if len(clean_road) >= 6:
        last6 = clean_road[-6:]
        if (last6[0] == last6[1] and
            last6[2] == last6[3] and
            last6[4] == last6[5] and
            last6[0] != last6[2] and
            last6[0] != last6[4] and
            last6[2] != last6[4]):
            patterns.append("double_jump_room")

    return patterns

# 定義一個 API 端點
@app.route('/detect', methods=['POST'])
def handle_detect():
    """
    接收 POST 請求，執行牌路偵測。
    """
    data = request.get_json(silent=True)
    if not data or 'roadmap' not in data:
        return jsonify({"error": "Invalid input, 'roadmap' key is missing."}), 400

    roadmap = data['roadmap']
    if not isinstance(roadmap, list):
        return jsonify({"error": "'roadmap' must be a list."}), 400

    patterns = detect_patterns(roadmap)
    return jsonify({"patterns": patterns})

# 檢查是否為本地運行
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=os.environ.get('PORT', 5000))
