from flask import Flask, request, jsonify
import os

app = Flask(__name__)

def detect_patterns(roadmap):
    patterns = []

    if not roadmap or not isinstance(roadmap, list):
        return patterns

    clean_road = [x for x in roadmap if x in ["莊", "閒"]]
    if not clean_road:
        return patterns

    # 長龍偵測
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
                prev_index = len(clean_road) - streak_len - 1
                if prev_index >= 0 and clean_road[prev_index] != clean_road[-1]:
                    patterns.append("long_dragon_break")
            if len(clean_road) >= streak_len + 2:
                if (clean_road[-streak_len-1] != clean_road[-1] and
                    clean_road[-1] == clean_road[-streak_len-2]):
                    patterns.append("long_dragon_recover")

    # 單跳偵測
    if len(clean_road) >= 4:
        recent = clean_road[-4:]
        if all(recent[i] != recent[i-1] for i in range(1, 4)):
            patterns.append("single_jump")

    # 雙跳一房兩廳偵測
    if len(clean_road) >= 6:
        last6 = clean_road[-6:]
        if (last6[0] == last6[1] and
            last6[2] == last6[3] and
            last6[4] == last6[5] and
            last6[0] != last6[2]):
            patterns.append("double_jump_room")

    return patterns

@app.route("/detect", methods=["POST"])
def detect():
    data = request.get_json()
    roadmap = data.get("roadmap")
    result = detect_patterns(roadmap)
    return jsonify({"patterns": result})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
