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
            # 偵測破點
            if len(clean_road) > streak_len:
                prev = clean_road[-streak_len-1]
                if prev != clean_road[-1]:
                    patterns.append("long_dragon_break")
            # 偵測回補（檢查索引安全性）
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
