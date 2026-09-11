def minWindow( s: str, t: str) -> str:
    s_len, t_len = len(s), len(t)
    s_map, t_map = dict(), dict()

    for i in range(t_len):
        t_map[t[i]] = t_map.get(t[i], 0) + 1

    start, end = 0, 0
    have = 0
    i, j = -1, -1
    num = len(t_map)
    min_len = 100000

    while end < s_len:
        if t_map.get(s[end]):
            s_map[s[end]] = s_map.get(s[end], 0) + 1
            if t_map.get(s[end]) and s_map[s[end]] == t_map[s[end]]:
                have = have + 1

        while have == num:
            # 如果更小，更新最小字符串的index
            if min_len > end - start + 1:
                min_len = end - start + 1
                i, j = start, end
            # 移动头指针位置，并把头指针对应的s_map中的count减掉
            s_map[s[start]] = s_map.get(s[start]) - 1 if s_map.get(s[start]) else 0
            if t_map.get(s[start]) and s_map[s[start]] < t_map[s[start]]:
                have = have - 1
            start = start + 1
        end += 1

    return s[i:j + 1] if (i != -1 and j != -1) else ""

s = "ADOBECODEBANC"
t = "ABC"

print(minWindow(s, t))
