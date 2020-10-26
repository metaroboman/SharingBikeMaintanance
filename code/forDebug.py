m, n = 3, 3
values = list(range(1, 10))
target = [[0] * n for i in range(m)]
r, c = 0, 0
di = [[0, 1], [-1, 0], [0, -1], [1, 0]]
cursor = 0
for j, i in enumerate(values):
    # 填当前位置
    target[r][c] = i
    # 判断下一个位置
    nr, nc = r + di[cursor][0], c + di[cursor][1]
    if nr == m or nc == n or target[r][c] != 0:
        cursor = (cursor + 1) % 4
        r, c = r + di[cursor][0], c + di[cursor][1]

for t in target:
    re = ''
    for _ in t:
        re += str(_) + ' '
        print(re)