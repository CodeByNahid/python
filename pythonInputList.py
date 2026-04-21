points = [tuple(map(int, input().split())) for _ in range(3)]
v1 = sorted(x for x, _ in points)
v2 = sorted(y for _, y in points)
