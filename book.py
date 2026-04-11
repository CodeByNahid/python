# CodeByNahid
#Problem Link: https://codeforces.com/contest/279/problem/B
n, t = map(int, input().split())
a = list(map(int, input().split()))
 
r = -1
window_sum = 0
ans = 0
 
for l in range(n):
	while r + 1 < n and window_sum + a[r + 1] <= t:
		r += 1
		window_sum += a[r]
	ans = max(ans, r - l + 1)
	window_sum -= a[l]
 
print(ans)
