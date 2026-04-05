#Problem Link: https://codeforces.com/contest/2218/problem/B
# CodeByNahid

def solve():
    n = 7
    v = list(map(int, input().split()))
    
    v.sort(reverse=True) 
    
    s = v[0]
    for i in range(1, n):
        s -= v[i]
    
    print(s)

def main():
    t = int(input())
    for _ in range(t):
        solve()

if __name__ == "__main__":
    main()
