#Problem Link: https://codeforces.com/contest/2217/problem/A
# CodeByNahid
 
def solve():
    n,k=map(int,input().split())
    v = list(map(int, input().split()))
    s=0
    for i in range(n):
        s+=v[i]
    if s&1 or (n*k)%2==0:
        print("YES")
    else:
        print("NO")
 
def main():
    t = int(input())
    for _ in range(t):
        solve()
        
if __name__ == "__main__":
    main()
