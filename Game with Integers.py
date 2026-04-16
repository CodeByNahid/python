#https://codeforces.com/contest/1899/problem/A
# CodeByNahid
 
def solve():
   n=int(input())
   print("Second") if n%3==0 else print("First")
 
def main():
    t = int(input())
    for _ in range(t):
        solve()
 
if __name__ == "__main__":
    main()
