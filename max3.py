def maxm(a,b,c):
    if(a>=b) and (a>=c):
        return a
    elif b>=a and b>=c:
        return b
    else:
        return c
print("Enter three nos.")
a=int(input())
b=int(input())
c=int(input())
print("Maximum no. is:",maxm(a,b,c))
