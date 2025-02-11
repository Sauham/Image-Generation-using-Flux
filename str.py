s="Example"
def remove(s, i):
    # Write code here
    for i in range(1,len(s)):
        return s[::i]+s[i+1::]

print(remove(s, 2))
    