import pandas as pd

df = pd.read_csv("enjoysport.csv")

x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

s = ['0'] * len(x[0])

for i in range(len(x)):
    if y[i] == 1:
        for j in range(len(s)):
            if s[j] == '0':
                s[j] = x[i][j]
            elif s[j] != x[i][j]:
                s[j] = '?'

g = ['?'] * len(s)

for i in range(len(x)):
    if y[i] == 0:
        for j in range(len(g)):
            if s[j] != '?' and s[j] != x[i][j]:
                g[j] = s[j]

print("Find-S:", s)
print("Candidate Elimination:")
print("S:", s)
print("G:", g)
