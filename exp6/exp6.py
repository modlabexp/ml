import kagglehub
import pandas as pd
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares

path = kagglehub.dataset_download("ruchi798/bookcrossing-dataset")

r = pd.read_csv(path + "/Book reviews/Book reviews/BX-Book-Ratings.csv", sep=";", encoding="latin-1")
b = pd.read_csv(path + "/Book reviews/Book reviews/BX_Books.csv", sep=";", encoding="latin-1")

r = r[r["Book-Rating"] > 0]
r = r.sample(n=min(10000, len(r)), random_state=42)

r["u"], users = pd.factorize(r["User-ID"])
r["b"], isbns = pd.factorize(r["ISBN"])

x = csr_matrix(
    (r["Book-Rating"], (r["u"], r["b"])),
    shape=(len(users), len(isbns))
)

model = AlternatingLeastSquares(factors=20, iterations=10, random_state=42)
model.fit(x)

print("Book Recommendations for 3 Users")

for uid in range(min(3, len(users))):
    p = model.recommend(uid, x[uid], N=3)

    print("\nUser ID:", users[uid])
    print("Recommended Books:")

    for i in p[0]:
        title = b.loc[b["ISBN"] == isbns[i], "Book-Title"]
        if not title.empty:
            print("-", title.iloc[0])
