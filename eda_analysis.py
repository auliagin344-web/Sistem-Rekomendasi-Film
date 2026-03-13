import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/movies_clean.csv")

genre_counts = df["genres"].str.split("|").explode().value_counts()

genre_counts.head(10).plot(kind="bar")
plt.title("Top Genres")
plt.savefig("eda_results/top_genres.png")
plt.show()
