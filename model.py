import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from pickle import load, dump
import matplotlib.pyplot as plt

path = "https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"


df = pd.read_csv(path)
print(df.Hours)
print(df)
X = df[["Hours"]]
y = df[["Scores"]]
print(X)
seborn = sns.scatterplot(df.Hours, df.Scores)
LR_model = LinearRegression()
LR_model = LR_model.fit(X, y)

dump(seborn, open("scater.pkl", "wb"))

plt.savefig("sebon.jpeg")

dump(LR_model, open("model.pkl", "wb"))
# seb=load(open("scater.pkl", "rb"))
# print(seb)

# print(2.9 % 10)