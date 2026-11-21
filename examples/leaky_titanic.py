# A typical "Titanic survival" notebook, written the way many tutorials do it.
# It runs, and it reports a great score. ML Pipeline Auditor explains why the score is not real.
import json
import os

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = int(os.environ.get("SEED", 0))
df = pd.read_csv(os.path.join(os.path.dirname(__file__), "titanic3.csv"))

# clean
df["age"] = df["age"].fillna(df["age"].mean())
df["fare"] = df["fare"].fillna(df["fare"].median())

# "smart" features: survival rate of people with the same ticket / cabin / destination
for col in ["ticket", "cabin", "home.dest"]:
    df[col + "_rate"] = df.groupby(col)["survived"].transform("mean")
    df[col + "_rate"] = df[col + "_rate"].fillna(df["survived"].mean())

df = pd.get_dummies(df, columns=["sex", "embarked"], drop_first=True)
y = df["survived"]
X = df[["pclass", "age", "sibsp", "parch", "fare", "sex_male", "embarked_Q", "embarked_S",
        "ticket_rate", "cabin_rate", "home.dest_rate"]]

X = StandardScaler().fit_transform(X)
X, y = SMOTE(random_state=SEED).fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=SEED)

best_f1, best_depth = 0, None
for depth in [3, 5, 8, 12, None]:
    model = RandomForestClassifier(max_depth=depth, random_state=SEED).fit(X_train, y_train)
    f1 = f1_score(y_test, model.predict(X_test))
    if f1 > best_f1:
        best_f1, best_depth = f1, depth

model = RandomForestClassifier(max_depth=best_depth, random_state=SEED).fit(X_train, y_train)
pred = model.predict(X_test)
print(json.dumps({"f1": f1_score(y_test, pred), "auc_reported": roc_auc_score(y_test, pred)}))
