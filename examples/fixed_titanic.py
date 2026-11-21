# The same model after applying ML Pipeline Auditor's fixes:
# split first, every learned step inside one pipeline, tuning by cross-validation,
# the test set used once, AUC from probabilities.
import json
import os

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

SEED = int(os.environ.get("SEED", 0))
df = pd.read_csv(os.path.join(os.path.dirname(__file__), "titanic3.csv"))
TARGET = "survived"
y = df[TARGET]
X = df[["pclass", "age", "sibsp", "parch", "fare", "sex", "embarked", "ticket", "cabin", "home.dest"]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)

num = ["pclass", "age", "sibsp", "parch", "fare"]
cat = ["sex", "embarked"]
high_card = ["ticket", "cabin", "home.dest"]
pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ("te", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="missing")),
                     ("enc", TargetEncoder(cv=KFold(5, shuffle=True, random_state=SEED)))]), high_card),
])
pipe = Pipeline([("pre", pre), ("smote", SMOTE(random_state=SEED)),
                 ("rf", RandomForestClassifier(random_state=SEED))])
search = GridSearchCV(pipe, {"rf__max_depth": [3, 5, 8, 12, None]},
                      cv=StratifiedKFold(5, shuffle=True, random_state=SEED), scoring="f1")
search.fit(X_train, y_train)
pred = search.predict(X_test)
proba = search.predict_proba(X_test)[:, 1]
print(json.dumps({"f1": f1_score(y_test, pred), "auc": roc_auc_score(y_test, proba)}))
