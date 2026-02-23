import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

# LOAD THE DATASET
dataset= pd.read_csv('/Users/adityayadav/Desktop/ DS  Nareshit/All_Projects/Resturant_review/Restaurant_Reviews.tsv',delimiter='\t', quoting=3)

# duplicated the dataset to better accuaracy
dataset_duplicated = pd.concat([dataset, dataset], ignore_index=True)
dataset_duplicated = dataset_duplicated.sample(frac=1, random_state=42).reset_index(drop=True)


# filtering and cleaning the by using the Nlp
ps=PorterStemmer()
wordnet= WordNetLemmatizer()

corpus=[]
for i in range(len(dataset_duplicated)):
    review=re.sub('[^a-zA-Z]',' ', dataset_duplicated['Review'][i])
    review=review.lower()
    review=review.split()
    review= [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus.append(review)
    
# use the bow model to separate the data model
cntvect= CountVectorizer(max_features=1500)
X=cntvect.fit_transform(corpus).toarray()
y=dataset_duplicated.iloc[:,1].values

# separate the model for the train and test split the data
X_train, X_test, y_train,y_test= train_test_split(X,y, test_size=0.20, random_state=0)

# ── HELPER: print results neatly ──────────────────────────────────────────────
def evaluate(name, y_test, y_pred, y_train_pred=None):
    print(f"\n{'='*50}")
    print(f"  {name}")
    print('='*50)
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print(f"Test  Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    if y_train_pred is not None:
        print(f"Train Accuracy : {accuracy_score(y_train, y_train_pred):.4f}")

# ── 1. DECISION TREE ──────────────────────────────────────────────────────────
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
evaluate("Decision Tree", y_test, dt.predict(X_test), dt.predict(X_train))

 

# ── 2. RANDOM FOREST ──────────────────────────────────────────────────────────
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
evaluate("Random Forest", y_test, rf.predict(X_test), rf.predict(X_train))


# ── 3. MULTINOMIAL NAIVE BAYES ────────────────────────────────────────────────
mnb = MultinomialNB()
mnb.fit(X_train, y_train)
evaluate("Multinomial Naive Bayes", y_test, mnb.predict(X_test))


# ── 4. XGBOOST ────────────────────────────────────────────────────────────────
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb.fit(X_train, y_train)
evaluate("XGBoost", y_test, xgb.predict(X_test), xgb.predict(X_train))

# ── ACCURACY COMPARISON CHART ─────────────────────────────────────────────────
models      = ["Decision Tree", "Random Forest", "Multinomial NB", "XGBoost"]
accuracies  = [
    accuracy_score(y_test, dt.predict(X_test)),
    accuracy_score(y_test, rf.predict(X_test)),
    accuracy_score(y_test, mnb.predict(X_test)),
    accuracy_score(y_test, xgb.predict(X_test)),
]



plt.figure(figsize=(8, 5))
sns.barplot(x=models, y=accuracies, palette="viridis")
plt.title("Model Accuracy Comparison (Duplicated Dataset)")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.01, f"{v:.3f}", ha='center', fontweight='bold')
plt.tight_layout()
plt.show()





