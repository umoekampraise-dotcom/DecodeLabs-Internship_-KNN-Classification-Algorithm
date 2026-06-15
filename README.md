# DecodeLabs-Internship_-KNN-Classification-Algorithm
Iris flower classifier built in Python using KNN algorithm, StandardScaler and F1 Score evaluation — DecodeLabs 

# 🌸 Iris Flower Classification — KNN Supervised Learning


## 🎯 Results — Two Datasets, One Pipeline

| Dataset | Features | Samples | F1 Score | Accuracy |
|---------|----------|---------|----------|----------|
| Iris    | 4        | 150     | 1.0000   | 100%     |


After completing Iris — I independently applied the exact same 
pipeline to the Wine dataset without any guidance.
That second result proved I understood the pipeline — 
not just followed instructions.

---

## 📊 Dataset
- **Name:** Iris Dataset (Fisher, 1936)
- **Samples:** 150 flowers
- **Classes:** 3 — Setosa, Versicolor, Virginica
- **Features:** 4 — Sepal Length, Sepal Width, Petal Length, Petal Width
- **Source:** Built into Scikit-Learn

---

## 📈 Confusion Matrix

PREDICTED
          Setosa  Versicolor  Virginica



          Zero misclassifications across all 3 species.

---

## 🔧 Complete Pipeline

### Step 1 — Import Libraries
```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
```

### Step 2 — Load Dataset
```python
iris = load_iris()
X = iris.data
y = iris.target
df = pd.DataFrame(X, columns=iris.feature_names)
```

### Step 3 — Handle Missing Data
```python
df.replace(0, np.nan, inplace=True)
df.fillna(df.mean(), inplace=True)
```

### Step 4 — Train-Test Split 80/20
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)
```

### Step 5 — Feature Scaling
```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)  # transform only — never fit_transform
```

### Step 6 — Define and Train Model
```python
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
```

### Step 7 — Evaluate
```python
predictions = model.predict(X_test)
cm = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(cm)
print("F1 Score:", f1_score(y_test, predictions, average='weighted'))
print("Accuracy:", accuracy_score(y_test, predictions))
```

---

## 💡 What I Learned — Including My Mistakes

The most surprising discovery was that 99% accuracy can be 
completely meaningless — what the industry calls the 
**Accuracy Mirage.**

On imbalanced datasets — an AI that always predicts the 
majority class scores 99% while never catching a single 
real case. F1 Score exposes this lie.

I also discovered that using fit_transform on test data — 
a mistake I made initially — silently causes data leakage 
and produces unreliable results.

Fixing that one line improved my Wine classifier from 
94.65% to 97.28% accuracy.

Small code mistakes — huge real world consequences.


---

## 🛠️ Technologies Used
- Python 3
- NumPy
- Pandas
- Scikit-Learn
- PyCharm IDE

---

## 🚀 How to Run
1. Clone this repository
2. Install dependencies:


---

## 👨‍💻 About Me
**Torobong Umoekam Edet**
AI Engineering Intern at DecodeLabs | AI Automation Specialist
Python | Machine Learning | n8n | Make.com | Zapier

📧 umoekampraise@gmail.com
🔗 LinkedIn: [https://www.linkedin.com/posts/torobong-umoekam-32b0b2291_machinelearning-supervisedlearning-knn-activity-7469683263650041856-ocld?utm_source=social_share_send&utm_medium=member_desktop_web&rcm=ACoAAEar2lMBqELbRbyxgXZ3BWJsG7gKPIb8iOQ]
