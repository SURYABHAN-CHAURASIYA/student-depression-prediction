import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler

# 1. Dataset Load (NA values handle karo)
df = pd.read_csv("student_depression_dataset.csv", na_values="?") 

# Column names fix karo (extra spaces hatao)
df.columns = df.columns.str.strip()

# TARGET COLUMN KA NAAM CHECK KARO:
# Dataset me 'Depression' hai, aur suicidal thoughts wala column 'Have you ever had suicidal thoughts 0' hai
target = "Depression"
X = df.drop(columns=[target, "id", "City"]) 
y = df[target]

# Encoding
le = LabelEncoder()
y = le.fit_transform(y)

# Columns list (Dataset ke exact naam)
cat_cols = ["Gender", "Profession", "Sleep Duration", "Dietary Habits", "Degree", 
            "Have you ever had suicidal thoughts 0", "Family History of Mental Illness"]
num_cols = ["Age", "Academic Pressure", "Work Pressure", "CGPA", "Study Satisfaction", 
            "Job Satisfaction", "Work/Study Hours", "Financial Stress"]

# Pipeline
preprocess = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num_cols),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), 
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols)
])

# Balancing
rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(X, y)

model = Pipeline([
    ("preprocess", preprocess),
    ("classifier", RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Save
os.makedirs("models", exist_ok=True)
with open("models/depression_model.pkl", "wb") as f: pickle.dump(model, f)
with open("models/label_encoder.pkl", "wb") as f: pickle.dump(le, f)
print("Model trained successfully!")