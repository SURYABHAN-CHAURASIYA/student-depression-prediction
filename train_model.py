import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler

def get_model():

    # Dataset Load
    df = pd.read_csv("student_depression_dataset.csv", na_values="?")
    df.columns = df.columns.str.strip()

    target = "Depression"
    X = df.drop(columns=[target, "id", "City"])
    y = df[target]

    # Encoding
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Columns
    cat_cols = ["Gender", "Profession", "Sleep Duration", "Dietary Habits", "Degree", 
                "Have you ever had suicidal thoughts 0", "Family History of Mental Illness"]

    num_cols = ["Age", "Academic Pressure", "Work Pressure", "CGPA", "Study Satisfaction", 
                "Job Satisfaction", "Work/Study Hours", "Financial Stress"]

    # Preprocess
    preprocess = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols)
    ])

    # Balancing
    rus = RandomUnderSampler(random_state=42)
    X_res, y_res = rus.fit_resample(X, y)

    # Model
    model = Pipeline([
        ("preprocess", preprocess),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42
        ))
    ])

    # Train
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)

    return model, le
