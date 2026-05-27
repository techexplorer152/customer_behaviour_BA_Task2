import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


df = pd.read_csv("customer_booking.csv", encoding="ISO-8859-1")


print(df["booking_complete"].value_counts())

day_mapping={"Mon":1,"Tue":2,"Wed":3,"Thu":4,"Fri":5,"Sat":6,"Sun":7}
df["flight_day"] = df["flight_day"].map(day_mapping)

df = pd.get_dummies(df, columns=["sales_channel", "trip_type"], drop_first=True, dtype=int)


df["booking_origin"] = df["booking_origin"].astype("category").cat.codes
df["route"] = df["route"].astype("category").cat.codes

print()
#ds=df["booking_origin"].value_counts()df.to_csv("countries.csv")
df = df.loc[:, ~df.columns.duplicated()]

#Step 2: ML
X = df.drop(columns=["booking_complete"])
y = df["booking_complete"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)


#Step 3: Evaluation


y_pred = model.predict(X_test)


print(classification_report(y_test, y_pred))


importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)


print(feature_importance_df)

