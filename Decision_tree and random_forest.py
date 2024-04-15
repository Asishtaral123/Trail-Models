import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


data = pd.read_csv("C:\\Users\\asish\\Documents\\soil-data (3).csv")

data = data.drop(columns=['Unnamed: 0'])

# Check class distribution for Crop_Type
print("Crop_Type class distribution:")
print(data['Crop_Type'].value_counts())

# Check class distribution for Fertilizer_Type
print("Fertilizer_Type class distribution:")
print(data['Fertilizer_Type'].value_counts())

from sklearn.preprocessing import MinMaxScaler, StandardScaler
numerical_features = ['Temperature', 'Humidity', 'Soil_Moisture', 'Raindrop', 'Soil_pH', 'Soil_Nitrogen', 'Soil_Phosphorus', 'Soil_Potassium']
min_max_scaler = MinMaxScaler()
standard_scaler = StandardScaler()
data_min_max_scaled = data.copy()
data_min_max_scaled[numerical_features] = min_max_scaler.fit_transform(data[numerical_features])
data_standard_scaled = data.copy()
data_standard_scaled[numerical_features] = standard_scaler.fit_transform(data[numerical_features])
print("Min-Max Scaled Data:")
print(data_min_max_scaled.head())

print("\nStandard Scaled Data:")
print(data_standard_scaled.head())

X = data[['Temperature', 'Humidity', 'Soil_Moisture', 'Raindrop', 'Soil_pH', 'Soil_Nitrogen', 'Soil_Phosphorus', 'Soil_Potassium']]
y = data[['Crop_Type', 'Fertilizer_Type']]


label_encoder = LabelEncoder()
data['Crop_Type'] = label_encoder.fit_transform(data['Crop_Type'])
data['Fertilizer_Type'] = label_encoder.fit_transform(data['Fertilizer_Type'])

# print(data.head(30))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

""" from sklearn.model_selection import GridSearchCV
param_grid = {
    'n_estimators' : [100, 200, 300],
    'max_depth' : [None, 10, 20],
    'min_samples_split' : [2, 5, 10],
    'min_samples_leaf' : [1, 2, 4]
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                           param_grid=param_grid,
                           cv=5,  # 5-fold cross-validation
                           scoring='accuracy',
                           n_jobs=-1)

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Parameters:", best_params)
print("Best Score:", best_score) """

# multiclass-multioutput is not supported in grid search cv

rf_classifier = RandomForestClassifier(n_estimators= 100, random_state= 42)
multi_target_classifier = MultiOutputClassifier(rf_classifier, n_jobs=-1)
multi_target_classifier.fit(X_train, y_train)


y_pred = multi_target_classifier.predict(X_test)

accuracy_crop_type = accuracy_score(y_test['Crop_Type'], y_pred[:, 0])
print("accuracy for crop type: ", accuracy_crop_type)

accuracy_fertilizer_type = accuracy_score(y_test['Fertilizer_Type'], y_pred[:, 1])
print("Accuracy for Fertilizer_Type:", accuracy_fertilizer_type) 


from sklearn.tree import DecisionTreeClassifier 

dt_classifier = DecisionTreeClassifier(random_state=42)
multi_target_classifier_dt = MultiOutputClassifier(dt_classifier, n_jobs=-1)
multi_target_classifier_dt.fit(X_train, y_train)

y_pred_dt = multi_target_classifier_dt.predict(X_test)

accuracy_crop_type_dt = accuracy_score(y_test['Crop_Type'].values, y_pred_dt[:, 0])
print("Accuracy for Crop_Type (Decision Tree):", accuracy_crop_type_dt)

accuracy_fertilizer_type_dt = accuracy_score(y_test['Fertilizer_Type'].values, y_pred_dt[:, 1])
print("Accuracy for Fertilizer_Type (Decision Tree):", accuracy_fertilizer_type_dt)
