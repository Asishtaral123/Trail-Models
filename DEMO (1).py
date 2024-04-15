import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from minisom import MiniSom
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, LSTM

# Load and preprocess the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Convolutional Neural Network (CNN)
cnn_model = Sequential([
    Conv2D(32, kernel_size=(1, 1), activation='relu', input_shape=(1, 4, 1)),  # Adjusted input shape and kernel size
    MaxPooling2D(pool_size=(1, 1)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax')
])

cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn_model.fit(X_train.reshape(-1, 4, 1, 1), y_train, epochs=10, batch_size=32, verbose=0)

# Extract CNN features
cnn_features = cnn_model.predict(X_train.reshape(-1, 4, 1, 1))

# 2. Radial Basis Function Network (RBFN) implemented using RBF kernel in SVM
rbf_svm = SVC(kernel='rbf', gamma='scale')
rbf_svm.fit(X_train, y_train)

# 3. Long Short-Term Memory (LSTM)
lstm_model = Sequential([
    LSTM(50, input_shape=(1, 4)),
    Dense(3, activation='softmax')
])
lstm_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
lstm_model.fit(X_train.reshape(-1, 1, 4), y_train, epochs=10, batch_size=32, verbose=0)

# 4. Self-Organizing Maps (SOMs)
som_model = MiniSom(5, 5, 4)
som_model.random_weights_init(X_train)
som_model.train_random(X_train, 100)

# 5. Support Vector Machine (SVM)
# Combine features from all models

# Reshape cnn_features to have the same number of dimensions as rbf_svm.decision_function(X_train)
cnn_features_reshaped = cnn_features.reshape(-1, 1)

print("cnn_features_reshaped shape:", cnn_features_reshaped.shape)
print("rbf_svm decision function shape:", np.expand_dims(rbf_svm.decision_function(X_train), axis=1).shape)

# Flatten the output of rbf_svm.decision_function(X_train)
rbf_decision_function_flat = rbf_svm.decision_function(X_train).reshape(-1, len(np.unique(y_train)))
#print(rbf_decision_function_flat)

# Select the first 120 samples from cnn_features_reshaped
cnn_features_reshaped_selected = cnn_features_reshaped[:120]

# Combine features from all models
combined_features_train = np.concatenate([
    cnn_features_reshaped_selected, 
    rbf_decision_function_flat,
    lstm_model.predict(X_train.reshape(-1, 1, 4))[:len(cnn_features_reshaped_selected)], 
    som_model.quantization(X_train)[:len(cnn_features_reshaped_selected)]
], axis=1)

# Select the first 30 samples from each array
cnn_features_test_selected = cnn_model.predict(X_test.reshape(-1, 4, 1, 1))[:30]
rbf_decision_function_test_selected = rbf_svm.decision_function(X_test)[:30]
lstm_features_test_selected = lstm_model.predict(X_test.reshape(-1, 1, 4))[:30]
som_features_test_selected = som_model.quantization(X_test)[:30]

# Combine features from all models for the test set
combined_features_test = np.concatenate([
    cnn_features_test_selected, 
    rbf_decision_function_test_selected[:, :1],  # Extracting only the first column from the decision function values of the RBF SVM classifier
    lstm_features_test_selected, 
    som_features_test_selected
], axis=1)

# Train SVM classifier
svm_classifier = SVC()
svm_classifier.fit(combined_features_train, y_train)

# Evaluate the model
y_pred = svm_classifier.predict(combined_features_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("cnn_features shape:", cnn_features.shape)
print("rbf_svm decision function shape:", rbf_svm.decision_function(X_train).reshape(-1, 1).shape)
print("lstm_model prediction shape:", lstm_model.predict(X_train.reshape(-1, 1, 4)).shape)
print("som_model quantization shape:", som_model.quantization(X_train).shape)