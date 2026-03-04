import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pickle


np.random.seed(42)
n_samples = 2000


X = np.random.rand(n_samples, 6) * np.array([1000, 50, 5, 5, 5, 5])

y = (
    X[:, 0] * 0.004 +  
    (50 - X[:, 1]) * 0.3 +  
    X[:, 2] * 0.5 +  
    X[:, 3] * 0.3 +  
    X[:, 4] * 0.2 +  
    X[:, 5] * 0.15   
)
y = np.maximum(y + np.random.randn(len(y)) * 0.5, 0)  

model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
model.fit(X, y)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print(f" Model trained on {n_samples} samples")
print(f" Saved model to model.pkl")
print(f"   Feature importance: {model.feature_importances_}")