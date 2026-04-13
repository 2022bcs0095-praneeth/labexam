import json
import pickle
from sklearn.linear_model import LinearRegression
import numpy as np

# dummy data
X = np.array([[1],[2],[3],[4],[5]])
y = np.array([2,4,6,8,10])

# train model
model = LinearRegression()
model.fit(X,y)

# fake prediction + mse
pred = model.predict(X)
mse = ((pred - y) ** 2).mean()

# save model
with open("model.pkl","wb") as f:
    pickle.dump(model,f)

# save metrics
with open("metrics.json","w") as f:
    json.dump({"mse": float(mse)}, f)

print("Training done. MSE:", mse)