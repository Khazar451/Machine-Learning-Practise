import pandas as pd
from sklearn.model_selection import train_test_split
import torch

# 1. Load the dataset directly via pandas from Géron's GitHub repo
url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
df = pd.read_csv(url).dropna()

# 2. Add the 3 ratios so it matches the 8 features in fetch_california_housing()
df["AveRooms"] = df["total_rooms"] / df["households"]
df["AveBedrms"] = df["total_bedrooms"] / df["households"]
df["AveOccup"] = df["population"] / df["households"]

feature_cols = [
    "median_income",
    "housing_median_age",
    "AveRooms",
    "AveBedrms",
    "population",
    "AveOccup",
    "latitude",
    "longitude",
]

# Scale house values to units of $100k
X = df[feature_cols].values
y = (df["median_house_value"].values / 100_000.0).reshape(-1, 1)

# 3. Splits (60% Train, 20% Valid, 20% Test)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, test_size=0.25, random_state=42
)

# 4. Standardize and convert to float32 PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_valid = torch.tensor(X_valid, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

means = X_train.mean(dim=0, keepdim=True)
stds = X_train.std(dim=0, keepdim=True)

X_train = (X_train - means) / stds
X_valid = (X_valid - means) / stds
X_test = (X_test - means) / stds

y_train = torch.tensor(y_train, dtype=torch.float32)
y_valid = torch.tensor(y_valid, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# 5. Model Parameters
torch.manual_seed(42)
n_features = X_train.shape[1]
w = torch.randn((n_features, 1), requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

# 6. Training Loop (Exact book code)
learning_rate = 0.4
n_epochs = 20

for epoch in range(n_epochs):
    y_pred = X_train @ w + b
    loss = ((y_pred - y_train) ** 2).mean()
    loss.backward()

    with torch.no_grad():
        b -= learning_rate * b.grad
        w -= learning_rate * w.grad
        b.grad.zero_()
        w.grad.zero_()

    print(f"Epoch {epoch + 1:2d}/{n_epochs}, Loss: {loss.item():.4f}")