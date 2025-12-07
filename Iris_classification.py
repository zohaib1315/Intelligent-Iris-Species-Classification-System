import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

# Load iris dataset - USE ALL 4 FEATURES
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Binary classification (Setosa vs non-Setosa)
y = (y == 0).astype(int)

def perceptron(X, y):
    Xb = np.insert(X, 0, 1, axis=1)
    weights1 = np.ones(Xb.shape[1])
    lr = 0.1
    n = Xb.shape[0]

    for i in range(1000):
        j = np.random.randint(0, n)
        y_hat = step(np.dot(Xb[j], weights1))
        weights1 = weights1 + lr * (y[j] - y_hat) * Xb[j]

    intercept = weights1[0]
    coef = weights1[1:]
    return intercept, coef

def step(z):
    return 1 if z > 0 else 0

def perceptron_sigmoid(X, y):
    Xb = np.insert(X, 0, 1, axis=1)
    weights2 = np.ones(Xb.shape[1])
    lr = 0.1
    n = Xb.shape[0]

    for i in range(1000):
        j = np.random.randint(0, n)
        z = np.dot(Xb[j], weights2)
        y_hat = sigmoid(z)
        gradient = (y[j] - y_hat) * y_hat * (1 - y_hat)
        weights2 = weights2 + lr * gradient * Xb[j]

    intercept = weights2[0]
    coef = weights2[1:]
    return intercept, coef

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

# Train/test split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Scale the data
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

# Train models
intercept_, coef_ = perceptron(X_train_scaled, y_train)
intercept_s, coef_s = perceptron_sigmoid(X_train_scaled, y_train)
lor = LogisticRegression(random_state=42)
lor.fit(X_train_scaled, y_train)

# Predictions and accuracies
def predict_perceptron(X, intercept, coef):
    Xb = np.insert(X, 0, 1, axis=1)
    weights = np.insert(coef, 0, intercept)
    predictions = np.dot(Xb, weights) > 0
    return predictions.astype(int)

def predict_perceptron_sigmoid(X, intercept, coef):
    Xb = np.insert(X, 0, 1, axis=1)
    weights = np.insert(coef, 0, intercept)
    probabilities = sigmoid(np.dot(Xb, weights))
    return (probabilities > 0.5).astype(int)

y_pred_step = predict_perceptron(X_test_scaled, intercept_, coef_)
y_pred_sigmoid = predict_perceptron_sigmoid(X_test_scaled, intercept_s, coef_s)
y_pred_lr = lor.predict(X_test_scaled)

acc_step = accuracy_score(y_test, y_pred_step)
acc_sigmoid = accuracy_score(y_test, y_pred_sigmoid)
acc_lr = accuracy_score(y_test, y_pred_lr)

print("="*60)
print("PERFORMANCE SUMMARY (Using ALL 4 Features)")
print("="*60)
print(f"Perceptron (Step):      {acc_step:.4f}")
print(f"Perceptron (Sigmoid):   {acc_sigmoid:.4f}")
print(f"Logistic Regression:    {acc_lr:.4f}")
print("="*60)

# =================================================================
# VISUALIZATION 1: Petal Features Only (2D)
# =================================================================
feature_indices = [2, 3]
X_train_petal = X_train_scaled[:, feature_indices]

# Train 2D models for petal features
intercept_step_2d, coef_step_2d = perceptron(X_train_petal, y_train)
intercept_sigmoid_2d, coef_sigmoid_2d = perceptron_sigmoid(X_train_petal, y_train)
lor_2d = LogisticRegression(random_state=42)
lor_2d.fit(X_train_petal, y_train)

# Decision boundaries for petal features
m_step = -(coef_step_2d[0] / coef_step_2d[1])
b_step = -(intercept_step_2d / coef_step_2d[1])
m_sigmoid = -(coef_sigmoid_2d[0] / coef_sigmoid_2d[1])
b_sigmoid = -(intercept_sigmoid_2d / coef_sigmoid_2d[1])
m_lr = -(lor_2d.coef_[0][0] / lor_2d.coef_[0][1])
b_lr = -(lor_2d.intercept_[0] / lor_2d.coef_[0][1])

x_min1, x_max1 = X_train_petal[:, 0].min() - 0.5, X_train_petal[:, 0].max() + 0.5
x_input1 = np.linspace(x_min1, x_max1, 100)
y_step = m_step * x_input1 + b_step
y_sigmoid = m_sigmoid * x_input1 + b_sigmoid
y_lr = m_lr * x_input1 + b_lr

# =================================================================
# VISUALIZATION 2: PCA (All 4 Features)
# =================================================================
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)

intercept_step_pca, coef_step_pca = perceptron(X_train_pca, y_train)
intercept_sigmoid_pca, coef_sigmoid_pca = perceptron_sigmoid(X_train_pca, y_train)
lor_pca = LogisticRegression(random_state=42)
lor_pca.fit(X_train_pca, y_train)

m_step_pca = -(coef_step_pca[0] / coef_step_pca[1])
b_step_pca = -(intercept_step_pca / coef_step_pca[1])
m_sigmoid_pca = -(coef_sigmoid_pca[0] / coef_sigmoid_pca[1])
b_sigmoid_pca = -(intercept_sigmoid_pca / coef_sigmoid_pca[1])
m_lr_pca = -(lor_pca.coef_[0][0] / lor_pca.coef_[0][1])
b_lr_pca = -(lor_pca.intercept_[0] / lor_pca.coef_[0][1])

x_min2, x_max2 = X_train_pca[:, 0].min() - 0.5, X_train_pca[:, 0].max() + 0.5
x_input2 = np.linspace(x_min2, x_max2, 100)
y_step_pca = m_step_pca * x_input2 + b_step_pca
y_sigmoid_pca = m_sigmoid_pca * x_input2 + b_sigmoid_pca
y_lr_pca = m_lr_pca * x_input2 + b_lr_pca

# =================================================================
# CREATE 2x3 GRID FOR BOTH VISUALIZATIONS
# =================================================================
plt.figure(figsize=(15, 10))

# Row 1: Petal Features Only
plt.subplot(2, 3, 1)
plt.plot(x_input1, y_step, color='red', linewidth=3)
plt.scatter(X_train_petal[:, 0], X_train_petal[:, 1], c=y_train, cmap='RdYlBu', s=80, edgecolor='black')
plt.xlabel('Petal Length', fontweight='bold')
plt.ylabel('Petal Width', fontweight='bold')
plt.title('Perceptron (Step)\nUsing Petal Features Only', fontweight='bold', fontsize=11)
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 2)
plt.plot(x_input1, y_sigmoid, color='orange', linewidth=3)
plt.scatter(X_train_petal[:, 0], X_train_petal[:, 1], c=y_train, cmap='RdYlBu', s=80, edgecolor='black')
plt.xlabel('Petal Length', fontweight='bold')
plt.ylabel('Petal Width', fontweight='bold')
plt.title('Perceptron (Sigmoid)\nUsing Petal Features Only', fontweight='bold', fontsize=11)
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 3)
plt.plot(x_input1, y_lr, color='green', linewidth=3)
plt.scatter(X_train_petal[:, 0], X_train_petal[:, 1], c=y_train, cmap='RdYlBu', s=80, edgecolor='black')
plt.xlabel('Petal Length', fontweight='bold')
plt.ylabel('Petal Width', fontweight='bold')
plt.title('Logistic Regression\nUsing Petal Features Only', fontweight='bold', fontsize=11)
plt.grid(True, alpha=0.3)

# Row 2: PCA (All 4 Features)
plt.subplot(2, 3, 4)
plt.plot(x_input2, y_step_pca, color='red', linewidth=3)
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='RdYlBu', s=80, edgecolor='black')
plt.xlabel('PCA Component 1', fontweight='bold')
plt.ylabel('PCA Component 2', fontweight='bold')
plt.title('Perceptron (Step)\nUsing ALL 4 Features (PCA)', fontweight='bold', fontsize=11)
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 5)
plt.plot(x_input2, y_sigmoid_pca, color='orange', linewidth=3)
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='RdYlBu', s=80, edgecolor='black')
plt.xlabel('PCA Component 1', fontweight='bold')
plt.ylabel('PCA Component 2', fontweight='bold')
plt.title('Perceptron (Sigmoid)\nUsing ALL 4 Features (PCA)', fontweight='bold', fontsize=11)
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 6)
plt.plot(x_input2, y_lr_pca, color='green', linewidth=3)
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='RdYlBu', s=80, edgecolor='black')
plt.xlabel('PCA Component 1', fontweight='bold')
plt.ylabel('PCA Component 2', fontweight='bold')
plt.title('Logistic Regression\nUsing ALL 4 Features (PCA)', fontweight='bold', fontsize=11)
plt.grid(True, alpha=0.3)

plt.suptitle('Decision Boundary Comparison: Petal Features vs ALL Features (PCA)', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# Print PCA information
print("\n" + "="*60)
print("PCA INFORMATION")
print("="*60)
print(f"PCA Explained Variance Ratio: {pca.explained_variance_ratio_}")
print(f"Total Variance Captured: {sum(pca.explained_variance_ratio_):.2%}")
print("\nFeature Contributions to PCA Components:")
feature_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
for i in range(pca.components_.shape[0]):
    print(f"  PCA Component {i+1}:")
    for j, (name, loading) in enumerate(zip(feature_names, pca.components_[i])):
        print(f"    {name}: {loading:.3f}")