# Logistic Regression from Scratch

Figuring out how logistic regression works: implemented it in NumPy, no `sklearn.linear_model`.

## Core idea

- Linear model: `z = θ₀ + θ₁x₁ + … + θₙxₙ`
- Sigmoid maps `z` to a probability: `σ(z) = 1 / (1 + e^(-z))`
- Threshold 0.5: `p ≥ 0.5` → class 1, else 0
- Loss: log-loss (binary cross-entropy)
- Gradient: `∇J = Xᵀ(σ(Xθ) − y) / m`
- Training: gradient descent `θ := θ − α·∇J`, stops on `tol` or `num_iters`

## Code

| Function | What it does |
|---|---|
| `sigmoid` | σ(z) |
| `calculate_gradient` | log-loss gradient |
| `gradient_descent` | adds bias column, learns θ |
| `predict_proba` | probabilities |
| `predict` | labels by threshold |

Dataset: `load_breast_cancer` (binary classification), 80/20 split, features scaled with `StandardScaler`. Metric: accuracy on train and test.

## Run

```bash
pip install numpy scikit-learn
python main.py
```

Output: `train_accuracy test_accuracy`.
