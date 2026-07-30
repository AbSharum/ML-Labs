# Machine Learning Labs

Lab notebooks from a Machine Learning course, progressing from basic Python/OOP warm-ups to trained classifiers and a PyTorch neural network.

## Contents

- `Lab 1/Lab1Sharum.ipynb` — Introductory Python/OOP exercise (a `Customer` class with callable behavior for updating credit limits/orders); no ML libraries used.
- `Lab 2/Classifier.ipynb` — KNN classifier trained on the Iris dataset (`iris/iris.data`) using scikit-learn (`train_test_split`, `KFold` cross-validation, `confusion_matrix`); the trained model is pickled to `knn.pkl`.
- `Lab 3/knn.ipynb` — A second scikit-learn KNN model (regression/classification with `KFold`, `confusion_matrix`, `mean_absolute_error`) trained on `Market.csv`/`ps3data.csv`, saved to `knn.pkl`.
- `Lab 3/nn.ipynb` — A PyTorch feedforward neural network (`torch`, `torch.nn`, `MinMaxScaler` preprocessing) trained on the same market data; the trained model is saved to `model.pth`.

## Stack

Python, Jupyter Notebook, `scikit-learn`, `pandas`, `numpy`, `pickle`, `torch` (PyTorch).

## Running

```
pip install jupyter scikit-learn pandas numpy torch
jupyter notebook
```

Then open any `.ipynb` file. Pretrained artifacts (`knn.pkl` in Lab 2 and Lab 3, `model.pth` in Lab 3) are included so notebooks can load saved models instead of retraining from scratch.
