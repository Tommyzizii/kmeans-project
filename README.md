# 🧠 Machine Learning Lab — K-Means clustering

This repository contains a hands-on data science lab that applies **unsupervised learning** techniques, mainly **K-Means clustering**, on multiple classic datasets including **Iris**, **Wine**, **Seeds**, and **Fish**.  
It demonstrates the end-to-end process of preprocessing, model training, visualization, and interpretation using Python and Scikit-learn.

---

## 📂 Datasets

| File | Description |
|------|--------------|
| `wine.data` / `wine.txt` | UCI Wine dataset — chemical analysis of wines from three cultivars. |
| `seeds.csv` | Wheat seed dataset for cluster analysis and classification. |
| `fish.csv` | Fish species data with body-dimension features. |
| `points.csv`, `new_points.csv` | 2D coordinate data used to visualize K-Means clustering. |

---

## ⚙️ Technologies & Libraries

- **Python 3.10+**
- **NumPy**, **Pandas**
- **Matplotlib**, **Seaborn**
- **Scikit-learn** (K-Means, metrics)
- **Jupyter Notebook**

---

## 🚀 Project Objectives

- Perform **data preprocessing** (cleaning, normalization, feature extraction).  
- Apply **K-Means clustering** to identify groups in datasets.  
- Visualize clusters in 2D space.  
- Predict cluster membership for new unseen points.  
- Evaluate and interpret results through visual analysis.

---

## 🧩 Example Workflow

```python
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns

iris = load_iris()
model = KMeans(n_clusters=3, random_state=42)
model.fit(iris.data)
labels = model.predict(iris.data)

plt.scatter(iris.data[:,0], iris.data[:,2], c=labels, cmap="viridis")
plt.title("K-Means Clustering on Iris Dataset")
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.show()
