# Importer la bibliothèque scikit-learn
import sklearn

# Charger l'ensemble de données iris
from sklearn.datasets import load_iris
iris = load_iris()

# Afficher les noms des caractéristiques et des cibles
print(iris.feature_names)
print(iris.target_names)

# Créer les variables X et y qui contiennent les caractéristiques et les cibles
X = iris.data
y = iris.target
print(x)

# Diviser les données en ensemble d'entraînement et de test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer le modèle k-NN avec k=3
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)

# Entraîner le modèle sur les données d'entraînement
knn.fit(X_train, y_train)

# Prédire les cibles sur les données de test
y_pred = knn.predict(X_test)

# Évaluer la précision du modèle
from sklearn.metrics import accuracy_score
print("La précision du modèle est de :", accuracy_score(y_test, y_pred))
