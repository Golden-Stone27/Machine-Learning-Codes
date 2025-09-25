from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import pandas as pd

# (1) Veri seti incelenmesi
cancer = load_breast_cancer()
df = pd.DataFrame(data=cancer.data, columns= cancer.feature_names)
df["target"] = cancer.target

# (2) Makine Öğrenmesi Modelinin Seçilmesi -KNN Sınıflandırıcı
# (3) Modelin Train Edilmesi

X = cancer.data #features
y = cancer.target # target

#train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

# ölçeklendirme
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) # Train verisi fit edilip scale parametreleri öğrenilir, test verisi bu parametrelerle transform edilir ve KNN buna göre tahmin yapar
X_test = scaler.transform(X_test)

#KNN modeli oluştur ve train et
knn = KNeighborsClassifier() # Modelin oluşturulma komşu parametresini unutma ********
knn.fit(X_train, y_train) # fit fonksiyonu verimizi (sample + target) kullanarak knn algoritmasını eğitir

# (4) Sonuçların değerlendirilmesi: Test

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("accuracy:", accuracy)

conf_matrix = confusion_matrix(y_test, y_pred)
print("confusion_matrix: ")
print(conf_matrix)


# (5) Hiperparametre Ayarlaması
'''
KNN: Hyperparameter = K K: 1,2,3 N
Accuracy: %A, %B, %C....
'''

accuracy_values = []
k_values = []

for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_values.append(accuracy)
    k_values.append(k)

plt.figure()
plt.plot(k_values, accuracy_values, marker = "o", linestyle = "-") # marker noktaların ne olduğu linestyle noktalar arası çizginin ne olduğu
plt.title("K degerine gore dogruluk")
plt.xlabel("K degeri")
plt.ylabel("Dogruluk")
plt.xticks(k_values)
plt.grid(True)
plt.show()