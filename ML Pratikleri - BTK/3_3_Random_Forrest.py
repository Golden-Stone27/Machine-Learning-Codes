from sklearn.datasets import fetch_olivetti_faces
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


oli = fetch_olivetti_faces()

'''
    2D (64x64) -> 1D (4096)
'''

plt.figure()
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.imshow(oli.images[i], cmap='gray')
    plt.axis('off')
plt.show()



X = oli.data
y = oli.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


accuracy_values = []
k_values = []


for k in range(200, 1001, 200):
    rf_clf = RandomForestClassifier(n_estimators=k, random_state=42)
    rf_clf.fit(X_train, y_train)
    y_pred = rf_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("accuracy: ", accuracy)
    accuracy_values.append(accuracy)
    k_values.append(k)
    print(confusion_matrix(y_test, y_pred))


plt.figure()
plt.plot(k_values, accuracy_values, marker = "o", linestyle = "-") # marker noktaların ne olduğu linestyle noktalar arası çizginin ne olduğu
plt.title("K degerine gore dogruluk")
plt.xlabel("K degeri")
plt.ylabel("Dogruluk")
plt.xticks(k_values)
plt.grid(True)
plt.show()








