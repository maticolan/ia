import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("learning_curve.csv")

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(data["epoch"], data["error"], marker='o')
plt.title("Curva de Error")
plt.xlabel("Época")
plt.ylabel("Error Promedio")

plt.subplot(1, 2, 2)
plt.plot(data["epoch"], data["accuracy"], marker='o', color='green')
plt.title("Curva de Accuracy")
plt.xlabel("Época")
plt.ylabel("Accuracy (%)")

plt.tight_layout()
plt.savefig("learning_curve.png")
plt.show()
