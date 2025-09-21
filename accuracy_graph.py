import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("runs/classify/train/results.csv")
print(df.columns)  # see available metrics

plt.plot(df["epoch"], df["metrics/accuracy_top1"], label="Top-1 Accuracy")
plt.plot(df["epoch"], df["metrics/accuracy_top5"], label="Top-5 Accuracy")
plt.plot(df["epoch"], df["train/loss"], label="Train Loss")
plt.plot(df["epoch"], df["val/loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.legend()
plt.show()