import pandas as pd

df = pd.read_csv("training_concat.csv")

print("Training")
print(df.max())

df = pd.read_csv("inference_concat.csv")

print("Inference")
print(df.max())

