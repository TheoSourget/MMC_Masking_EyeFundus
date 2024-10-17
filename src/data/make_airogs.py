import pandas as pd
import glob
df_base = pd.read_csv("./data/raw/airogs/train_labels.csv")
images = [p.removeprefix("./data/processed/airogs/images/").removesuffix(".jpg") for p in glob.glob("./data/processed/airogs/images/*.jpg")]
df_filtered = df_base[df_base["challenge_id"].isin(images)]
df_filtered["Onehot"] = df_filtered["class"].apply(lambda x:[0] if x=="NRG" else [1])
df_filtered.to_csv("./data/processed/airogs/processed_labels.csv")