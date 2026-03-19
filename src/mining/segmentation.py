import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

plt.rcParams["font.family"] = "DejaVu Sans"


def customer_segmentation(df, output_dir="outputs/figures"):

    features = ["age", "balance", "campaign", "duration"]

    X = df[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=42)

    clusters = kmeans.fit_predict(X_scaled)

    df["Cụm khách hàng"] = clusters

    cluster_summary = df.groupby("Cụm khách hàng")[features].mean()

    print("\nTóm tắt các cụm khách hàng:")
    print(cluster_summary)

    plt.figure(figsize=(6,5))

    sns.scatterplot(
        x=df["age"],
        y=df["balance"],
        hue=df["Cụm khách hàng"],
        palette="Set2"
    )
    plt.legend(title="Cụm khách hàng")
    
    plt.title("Phân cụm khách hàng ngân hàng")

    plt.xlabel("Tuổi")

    plt.ylabel("Số dư tài khoản")

    plt.savefig(f"{output_dir}/customer_clusters.png", dpi=300)

    plt.close()

    return df