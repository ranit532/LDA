
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import joblib

def evaluate_model(data_path="c:/Users/2185206/LDA/data/lda_data.csv", model_path="c:/Users/2185206/LDA/src/lda_model.pkl"):
    """
    Evaluates the trained LDA model and generates plots.

    Args:
        data_path (str): The path to the dataset.
        model_path (str): The path to the trained model.
    """
    # Load data
    df = pd.read_csv(data_path)
    X = df.drop("target", axis=1)
    y = df["target"]

    # Split data to get the same test set as in training
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Load the model
    lda = joblib.load(model_path)

    # Transform the test data
    X_lda = lda.transform(X_test)

    # Create a scatter plot of the transformed data
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y_test, cmap="viridis", edgecolor="k")
    plt.title("LDA: Transformed Data")
    plt.xlabel("LD1")
    plt.ylabel("LD2")
    plt.legend(handles=scatter.legend_elements()[0], labels=[str(c) for c in lda.classes_])
    plt.grid(True)
    plt.savefig("c:/Users/2185206/LDA/images/lda_plot.png")
    print("LDA plot saved to images/lda_plot.png")

    # Create a confusion matrix plot
    y_pred = lda.predict(X_test)
    cm = pd.crosstab(y_test, y_pred, rownames=["Actual"], colnames=["Predicted"])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig("c:/Users/2185206/LDA/images/confusion_matrix.png")
    print("Confusion matrix plot saved to images/confusion_matrix.png")

if __name__ == "__main__":
    evaluate_model()
