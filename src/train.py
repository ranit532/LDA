
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix
import mlflow
import mlflow.sklearn
import joblib

def train_model(data_path="c:/Users/2185206/LDA/data/lda_data.csv"):
    """
    Trains an LDA model and logs the experiment with MLflow.

    Args:
        data_path (str): The path to the dataset.
    """
    # Load data
    df = pd.read_csv(data_path)
    X = df.drop("target", axis=1)
    y = df["target"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Set MLflow experiment
    mlflow.set_experiment("LDA_Classification")

    # Start MLflow run
    with mlflow.start_run():
        # Initialize and train the model
        lda = LinearDiscriminantAnalysis(n_components=2)
        lda.fit(X_train, y_train)

        # Make predictions
        y_pred = lda.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        # Log parameters and metrics
        mlflow.log_param("n_components", 2)
        mlflow.log_metric("accuracy", accuracy)

        # Log the model
        mlflow.sklearn.log_model(lda, "lda_model")

        # Save the model
        joblib.dump(lda, "c:/Users/2185206/LDA/src/lda_model.pkl")

        print(f"Accuracy: {accuracy}")
        print("Model trained and saved to src/lda_model.pkl")

if __name__ == "__main__":
    train_model()
