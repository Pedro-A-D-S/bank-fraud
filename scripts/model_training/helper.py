from matplotlib import pyplot as plt
import seaborn as sns
import mlflow
from sklearn.metrics import confusion_matrix, roc_auc_score
from typing import Any


def train(sk_model: Any, x_train: Any, y_train: Any) -> None:
    """
    Train a scikit-learn model, log train accuracy.
    
    Args:
        sk_model (Any): Scikit-learn model object.
        x_train (Any): Training input data.
        y_train (Any): Training target data.
    """
    sk_model = sk_model.fit(x_train, y_train)
    
    train_acc = sk_model.score(x_train, y_train)
    mlflow.log_metric('train_acc', train_acc)
    print(f'Train accuracy: {train_acc:.2%}')

def evaluate_and_plot(sk_model: Any, x_test: Any, y_test: Any) -> None:
    """
    Evaluate a scikit-learn model, log evaluation metrics, and plot confusion matrix.
    
    Args:
        sk_model (Any): Scikit-learn model object.
        x_test (Any): Testing input data.
        y_test (Any): Testing target data.
    """
    eval_acc = sk_model.score(x_test, y_test)
    preds = sk_model.predict(x_test)
    auc_score = roc_auc_score(y_test, preds)
    
    mlflow.log_metric('eval_acc', eval_acc)
    mlflow.log_metric('auc_score', auc_score)
    
    print(f'Test accuracy: {eval_acc:.2%}')
    print(f'AUC score: {auc_score:.2%}')
    
    conf_matrix = confusion_matrix(y_test, preds)
    plot_confusion_matrix(conf_matrix)
    save_confusion_matrix_plot()

def plot_confusion_matrix(conf_matrix: Any) -> None:
    """
    Plot a confusion matrix using seaborn.
    
    Args:
        conf_matrix (Any): Confusion matrix data.
    """
    ax = sns.heatmap(conf_matrix, annot=True, fmt='g')
    ax.invert_xaxis()
    ax.invert_yaxis()
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
def save_confusion_matrix_plot() -> None:
    """
    Save the confusion matrix plot as an artifact.
    """
    plt.savefig('sklearn_conf_matrix.png')
    mlflow.log_artifact('sklearn_conf_matrix.png')

def save_model(sk_model: Any) -> None:
    """
    Save a scikit-learn model as an artifact.
    
    Args:
        sk_model (Any): Scikit-learn model object.
    """
    mlflow.sklearn.log_model(sk_model, 'model')
    
def log_params(params: dict) -> None:
    """
    Log a dictionary of parameters.
    
    Args:
        params (dict): Dictionary of parameters.
    """
    for k, v in params.items():
        mlflow.log_param(k, v)

def log_model_params(sk_model: Any) -> None:
    """
    Log a scikit-learn model's parameters.
    
    Args:
        sk_model (Any): Scikit-learn model object.
    """
    for k, v in sk_model.get_params().items():
        mlflow.log_param(k, v)
    
def log_requirements(requirements_file: str) -> None:
    """
    Log a requirements.txt file as an artifact.
    
    Args:
        requirements_file (str): Path to requirements.txt file.
    """
    mlflow.log_artifact(requirements_file)

def log_data(data: Any, data_name: str) -> None:
    """
    Log a pandas DataFrame as an artifact.
    
    Args:
        data (Any): Pandas DataFrame.
        data_name (str): Name of the DataFrame.
    """
    data.to_csv(f'{data_name}.csv', index=False)
    mlflow.log_artifact(f'{data_name}.csv')