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