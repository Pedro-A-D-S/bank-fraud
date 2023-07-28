from matplotlib import pyplot as plt
import seaborn as sns
import mlflow
from sklearn.metrics import confusion_matrix, roc_auc_score


def train(sk_model, x_train, y_train):
    sk_model = sk_model.fit(x_train, y_train)
    
    train_acc = sk_model.score(x_train, y_train)
    mlflow.log_metric('train_acc', train_acc)
    print(f'Train accuracy: {train_acc:.2%}')

def evaluate(sk_model, x_test, y_test):
    
    eval_acc = sk_model.score(x_test, y_test)
    
    preds = sk_model.predict(x_test)
    auc_score = roc_auc_score(y_test, preds)
    
    mlflow.log_metric('eval_acc', eval_acc)
    mlflow.log_metric('auc_score', auc_score)
    
    print(f'Test accuracy: {eval_acc:.2%}')
    print(f'AUC score: {auc_score:.2%}')
    
    conf_matrix = confusion_matrix(y_test, preds)
    ax = sns.heatmap(conf_matrix, annot = True, fmt = 'g')
    ax.invert_xaxis()
    ax.invert_yaxis()
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('sklearn_conf_matrix.png')
    
    mlflow.log_artifact('sklearn_conf_matrix.png')