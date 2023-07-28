import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
import mlflow
import mlflow.sklearn
import os

# Set up
logging.basicConfig(filename = '../../logs/data_prep_training.log', level = logging.INFO,
                    format = '%(asctime)s - %(levelname)s - %(message)s')

# MLFlow folder tracking
mlflow_tracking_uri = '../../data/mlflow/mlruns'

# Set the environment variable
os.environ['MLFLOW_TRACKING_URI'] = mlflow_tracking_uri

# functions
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
    ax = sns.heatmap(conf_matrix, annot=True, fmt='g')
    ax.invert_xaxis()
    ax.invert_yaxis()
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('sklearn_conf_matrix.png')
    
    mlflow.log_artifact('sklearn_conf_matrix.png')

# Log the start of the script
logging.info('Data preprocessing script started.')

# Load data
data_path = '../../data/raw/archive/creditcard.csv'
df = pd.read_csv(data_path)
logging.info('Data file successfully loaded.')

# Split data into normal and anomaly samples
normal = df[df['Class'] == 0].sample(frac=0.5, random_state=2020).reset_index(drop=True)
anomaly = df[df['Class'] == 1]

# Split data into train, test, and validation sets
normal_train, normal_test = train_test_split(normal, test_size=0.2, random_state=2020)
anomaly_train, anomaly_test = train_test_split(anomaly, test_size=0.2, random_state=2020)
normal_train, normal_validate = train_test_split(normal_train, test_size=0.25, random_state=2020)
anomaly_train, anomaly_validate = train_test_split(anomaly_train, test_size=0.25, random_state=2020)

# Concatenate the train, test, and validation sets
x_train = pd.concat((normal_train, anomaly_train))
x_test = pd.concat((normal_test, anomaly_test))
x_validate = pd.concat((normal_validate, anomaly_validate))

# Extract labels
y_train = np.array(x_train['Class'])
y_test = np.array(x_test['Class'])
y_validate = np.array(x_validate['Class'])

# Remove labels from the features
x_train = x_train.drop('Class', axis=1)
x_test = x_test.drop('Class', axis=1)
x_validate = x_validate.drop('Class', axis=1)

logging.info('Data splitted.')

# Scale the data
scaler = StandardScaler()
scaler.fit(pd.concat((normal, anomaly)).drop('Class', axis=1))
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_validate = scaler.transform(x_validate)
logging.info('Data scaled.')

# Train the logistic regression model
sk_model = LogisticRegression(random_state = None, max_iter = 400,
                              solver = 'newton-cg').fit(x_train, y_train)

# Evaluate the model on the test set
eval_acc = sk_model.score(x_test, y_test)

# Make predictions and calculate AUC score
preds = sk_model.predict(x_test)
auc_score = roc_auc_score(y_test, preds)

# Model Validation
anomaly_weights = [1, 5, 10, 15]

# Set up MLflow experiment
mlflow.set_experiment('credit_card')
logs = []

# Cross-validation
num_folds = 5
kfold = KFold(n_splits = num_folds, shuffle = True, random_state = 2020)

logging.info('Cross-validation starting...')
for f in range(len(anomaly_weights)):
    fold = 1
    accuracies = []
    auc_scores = []
    for train_idx, test_idx in kfold.split(x_validate, y_validate):
        with mlflow.start_run():
            weight = anomaly_weights[f]
            mlflow.log_param('anomaly_weight', weight)
            class_weights = {0: 1, 1: weight}
            sk_model = LogisticRegression(random_state=None, max_iter=400, solver='newton-cg',
                                          class_weight=class_weights).fit(x_validate[train_idx], y_validate[train_idx])

            print(f"\nfold {fold}\nAnomaly Weight: {weight}")
            
            # Train the model on the training set
            train(sk_model, x_validate[train_idx], y_validate[train_idx])
            
            # Evaluate the model on the validation set
            evaluate(sk_model, x_validate[test_idx], y_validate[test_idx])
            
            accuracies.append(eval_acc)
            auc_scores.append(auc_score)

            log = [sk_model, x_validate[test_idx], y_validate[test_idx], preds]
            logs.append(log)
            
            # Log the model using MLflow
            mlflow.sklearn.log_model(sk_model, f'anom_weight_{weight}_fold_{fold}')

            fold = fold + 1
            mlflow.end_run()

    print('\nAverages: ')
    print('Accuracy: ', np.mean(accuracies))
    print('AUC: ', np.mean(auc_scores))

    print('Best: ')
    print('Accuracy: ', np.max(accuracies))
    print('AUC: ', np.max(auc_scores))
