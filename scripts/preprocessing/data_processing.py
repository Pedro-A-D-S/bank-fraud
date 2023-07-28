#
import pandas as pd
import numpy as np
import logging

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set up logging
logging.basicConfig(filename = '../../logs/data_preprocessing.log', level = logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Log the start of the script
logging.info('Data preprocessing script started.')

#
data_path = '../../data/raw/archive/creditcard.csv'

#
df = pd.read_csv(data_path)
logging.info('Data file successfully loaded.')

#
normal = df[df['Class'] == 0].sample(frac = 0.5,
                                    random_state = 2020).reset_index(drop = True)
anomaly = df[df['Class'] == 1]

#
normal_train, normal_test = train_test_split(normal, test_size = 0.2,
                                             random_state = 2020)
anomaly_train, anomaly_test = train_test_split(anomaly, test_size = 0.2,
                                              random_state = 2020)

#
normal_train, normal_validate = train_test_split(normal_train, test_size = 0.25,
                                                 random_state = 2020)
anomaly_train, anomaly_validate = train_test_split(anomaly_train, test_size = 0.25,
                                                  random_state = 2020)

#
x_train = pd.concat((normal_train, anomaly_train))
x_test = pd.concat((normal_test, anomaly_test))
x_validate = pd.concat((normal_validate, anomaly_validate))

#
y_train = np.array(x_train['Class'])
y_test = np.array(x_test['Class'])
y_validate = np.array(x_validate['Class'])

#
x_train = x_train.drop('Class', axis = 1)
x_test = x_test.drop('Class', axis = 1)
x_validate = x_validate.drop('Class', axis = 1)

logging.info('Data splitted.')

#
scaler = StandardScaler()
scaler.fit(pd.concat((normal, anomaly)).drop('Class', axis = 1))
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_validate = scaler.transform(x_validate)
logging.info('Data scaled.')

#
columns = [['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
            'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
            'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']]

# Convert NumPy arrays to pandas DataFrames
x_train_df = pd.DataFrame(x_train, columns = columns)
x_test_df = pd.DataFrame(x_test, columns = columns)
x_validate_df = pd.DataFrame(x_validate, columns = columns)
y_train_df = pd.DataFrame(y_train, columns=['Class'])
y_test_df = pd.DataFrame(y_test, columns=['Class'])
y_validate_df = pd.DataFrame(y_validate, columns=['Class'])
logging.info('Data converted to pandas DataFrames.')

# Save pandas DataFrames as CSV files
x_train_df.to_csv('../../data/processed/x_train.csv', index=False)
x_test_df.to_csv('../../data/processed/x_test.csv', index=False)
x_validate_df.to_csv('../../data/processed/x_validate.csv', index=False)
y_train_df.to_csv('../../data/processed/y_train.csv', index=False)
y_test_df.to_csv('../../data/processed/y_test.csv', index=False)
y_validate_df.to_csv('../../data/processed/y_validate.csv', index=False)
logging.info('Data saved as CSV files.')
