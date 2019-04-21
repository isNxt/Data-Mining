import pandas as pd


def read_data(file_path):
    df = pd.read_csv(file_path, header=None, low_memory=False)
    y = df.iloc[0].T
    X = df.iloc[1:].T
    # print('X:\n', X, 'y:\n', y)
    return X, y


if __name__ == '__main__':
    X_train, y_train = read_data('./data/GCM_train.data')
    X_test, y_test = read_data('./data/GCM_test.data')
