import pandas as pd
import numpy as np


def load_response_matrix(path='data/merged_label.csv'):
    df = pd.read_csv(path)

    models_name = [df.columns[i] for i in range(1, len(df.columns) - 1)]
    data = []
    label_value = df[df.columns[-1]].values

    for model in models_name:
        model_predict_value = df[model].values
        relative_error = list(abs(model_predict_value - label_value))
        data.append(relative_error)

    data = np.array(data)
    response_matrix = data / data.max()

    rows, cols = np.where(response_matrix >= 0)
    responses = response_matrix[rows, cols]
    response_tuple = [(row, col, response) for (row, col, response) in zip(rows, cols, responses)]