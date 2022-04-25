import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def quantile_cutter(df_in,col,upper,lower,verbose=False):
    df_new = df_in.copy()
    df_new[col] = pd.to_numeric(df_new[col])
    df_new = df_new[df_new[col]<df_new[col].quantile(upper)]
    df_new = df_new[df_new[col] >df_new[col].quantile(lower)]
    if verbose:
        df_new.hist(column=col, bins =100)
        plt.xlabel(f"{col}")
        plt.show()
    return df_new

# From https://towardsdatascience.com/
# hyperparameter-tuning-the-random-forest-in-python-
# using-scikit-learn-28d2aa77dd74
def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy