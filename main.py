import pandas as pd

from clean_data import clean_data
from check_OLS_assumps import check_OLS
from lin_reg import lin_reg
from statsmodels.stats.outliers_influence import variance_inflation_factor

if __name__ == '__main__':
    data = pd.read_csv('car_data.csv')

    data = clean_data(data)

    data = check_OLS(data)

    data = data.drop(['Price'], axis=1)

    data_dummies = pd.get_dummies(data, drop_first=True)

    vif = pd.DataFrame()
    variables = data_dummies.drop(['log_price'], axis=1)
    vif['VIF'] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
    vif['features'] = variables.columns
    # print(vif)

    # rearrange columns
    price = data_dummies['log_price']
    data_dummies = data_dummies.drop(['log_price'], axis=1)
    data_dummies.insert(0, 'log_price', price)

    # end of preprocessing
    data_preprocessed = data_dummies
    ins = data_preprocessed.drop(['log_price'], axis=1).columns.tolist()
    targets = 'log_price'

    data_model = lin_reg(data_preprocessed, ins, targets)
