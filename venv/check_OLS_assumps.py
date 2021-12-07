import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.stats.outliers_influence import variance_inflation_factor

def check_OLS(data):
    # check OLS assumptions
    # Linearity
    #f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(15,3))
    #ax1.scatter(data['Year'], data['Price'])
    #ax1.set_title('Price and Year')
    #ax2.scatter(data['EngineV'], data['Price'])
    #ax2.set_title('Price and EngineV')
    #ax3.scatter(data['Mileage'], data['Price'])
    #ax3.set_title('Price and Mileage')

    #plt.show()
    # logarithmize
    log_price = np.log(data['Price'])
    data['log_price'] = log_price

    #f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(15,3))
    #ax1.scatter(data['Year'], data['log_price'])
    #ax1.set_title('Log Price and Year')
    #ax2.scatter(data['EngineV'], data['log_price'])
    #ax2.set_title('Log Price and EngineV')
    #ax3.scatter(data['Mileage'], data['log_price'])
    #ax3.set_title('Log Price and Mileage')
    #plt.show()

    # Multicollinearity

    variables = data[['Mileage', 'Year', 'EngineV']]
    vif = pd.DataFrame()
    vif['VIF'] = [variance_inflation_factor(variables.values,i) for i in range(variables.shape[1])]
    vif['features'] = variables.columns

    #print(vif)

    data_no_col = data.drop(['Year'],axis=1)
    return data_no_col