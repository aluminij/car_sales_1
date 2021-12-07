from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def lin_reg(df,ins,targets):
    # regression
    target = df[targets]

    ### TODO: make custom scaler to skip dummy variables
    inputs = df[ins]

    # scale

    scaler = StandardScaler()
    scaler.fit(inputs)
    inputs_scaled = scaler.transform(inputs)

    x_train, x_test, y_train, y_test = train_test_split(inputs_scaled, target, test_size=0.2, random_state=365)

    reg = LinearRegression()
    reg.fit(x_train, y_train)

    y_hat = reg.predict(x_train)
    # plt.scatter(y_train,y_hat)
    # plt.xlabel('Targets (y_train)', size=18)
    # plt.ylabel('Predictions (y_hat)', size=18)
    # plt.xlim(6,13)
    # plt.ylim(6,13)
    # plt.show()
    #
    # sns.displot(y_train - y_hat)
    # plt.title('Residuals PDF', size=18)
    # plt.show()

    print(reg.score(x_train, y_train))

    y_hat_test = reg.predict(x_test)
    plt.scatter(y_test, y_hat_test, alpha=0.2)
    plt.xlabel('Targets (y_test)', size=18)
    plt.ylabel('Predictions (y_hat_test)', size=18)
    plt.xlim(6, 13)
    plt.ylim(6, 13)
    # plt.show()

    y_test = y_test.reset_index(drop=True)

    df_pf = pd.DataFrame(np.exp(y_hat_test), columns=['Prediction'])
    df_pf['Target'] = np.exp(y_test)
    df_pf['Residual'] = df_pf['Target'] - df_pf['Prediction']
    df_pf['Difference%'] = np.absolute(df_pf['Residual'] / df_pf['Target'] * 100)
    print(df_pf)
    print(df_pf.describe())