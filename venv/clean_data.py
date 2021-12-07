
import seaborn as sns
import matplotlib.pyplot as plt

def clean_data(df):
    df.describe(include='all')

    # drop model because too many values, drop registration because useless
    data_1 = df.drop(['Model'],axis=1)

    data_2 = data_1.dropna(axis=0) # small amount of missing values, so we remove
    #sns.histplot(data_2['Price'])
    #plt.show()
    #remove top 1% of outliers
    q_price = data_2['Price'].quantile(0.99)
    data_3 = data_2[data_2['Price'] < q_price]
    #sns.histplot(data_3['Price'])
    #plt.show()

    #sns.histplot(data_3['Mileage'])
    #plt.show()

    q_mileage = data_3['Mileage'].quantile(0.99)
    data_4 = data_3[data_3['Mileage'] < q_mileage]

    #sns.histplot(data_4['Mileage'])
    #plt.show()

    #sns.histplot(data_4['EngineV'])

    #only keep valid values
    data_5 = data_4[data_4['EngineV'] < 6.5]

    #sns.histplot(data_5['EngineV'])

    #sns.histplot(data_5['Year'])
    #plt.show()

    # remove bottom 1%
    q_year = data_5['Year'].quantile(0.01)
    data_6 = data_5[data_5['Year'] > q_year]

    #sns.histplot(data_6['Year'])
    #plt.show()

    data_clean = data_6.reset_index(drop=True)
    #print("We removed %4.2f%% of data." % (((df.size-data_clean.size)/df.size)*100))
    return data_clean