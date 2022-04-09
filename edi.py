	 		   		 		  
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import plotly as plotly
import seaborn as sns
from dython import nominal
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split




# to save image, follow the instructions here https://stackoverflow.com/questions/59815797/how-to-save-plotly-express-plot-into-a-html-or-static-image-file

def runKNN(df):
    labels = df.conversion
    X = df.loc[:, df.columns != 'conversion']
    y = df.conversion

    x_train, x_test, y_train, y_test = train_test_split(df,labels, test_size=0.2, random_state=123)
    knn = KNeighborsClassifier(n_neighbors=30)
    knn.fit(x_train, y_train)


    x_min, x_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
    y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])




def main():
    df = pd.read_csv("./data/journeys.csv", index_col='id', parse_dates=True, na_values=['nan']) 

    # dimensions
    # print(df.shape)
    # print(df.columns.values);


    # # correlation matrix
    # corr = df[['age', 'language', 'journey', 'touchpoint','duration', 'conversion','email', 'facebook', 'house_ads', 'instagram', 'push']]
    # nominal.associations(corr,figsize=(10,10),mark_columns=False, filename='correlationMatrix.jpg');

    # # EDI
    # for each in corr.columns.values:
    #     unique = corr[each].unique()
    #     print(sorted(unique))


    # # Histograms
    # fig = px.bar(df["age"].value_counts(), title="Age of users")
    # fig.update_layout(
    #     xaxis_title = "Age Group",
    #     yaxis_title = "Frequency",
    #     title_x = 0.5,
    #     showlegend = False
    # )

    # fig.write_image("age_group_users_histogram.jpg")
    # plotly.offline.plot(fig, filename='age_group_users_histogram.html')

    # ax = df['conversion'].astype(int).hist(bins=2, figsize=(12,8), color='#86bf91')
    # plt.savefig('conversion_histogram.jpg')





if __name__ == "__main__":  		  	   		   	 		  		  		    	 		 		   		 		  
    main()  