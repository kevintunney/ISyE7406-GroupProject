	 		   		 		  
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import plotly as plotly


# to save image, follow the instructions here https://stackoverflow.com/questions/59815797/how-to-save-plotly-express-plot-into-a-html-or-static-image-file

def main():
    test = pd.read_csv("./data/journeys.csv", index_col='start_date', parse_dates=True, na_values=['nan']) 
    # print(test.columns.values);
    # print(test)

    # touch = test["touchpoint_count"].hist(by=test['conversion'],figsize=(10, 8))
    # plt.savefig('touch.png')


    # https://plotly.com/python/histograms/

    # fig = px.bar(test["age_group"].value_counts(), title="Age of users")
    # fig.update_layout(
    #     xaxis_title = "Age Group",
    #     yaxis_title = "Frequency",
    #     title_x = 0.5,
    #     showlegend = False
    # )

    # fig.write_image("histogram.jpg")
    # plotly.offline.plot(fig, filename='histogram.html')




if __name__ == "__main__":  		  	   		   	 		  		  		    	 		 		   		 		  
    main()  