import pandas
import simplejson, urllib.request
import numpy
import sys
import pickle
import torch


def main():
    file_path = r"C:\Users\abdug\Python\DDOinVRP\DB Schenker\Data - Conf\\Routing_Data_Editable_ver05.xlsx"
    routing_data = pandas.read_excel(file_path, sheet_name="Sheet4")
    unique_dates = routing_data['Trip Date'].unique()
    KEY = "AIzaSyChAruzw5_-bIhd4MiD2e9ywGZ0fXDT6bI"

    for date in unique_dates:
        daily_instance = routing_data[routing_data['Trip Date'] == date]
        postcodes = daily_instance['Customer Postcode ']
        N = len(daily_instance)
        travel_times = torch.zeros((N, N), dtype=torch.float32)
        for i in range(N):
            for j in range(N):
                if i != j:
                    orig = str(postcodes.iloc[i]) + " Netherlands"
                    dest = str(postcodes.iloc[j]) + " Netherlands"
                    orig = orig.replace(" ", "+")
                    dest = dest.replace(" ", "+")
                    url = "https://maps.googleapis.com/maps/api/distancematrix/json?key={0}&origins={1}&destinations={" \
                          "2}&mode=driving&language=en-EN&sensor=false".format(KEY, str(orig), str(dest))
                    try:
                        result = simplejson.load(urllib.request.urlopen(url))
                        driving_time = result['rows'][0]['elements'][0]['duration']['text']
                        print(driving_time)
                        result = [float(x) for x in driving_time.split() if x.isdigit()]
                        if len(result) == 1:
                            travel_time = result[0] / 60
                        elif len(result) == 2:
                            travel_time = result[0] + result[1] / 60
                        else:
                            print("Stange Error")
                            sys.exit(0)
                        travel_times[i, j] = travel_time
                    except:
                        continue

    # routing_data.to_excel(r"C:\Users\abdug\Python\DDOinVRP\Data - Conf\Routing_Data_Editable_ver05.xlsx","Sheet4")
    '''pickle_out = open('DB Schenker Problem instances', 'wb')
    pickle.dump(routing_data, pickle_out)
    pickle_out.close()'''


if __name__ == "__main__":
    main()
