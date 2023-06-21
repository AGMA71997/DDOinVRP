import pandas
import simplejson, urllib.request
import numpy
import sys
import pickle

def main():
    file_path = r"C:\Users\abdug\Python\DDOinVRP\Data - Conf\\Routing_Data_Editable_ver05.xlsx"
    routing_data = pandas.read_excel(file_path, sheet_name="Sheet4")
    postcodes = routing_data['Customer Postcode ']
    travel_times=numpy.zeros(len(postcodes))
    KEY = "AIzaSyDeDZcPocz-Cr6kR8XrYi28IK6B53ZdJyQ"

    for i in range(len(postcodes)):
        print(routing_data['Event time'].iloc[i])
        print(i)
        print("--------")
        if i>0 and routing_data['Trip'].iloc[i]==routing_data['Trip'].iloc[i-1] and not pandas.isnull(routing_data['Event time'].iloc[i]):
            orig = str(postcodes.iloc[i-1]) + " Netherlands"
            dest = str(postcodes.iloc[i]) + " Netherlands"
            orig=orig.replace(" ","+")
            dest=dest.replace(" ", "+")
            url = "https://maps.googleapis.com/maps/api/distancematrix/json?key={0}&origins={1}&destinations={" \
                "2}&mode=driving&language=en-EN&sensor=false".format(KEY, str(orig), str(dest))
            try:
                result = simplejson.load(urllib.request.urlopen(url))
                driving_time = result['rows'][0]['elements'][0]['duration']['text']
                print(driving_time)
                result = [float(x) for x in driving_time.split() if x.isdigit()]
                if len(result)==1:
                    travel_times[i]=result[0]/60
                elif len(result)==2:
                    travel_times[i]=result[0]+result[1]/60
                else:
                    print("Stange Error")
                    sys.exit(0)
            except:
                continue

    routing_data['Travel Time']=travel_times

    routing_data.to_excel(r"C:\Users\abdug\Python\DDOinVRP\Data - Conf\Routing_Data_Editable_ver05.xlsx","Sheet4")
    pickle_out = open('DB Schenker dataframe', 'wb')
    pickle.dump(routing_data, pickle_out)
    pickle_out.close()


if __name__ == "__main__":
    main()
