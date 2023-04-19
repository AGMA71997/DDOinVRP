import pandas
import simplejson, urllib.request
import numpy
import sys

def main():
    file_path = r"C:\Users\abdug\Python\DDOinVRP\Data - Conf\Routing_Data_Editable.xlsx"
    routing_data = pandas.read_excel(file_path, sheet_name="Sheet3")
    postcodes = routing_data['Customer Postcode ']
    travel_times=numpy.zeros(len(postcodes))
    KEY = "AIzaSyDeDZcPocz-Cr6kR8XrYi28IK6B53ZdJyQ"

    for i in range(len(postcodes)):
        if i>0 and routing_data['Trip'].iloc[i]==routing_data['Trip'].iloc[i-1]:
            orig = str(postcodes.iloc[i-1]) + " Netherlands"
            dest = str(postcodes.iloc[i]) + " Netherlands"
            orig=orig.replace(" ","+")
            dest=dest.replace(" ", "+")
            url = "https://maps.googleapis.com/maps/api/distancematrix/json?key={0}&origins={1}&destinations={" \
                "2}&mode=driving&language=en-EN&sensor=false".format(KEY, str(orig), str(dest))
            result = simplejson.load(urllib.request.urlopen(url))
            try:
                driving_time = result['rows'][0]['elements'][0]['duration']['text']
                result = [float(x) for x in driving_time.split() if x.isdigit()]
                if len(result)==1:
                    travel_times[i]=result[0]/60
                elif len(result)==2:
                    travel_times[i]=result[0]+result[1]/60
                else:
                    print("Stange Error")
                    sys.exit(0)
            except:
                print(result)
                continue

            print(driving_time)
            print(result)
            print("-----")

    print(travel_times[0:10])


if __name__ == "__main__":
    main()
