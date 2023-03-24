import pandas

def main():

    file_path=r"C:\Users\abdug\Python\DDOinVRP\Data - Conf\Routing_Data_Editable.xlsx"
    routing_dataframe=pandas.read_excel(file_path,sheet_name="Sheet1")
    print(routing_dataframe.shape)

if __name__ == "__main__":
    main()