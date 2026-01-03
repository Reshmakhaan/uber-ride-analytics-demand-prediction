import pandas as pd

def load_and_clean_data(csv_path):
    data = pd.read_csv(csv_path)

    # Convert datetime
    data['Date/Time'] = pd.to_datetime(data['Date/Time'])

    # Remove duplicates
    data = data.drop_duplicates()

    # Remove invalid coordinates
    data = data[(data['Lat'] != 0) & (data['Lon'] != 0)]

    return data
