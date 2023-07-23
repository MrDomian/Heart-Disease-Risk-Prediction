import pandas as pd
from sqlalchemy import create_engine


def export_data_from_database(host, port, database, user, password, csv_file_path, table_name):
    engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{database}')
    query = f'SELECT * FROM {table_name};'
    df = pd.read_sql(query, engine)
    df.to_csv(csv_file_path, index=False)
    print("Data has been exported to the CSV file.")


host = 'localhost'
port = '5432'
database = 'database'
user = 'postgres'
password = 'Damian1999'

table_name = 'heart_disease_risk'
csv_file_path = '../../exported_data.csv'

export_data_from_database(host=host, port=port, database=database, user=user, password=password,
                          csv_file_path=csv_file_path, table_name=table_name)
