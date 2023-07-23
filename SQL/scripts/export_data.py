import pandas as pd
from sqlalchemy import create_engine


def import_data_to_database(host, port, database, user, password, csv_file_path, table_name):
    df = pd.read_csv(csv_file_path)
    engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{database}')
    df.to_sql(table_name, engine, if_exists='replace', index=False)
    print("The data has been imported into the table.")


host = 'localhost'
port = '5432'
database = 'database'
user = 'postgres'
password = 'Damian1999'

csv_file_path = '../../heart_disease_risk.csv'
table_name = 'heart_disease_risk'

import_data_to_database(host=host, port=port, database=database, user=user, password=password,
                        csv_file_path=csv_file_path, table_name=table_name)
