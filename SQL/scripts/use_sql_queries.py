import psycopg2


def execute_sql_query(host, port, database, user, password, sql_file_path):
    connection = psycopg2.connect(
        host=host,
        port=port,
        database=database,
        user=user,
        password=password
    )
    cursor = connection.cursor()

    with open(sql_file_path, 'r') as file:
        sql_query = file.read()
    try:
        cursor.execute(sql_query)
        connection.commit()
        print("SQL query executed successfully.")
    except Exception as e:
        connection.rollback()
        print("Error executing SQL query:", str(e))
    finally:
        cursor.close()
        connection.close()


host = 'localhost'
port = '5432'
database = 'database'
user = 'postgres'
password = 'Damian1999'

sql_file_path = '../sql_queries/create_id_column.sql'

execute_sql_query(host=host, port=port, database=database, user=user, password=password, sql_file_path=sql_file_path)
