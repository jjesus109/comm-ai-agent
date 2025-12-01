import os

import sqlite3

data_path = os.path.join(os.getcwd(), "data")
db_path = os.path.join(data_path, "example.db")
agents_db_conn = sqlite3.connect(db_path, check_same_thread=False)

car_catalog_db_path = os.path.join(data_path, "car_catalog.db")
car_catalog_db_conn = sqlite3.connect(car_catalog_db_path, check_same_thread=False)
