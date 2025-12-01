import os

import sqlite3
from psycopg import Connection

from app.config import Configuration

conf = Configuration()

DB_URI = (
    f"postgresql://{conf.db_user}:"
    f"{conf.db_password}@{conf.db_host}:{conf.db_port}/{conf.db_name}"
)
connection_kwargs = {
    "autocommit": True,
    "prepare_threshold": 0,
}


data_path = os.path.join(os.getcwd(), "data")

agents_db_conn = Connection.connect(DB_URI, **connection_kwargs)

car_catalog_db_path = os.path.join(data_path, "car_catalog.db")
car_catalog_db_conn = sqlite3.connect(car_catalog_db_path, check_same_thread=False)
