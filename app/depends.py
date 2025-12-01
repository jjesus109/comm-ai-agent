import os

from psycopg import Connection

from app.config import Configuration

conf = Configuration()

DB_URI = (
    f"postgresql://{conf.db_user}:"
    f"{conf.db_password}@{conf.db_host}:{conf.db_port}/{conf.db_name}"
)

CAR_CATALOG_DB_URI = (
    f"postgresql://{conf.catalog_db_user}:"
    f"{conf.catalog_db_password}@{conf.catalog_db_host}:{conf.catalog_db_port}/{conf.catalog_db_name}"
)
connection_kwargs = {
    "autocommit": True,
    "prepare_threshold": 0,
}


data_path = os.path.join(os.getcwd(), "data")

agents_db_conn = Connection.connect(DB_URI, **connection_kwargs)

car_catalog_db_conn = Connection.connect(CAR_CATALOG_DB_URI, **connection_kwargs)
