from psycopg_pool import ConnectionPool
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

agents_db_conn = ConnectionPool(
    DB_URI,
    max_size=10,
    max_idle=300.0,
    kwargs={"autocommit": True},
)


car_catalog_db_conn = ConnectionPool(
    CAR_CATALOG_DB_URI,
    max_size=10,
    max_idle=300.0,
    kwargs={"autocommit": True},
)
