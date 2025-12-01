import sqlite3

db_path = "/Users/jesusalbino/Projects/kavak/commercial-ai-agent/example.db"
agents_db_conn = sqlite3.connect(db_path, check_same_thread=False)

car_catalog_db_path = (
    "/Users/jesusalbino/Projects/kavak/commercial-ai-agent/car_catalog.db"
)
car_catalog_db_conn = sqlite3.connect(car_catalog_db_path, check_same_thread=False)
