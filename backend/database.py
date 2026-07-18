import os

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker


# Prefer DATABASE_URL from environment (PostgreSQL in Docker/production),
# fall back to local SQLite for development.
SQLALCHEMY_DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./analytics_v2.db",
)

connect_args = {}
if SQLALCHEMY_DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def run_migrations():
    """
    Add columns that SQLAlchemy create_all cannot add to existing tables.
    New tables are still created by Base.metadata.create_all in main.py.
    """
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())

    def columns(table_name: str) -> set[str]:
        if table_name not in tables:
            return set()
        return {col["name"] for col in inspector.get_columns(table_name)}

    user_columns = columns("users")
    order_columns = columns("orders")
    is_sqlite = engine.dialect.name == "sqlite"
    bool_default = "INTEGER DEFAULT 1" if is_sqlite else "BOOLEAN DEFAULT TRUE"
    datetime_type = "DATETIME" if is_sqlite else "TIMESTAMP"
    float_type = "FLOAT" if is_sqlite else "DOUBLE PRECISION"

    migrations = []
    if user_columns:
        migrations.extend(
            [
                ("users", "role", "ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'owner'"),
                ("users", "is_active", f"ALTER TABLE users ADD COLUMN is_active {bool_default}"),
                ("users", "created_at", f"ALTER TABLE users ADD COLUMN created_at {datetime_type}"),
                ("users", "parent_id", "ALTER TABLE users ADD COLUMN parent_id INTEGER"),
            ]
        )

    if order_columns:
        migrations.append(
            ("orders", "product_cost", f"ALTER TABLE orders ADD COLUMN product_cost {float_type} DEFAULT 0")
        )
        migrations.append(
            ("orders", "product_id", "ALTER TABLE orders ADD COLUMN product_id TEXT")
        )

    with engine.begin() as conn:
        for table, col, sql in migrations:
            existing = user_columns if table == "users" else order_columns
            if col not in existing:
                conn.execute(text(sql))
