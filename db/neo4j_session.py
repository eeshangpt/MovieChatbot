import os
from contextlib import contextmanager

from dotenv import find_dotenv, load_dotenv
from neo4j import GraphDatabase

# ======================================================
# DATABASE CONFIG
# ======================================================

load_dotenv(find_dotenv(".env.dev"))

NEO4J_HOST = os.getenv("NEO4J_HOST", "localhost")
NEO4J_PORT = os.getenv("NEO4J_PORT", "7687")
NEO4J_USER = os.getenv("NEO4J_USER", "")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

NEO4J_URI = f"bolt://{NEO4J_HOST}:{NEO4J_PORT}"

# ======================================================
# NEO4J DRIVER
# ======================================================

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD),
    # Maximum connections kept open in the pool
    max_connection_pool_size=30,
    # Seconds to wait when acquiring a connection from the pool
    connection_acquisition_timeout=30,
    # Seconds to wait when establishing a new TCP connection
    connection_timeout=10,
)


# ======================================================
# SESSION FACTORY
# ======================================================


@contextmanager
def get_neo4j_session(**kwargs):
    session = driver.session(**kwargs)
    try:
        yield session
    finally:
        session.close()


__all__ = ["get_neo4j_session"]
