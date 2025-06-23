from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
import os

Base = declarative_base()
engine = create_engine(os.getenv("DB_URL"))
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def initialize_database():
    """Create all tables on startup"""
    Base.metadata.create_all(bind=engine)  # No need to import models here

# Initialize on import
initialize_database() 