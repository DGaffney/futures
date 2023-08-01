import os
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, create_engine
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Symbol(Base):
    """
    Represents a stock symbol being tracked.
    """
    __tablename__ = 'symbols'

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    predictions = relationship('Prediction', back_populates='symbol')

class Prediction(Base):
    """
    Represents a prediction for a stock symbol.
    """
    __tablename__ = 'predictions'

    id = Column(Integer, primary_key=True, index=True)
    symbol_id = Column(Integer, ForeignKey('symbols.id'))
    prediction_time = Column(DateTime, index=True)
    prediction_value = Column(String)

    symbol = relationship('Symbol', back_populates='predictions')

# create session
database_url = os.getenv('DATABASE_URL', 'postgresql://user:password@localhost/dbname')
engine = create_engine(database_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
