from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, create_engine, func, JSON
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.declarative import declarative_base
import os

Base = declarative_base()

class TimestampMixin:
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

class Symbol(TimestampMixin, Base):
    """
    Represents a stock symbol being tracked.
    """
    __tablename__ = 'symbols'

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    predictions = relationship('Prediction', back_populates='symbol')

class Prediction(TimestampMixin, Base):
    """
    Represents a prediction for a stock symbol.
    """
    __tablename__ = 'predictions'

    id = Column(Integer, primary_key=True, index=True)
    symbol_id = Column(Integer, ForeignKey('symbols.id'))
    symbol_aspect = Column(String, unique=True, index=True)
    prediction_start_time = Column(DateTime, index=True)
    prediction_end_time = Column(DateTime, index=True)
    prediction_value = Column(JSON)

    symbol = relationship('Symbol', back_populates='predictions')

# create session
database_url = os.getenv('DATABASE_URL', 'postgresql://user:password@localhost/dbname')
engine = create_engine(database_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
