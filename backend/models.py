from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, Text, Boolean
from sqlalchemy.orm import relationship
from database import Base
import datetime


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    business_name = Column(String)
    full_name = Column(String, nullable=True)
    role = Column(String, default="owner")  # owner | admin | staff
    parent_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


    @property
    def shop_owner_id(self):
        return self.parent_id if self.role == "staff" and self.parent_id else self.id

    parent = relationship("User", remote_side="User.id", foreign_keys=[parent_id])
    orders = relationship("Order", back_populates="owner", cascade="all, delete-orphan")
    investments = relationship("Investment", back_populates="owner", cascade="all, delete-orphan")
    products = relationship("Product", back_populates="owner", cascade="all, delete-orphan")
    daily_insights = relationship("DailyInsight", back_populates="owner", cascade="all, delete-orphan")
    settings = relationship("UserSettings", back_populates="user", uselist=False, cascade="all, delete-orphan")


class Order(Base):
    __tablename__ = "orders"

    id = Column(Integer, primary_key=True, index=True)
    owner_id = Column(Integer, ForeignKey("users.id"), index=True)
    date = Column(String, index=True)           # "YYYY-MM-DD"
    consignment_id = Column(String, nullable=True)
    order_id = Column(String, nullable=True)
    customer_name = Column(String, nullable=True)
    customer_phone = Column(String, nullable=True)
    product_name = Column(String, nullable=True)
    product_id = Column(String, nullable=True)
    quantity = Column(Integer, default=1)
    amount = Column(Float, default=0.0)
    delivery_cost = Column(Float, default=100.0)
    product_cost = Column(Float, default=0.0)
    profit = Column(Float, default=0.0)
    status = Column(String, default="delivered")  # pending | delivered | returned | cancelled
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow)

    owner = relationship("User", back_populates="orders")


class Investment(Base):
    __tablename__ = "investments"

    id = Column(Integer, primary_key=True, index=True)
    owner_id = Column(Integer, ForeignKey("users.id"), index=True)
    date = Column(String, index=True)  # "YYYY-MM-DD"
    category = Column(String, default="Other", index=True)
    amount = Column(Float, default=0.0)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow)

    owner = relationship("User", back_populates="investments")


class Product(Base):
    __tablename__ = "products"

    id = Column(Integer, primary_key=True, index=True)
    owner_id = Column(Integer, ForeignKey("users.id"), index=True)
    sku = Column(String, nullable=True, index=True)
    name = Column(String, nullable=False)
    category = Column(String, nullable=True)
    supplier = Column(String, nullable=True)
    unit_cost = Column(Float, default=0.0)
    sell_price = Column(Float, default=0.0)
    current_stock = Column(Integer, default=0)
    reorder_level = Column(Integer, default=0)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow)

    owner = relationship("User", back_populates="products")


class DailyInsight(Base):
    """Aggregated daily cache — kept for backward compat with old upload flow."""
    __tablename__ = "daily_insights"

    id = Column(Integer, primary_key=True, index=True)
    owner_id = Column(Integer, ForeignKey("users.id"))
    date_str = Column(String, index=True)
    orders = Column(Integer, default=0)
    revenue = Column(Float, default=0.0)
    delivery_cost = Column(Float, default=0.0)
    profit = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    owner = relationship("User", back_populates="daily_insights")


class UserSettings(Base):
    __tablename__ = "user_settings"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True)
    delivery_cost_per_order = Column(Float, default=100.0)
    notification_email = Column(String, nullable=True)
    google_sheet_id = Column(String, nullable=True)
    google_credentials = Column(Text, nullable=True)  # JSON string of service-account key
    appscript_webhook_url = Column(String, nullable=True)
    gemini_api_key = Column(String, nullable=True)
    report_time = Column(String, default="20:00")  # e.g., "20:00" for 8 PM

    user = relationship("User", back_populates="settings")


class AuditLog(Base):
    """Immutable log of every create / update / delete action."""
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    owner_id = Column(Integer, ForeignKey("users.id"), index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    user_email = Column(String)
    action = Column(String, index=True)   # create | update | delete
    entity = Column(String, index=True)   # order | product | investment | staff | settings
    entity_id = Column(Integer, nullable=True)
    summary = Column(Text)
    details = Column(Text, nullable=True)  # JSON of changed fields
    created_at = Column(DateTime, default=datetime.datetime.utcnow, index=True)

