"""
main.py — myShop AI · FastAPI backend
======================================
Routes:
  Auth         POST /auth/register  POST /auth/login  GET /auth/me
  Orders       GET POST /orders     PUT DELETE /orders/{id}
               POST /orders/bulk-upload
  Analytics    GET /analytics/summary  /analytics/daily  /analytics/top-customers
  AI           POST /ai/insights    POST /ai/query
  Sheets       POST /sheets/connect  /sheets/sync  /sheets/import
  Settings     GET PUT /settings
"""

import io
import json
import logging
import os
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

load_dotenv()

import ai_service
import auth
import database
import email_service
import models
import sheets_service
from database import engine, get_db

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ─── App init ────────────────────────────────────────────────────────────────
models.Base.metadata.create_all(bind=engine)  # create any missing tables
database.run_migrations()                      # add any missing columns to existing tables

DELIVERY_CHARGE = float(os.getenv("DELIVERY_CHARGE_PER_ORDER", "100"))
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "15"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    email_service.start_scheduler()
    yield


app = FastAPI(
    title="myShop AI — Sales Management API",
    description="AI-powered sales platform using Gemma 4. Supports CRUD orders, analytics, Google Sheets sync.",
    version="2.0.0",
    lifespan=lifespan,
)

cors_origins_env = os.getenv("CORS_ORIGINS", "")
if cors_origins_env:
    cors_origins = [o.strip() for o in cors_origins_env.split(",")]
else:
    cors_origins = ["*"]  # Allow all origins in development

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=cors_origins != ["*"],  # credentials not allowed with wildcard
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Auth helpers ─────────────────────────────────────────────────────────────
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")


def get_current_user(
    token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)
) -> models.User:
    exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = auth.jwt.decode(token, auth.SECRET_KEY, algorithms=[auth.ALGORITHM])
        email: str = payload.get("sub")
        if not email:
            raise exc
    except auth.JWTError:
        raise exc
    user = db.query(models.User).filter(models.User.email == email).first()
    if not user or not user.is_active:
        raise exc
    return user


def get_user_dc(user: models.User, db: Session) -> float:
    """Return the delivery cost configured for this user."""
    s = db.query(models.UserSettings).filter(
        models.UserSettings.user_id == user.shop_owner_id
    ).first()
    return s.delivery_cost_per_order if s else DELIVERY_CHARGE


def _normalize_text(value: Optional[str]) -> str:
    return (value or "").strip().lower()


def _log_audit(
    db: Session,
    *,
    owner_id: int = None,
    user: models.User,
    action: str,
    entity: str,
    entity_id: int = None,
    summary: str,
    details: dict = None,
):
    """Record an immutable audit-log entry.  Rides the caller's transaction."""
    actual_owner_id = user.parent_id if user.role == "staff" and user.parent_id else user.id
    
    entry = models.AuditLog(
        owner_id=actual_owner_id,
        user_id=user.id,
        user_email=user.email,
        action=action,
        entity=entity,
        entity_id=entity_id,
        summary=summary,
        details=json.dumps(details) if details else None,
    )
    db.add(entry)


def _is_stock_consuming_status(status: Optional[str]) -> bool:
    return _normalize_text(status) not in {"cancelled", "canceled", "returned"}


def _find_product_for_order(db: Session, user_id: int, product_name: Optional[str], sku: Optional[str] = None):
    if sku:
        sku_key = sku.strip().lower()
        match = db.query(models.Product).filter(
            models.Product.owner_id == user_id,
            func.lower(models.Product.sku) == sku_key
        ).first()
        if match: return match

    if not product_name:
        return None
    # Parse frontend's "SKU - Name" format if present
    parts = [p.strip() for p in product_name.split(" - ", 1)]
    if len(parts) == 2:
        sku_key = parts[0].lower()
        name_key = parts[1].lower()
        match = db.query(models.Product).filter(
            models.Product.owner_id == user_id,
            (func.lower(models.Product.sku) == sku_key) | (func.lower(models.Product.name) == name_key)
        ).first()
        if match: return match
        
    key = _normalize_text(product_name)
    if not key:
        return None
    return (
        db.query(models.Product)
        .filter(models.Product.owner_id == user_id)
        .filter(
            (func.lower(models.Product.name) == key)
            | (func.lower(models.Product.sku) == key)
        )
        .first()
    )


def _consume_product_stock(db: Session, user_id: int, product_name: Optional[str], quantity: int = 1) -> None:
    if quantity <= 0:
        return
    product = _find_product_for_order(db, user_id, product_name)
    if not product:
        return
    current_stock = int(product.current_stock or 0)
    if current_stock < quantity:
        raise HTTPException(status_code=400, detail=f"Insufficient stock for {product.name}")
    product.current_stock = current_stock - quantity
    product.updated_at = datetime.utcnow()


def _restore_product_stock(db: Session, user_id: int, product_name: Optional[str], quantity: int = 1) -> None:
    if quantity <= 0:
        return
    product = _find_product_for_order(db, user_id, product_name)
    if not product:
        return
    product.current_stock = int(product.current_stock or 0) + quantity
    product.updated_at = datetime.utcnow()


def _generate_order_id(db: Session, user_id: int) -> str:
    prefix = datetime.utcnow().strftime("ORD-%Y%m%d-%H%M%S")
    for _ in range(20):
        candidate = f"{prefix}-{uuid.uuid4().hex[:4].upper()}"
        exists = db.query(models.Order.id).filter(
            models.Order.owner_id == user_id,
            models.Order.order_id == candidate,
        ).first()
        if not exists:
            return candidate
    return f"{prefix}-{uuid.uuid4().hex[:8].upper()}"


# ─── Pydantic schemas ─────────────────────────────────────────────────────────

class OrderCreate(BaseModel):
    date: str
    consignment_id: Optional[str] = None
    order_id: Optional[str] = None
    customer_name: Optional[str] = None
    customer_phone: Optional[str] = None
    product_name: Optional[str] = None
    product_id: Optional[str] = None
    quantity: int = 1
    amount: float
    delivery_cost: Optional[float] = None
    product_cost: Optional[float] = 0.0
    status: Optional[str] = "delivered"
    notes: Optional[str] = None


class OrderUpdate(BaseModel):
    date: Optional[str] = None
    consignment_id: Optional[str] = None
    order_id: Optional[str] = None
    customer_name: Optional[str] = None
    customer_phone: Optional[str] = None
    product_name: Optional[str] = None
    product_id: Optional[str] = None
    quantity: Optional[int] = None
    amount: Optional[float] = None
    delivery_cost: Optional[float] = None
    product_cost: Optional[float] = None
    status: Optional[str] = None
    notes: Optional[str] = None


class AIQueryRequest(BaseModel):
    question: str


class SheetsConnectRequest(BaseModel):
    sheet_id: str
    credentials_json: Optional[str] = None


class SettingsUpdate(BaseModel):
    delivery_cost_per_order: Optional[float] = None
    notification_email: Optional[str] = None
    google_sheet_id: Optional[str] = None
    report_time: Optional[str] = None


class InvestmentCreate(BaseModel):
    date: str
    category: str
    amount: float
    notes: Optional[str] = None
    notes_quantity: Optional[str] = None


class InvestmentUpdate(BaseModel):
    date: Optional[str] = None
    category: Optional[str] = None
    amount: Optional[float] = None
    notes: Optional[str] = None
    notes_quantity: Optional[str] = None


class ProductCreate(BaseModel):
    sku: Optional[str] = None
    name: str
    category: Optional[str] = None
    supplier: Optional[str] = None
    unit_cost: Optional[float] = 0.0
    sell_price: Optional[float] = 0.0
    current_stock: Optional[int] = 0
    reorder_level: Optional[int] = 0
    notes: Optional[str] = None


class ProductUpdate(BaseModel):
    sku: Optional[str] = None
    name: Optional[str] = None
    category: Optional[str] = None
    supplier: Optional[str] = None
    unit_cost: Optional[float] = None
    sell_price: Optional[float] = None
    current_stock: Optional[int] = None
    reorder_level: Optional[int] = None
    notes: Optional[str] = None


class StaffCreate(BaseModel):
    name: str
    email: str
    password: str



def _round_money(value: float) -> float:
    return round(float(value or 0), 2)


def _profit(amount: float, delivery_cost: float, product_cost: float = 0.0) -> float:
    return _round_money((amount or 0) - (product_cost or 0))


def _order_sign(order) -> int:
    """Return -1 for cancelled/returned orders, +1 otherwise.
    Cancelled/returned amounts are subtracted from financial totals."""
    return -1 if (order.status or "").strip().lower() in ("cancelled", "canceled", "returned") else 1


def _investment_notes(data: InvestmentCreate | InvestmentUpdate) -> Optional[str]:
    return data.notes if data.notes is not None else data.notes_quantity


def _investment_to_dict(item: models.Investment) -> dict:
    return {
        "id": item.id,
        "date": item.date,
        "category": item.category,
        "amount": _round_money(item.amount),
        "notes": item.notes,
        "notes_quantity": item.notes,
        "created_at": item.created_at.isoformat() if item.created_at else None,
        "updated_at": item.updated_at.isoformat() if item.updated_at else None,
    }


def _product_status(product: models.Product) -> str:
    stock = int(product.current_stock or 0)
    reorder = int(product.reorder_level or 0)
    if stock <= 0:
        return "Out of Stock"
    if stock <= reorder:
        return "Low Stock"
    return "In Stock"


def _product_to_dict(product: models.Product) -> dict:
    return {
        "id": product.id,
        "sku": product.sku,
        "name": product.name,
        "category": product.category,
        "supplier": product.supplier,
        "unit_cost": _round_money(product.unit_cost),
        "sell_price": _round_money(product.sell_price),
        "current_stock": int(product.current_stock or 0),
        "reorder_level": int(product.reorder_level or 0),
        "status": _product_status(product),
        "notes": product.notes,
        "created_at": product.created_at.isoformat() if product.created_at else None,
        "updated_at": product.updated_at.isoformat() if product.updated_at else None,
    }


def _customer_aggregates(orders: list[models.Order], inactive_days: int = 30) -> list[dict]:
    from collections import Counter
    customers: dict = defaultdict(
        lambda: {
            "phone": "",
            "name": "",
            "first_order": "",
            "last_order": "",
            "total_orders": 0,
            "total_revenue": 0.0,
            "categories": Counter(),
            "timeline": []
        }
    )

    for order in orders:
        key = (order.customer_phone or order.customer_name or f"order-{order.id}").strip()
        row = customers[key]
        row["phone"] = order.customer_phone or row["phone"]
        row["name"] = order.customer_name or row["name"] or order.customer_phone or "Unknown"
        row["total_orders"] += 1
        
        if order.product_name:
            # Simple category detection based on name if no proper relation exists, or we just store product name
            cat = "Electronics" if any(x in order.product_name.lower() for x in ["monitor", "hub", "ssd", "hdd", "router"]) else "Accessories"
            row["categories"][cat] += 1

        sign = _order_sign(order)
        row["total_revenue"] += sign * (order.amount or 0)
        
        if order.date:
            if not row["first_order"] or order.date < row["first_order"]:
                row["first_order"] = order.date
            if not row["last_order"] or order.date > row["last_order"]:
                row["last_order"] = order.date
            row["timeline"].append({
                "date": order.date,
                "product": order.product_name,
                "amount": _round_money(order.amount or 0),
                "status": order.status
            })

    from datetime import datetime
    today = datetime.now().date()
    
    rows = []
    for row in customers.values():
        orders_count = row["total_orders"]
        revenue = row["total_revenue"]
        
        # Calculate days since last order
        days_since = 0
        if row["last_order"]:
            try:
                # Handle string dates (e.g. YYYY-MM-DD)
                if isinstance(row["last_order"], str):
                    last_dt = datetime.strptime(row["last_order"][:10], "%Y-%m-%d").date()
                else:
                    last_dt = row["last_order"].date() if hasattr(row["last_order"], "date") else row["last_order"]
                days_since = (today - last_dt).days
                if days_since < 0:
                    days_since = 0
            except:
                days_since = 0
                
        is_at_risk = days_since > inactive_days
        
        # Segment logic
        if is_at_risk:
            segment = "At-Risk"
        elif revenue > 5000:
            segment = "VIP"
        elif orders_count >= 3:
            segment = "Loyal"
        else:
            segment = "New"
            
        fav_cat = row["categories"].most_common(1)[0][0] if row["categories"] else "Unknown"
        
        row["timeline"].sort(key=lambda x: x["date"], reverse=True)

        rows.append(
            {
                **row,
                "total_revenue": _round_money(revenue),
                "avg_order": _round_money(revenue / orders_count) if orders_count else 0,
                "days_since_last_order": days_since,
                "is_at_risk": is_at_risk,
                "segment": segment,
                "favorite_category": fav_cat,
                "timeline": row["timeline"][:15] # keep recent 15 for payload size
            }
        )
        # cleanup Counter before returning
        del rows[-1]["categories"]

    return sorted(rows, key=lambda item: item["total_revenue"], reverse=True)


def _daily_aggregates(orders: list[models.Order]) -> list[dict]:
    daily: dict = defaultdict(
        lambda: {"orders": 0, "revenue": 0.0, "delivery_cost": 0.0, "profit": 0.0}
    )
    for order in orders:
        if not order.date:
            continue
        day = daily[order.date]
        day["orders"] += 1
        
        sign = _order_sign(order)
        day["revenue"] += sign * ((order.amount or 0) + (order.delivery_cost or 0))
        day["delivery_cost"] += sign * (order.delivery_cost or 0)
        day["profit"] += sign * (order.profit or 0)

    return [
        {
            "date": date,
            "orders": values["orders"],
            "revenue": _round_money(values["revenue"]),
            "delivery_cost": _round_money(values["delivery_cost"]),
            "profit": _round_money(values["profit"]),
            "average_order": _round_money(values["revenue"] / values["orders"])
            if values["orders"]
            else 0,
        }
        for date, values in sorted(daily.items())
    ]


def _monthly_aggregates(
    orders: list[models.Order], investments: list[models.Investment]
) -> list[dict]:
    monthly: dict = defaultdict(
        lambda: {
            "orders": 0,
            "revenue": 0.0,
            "gross_profit": 0.0,
            "delivery_cost": 0.0,
            "investment": 0.0,
        }
    )

    for order in orders:
        if not order.date:
            continue
        month = order.date[:7]
        row = monthly[month]
        row["orders"] += 1
        
        sign = _order_sign(order)
        row["revenue"] += sign * ((order.amount or 0) + (order.delivery_cost or 0))
        row["gross_profit"] += sign * ((order.amount or 0) - (order.product_cost or 0))
        row["delivery_cost"] += sign * (order.delivery_cost or 0)

    for item in investments:
        if item.date:
            monthly[item.date[:7]]["investment"] += item.amount or 0

    rows = []
    for month, row in sorted(monthly.items()):
        revenue = row["revenue"]
        gross_profit = row["gross_profit"]
        investment = row["investment"]
        rows.append(
            {
                "month": month,
                "orders": row["orders"],
                "revenue": _round_money(revenue),
                "gross_profit": _round_money(gross_profit),
                "delivery_cost": _round_money(row["delivery_cost"]),
                "investment": _round_money(investment),
                "net_cash_flow": _round_money(gross_profit - investment),
                "gross_margin": round((gross_profit / revenue) * 100, 1) if revenue else 0,
            }
        )
    return rows


def _dashboard_summary(
    orders: list[models.Order], investments: list[models.Investment]
) -> dict:
    cancelled_returned = [o for o in orders if (o.status or "").strip().lower() in ("cancelled", "canceled", "returned")]
    
    total_revenue = sum(_order_sign(o) * ((o.amount or 0) + (o.delivery_cost or 0)) for o in orders)
    total_delivery = sum(_order_sign(o) * (o.delivery_cost or 0) for o in orders)
    gross_profit = sum(_order_sign(o) * ((o.amount or 0) - (o.product_cost or 0)) for o in orders)
    total_investment = sum(item.amount or 0 for item in investments)
    total_orders = len(orders)
    cancelled_returned_count = len(cancelled_returned)
    cancelled_returned_amount = sum((o.amount or 0) + (o.delivery_cost or 0) for o in cancelled_returned)
    dates = [order.date for order in orders if order.date]

    return {
        "total_revenue": _round_money(total_revenue),
        "total_orders": total_orders,
        "gross_profit": _round_money(gross_profit),
        "total_profit": _round_money(gross_profit),
        "net_cash_flow": _round_money(gross_profit - total_investment),
        "total_investment": _round_money(total_investment),
        "gross_margin": round((gross_profit / total_revenue) * 100, 1) if total_revenue else 0,
        "profit_margin": round((gross_profit / total_revenue) * 100, 1) if total_revenue else 0,
        "avg_order_value": _round_money(total_revenue / total_orders) if total_orders else 0,
        "total_delivery_cost": _round_money(total_delivery),
        "total_customers": len({order.customer_phone for order in orders if order.customer_phone}),
        "investment_count": len(investments),
        "investment_entries": len(investments),
        "last_sale_date": max(dates) if dates else None,
        "last_order_date": max(dates) if dates else None,
        "highest_order_value": _round_money(max((order.amount or 0 for order in orders), default=0)),
        "cancelled_returned_count": cancelled_returned_count,
        "cancelled_returned_amount": _round_money(cancelled_returned_amount),
    }


def _sync_shop_sheet_for_user(user_id: int, db: Session) -> None:
    settings = db.query(models.UserSettings).filter(models.UserSettings.user_id == user_id).first()
    if not settings or not settings.google_sheet_id:
        return

    orders = db.query(models.Order).filter(models.Order.owner_id == user_id).all()
    investments = db.query(models.Investment).filter(models.Investment.owner_id == user_id).all()
    products = db.query(models.Product).filter(models.Product.owner_id == user_id).all()

    result = sheets_service.sync_shop_to_sheet(
        settings.google_sheet_id,
        orders=[
            {
                "id": order.id,
                "date": order.date,
                "consignment_id": order.consignment_id,
                "order_id": order.order_id,
                "customer_name": order.customer_name,
                "customer_phone": order.customer_phone,
                "product_name": order.product_name,
                "amount": order.amount,
                "delivery_cost": order.delivery_cost,
                "product_cost": order.product_cost,
                "profit": order.profit,
                "status": order.status,
                "notes": order.notes,
            }
            for order in orders
        ],
        investments=[_investment_to_dict(item) for item in investments],
        customers=_customer_aggregates(orders),
        daily=_daily_aggregates(orders),
        products=[_product_to_dict(product) for product in products],
        credentials_json=settings.google_credentials,
    )
    if not result.get("success"):
        logger.warning("Google Sheets auto-sync failed for user %s: %s", user_id, result.get("error"))


# ═══════════════════════════════════════════════════════════════════════════════
# AUTH ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/auth/register", tags=["Auth"])
def register(user_data: auth.UserCreate, db: Session = Depends(get_db)):
    if db.query(models.User).filter(models.User.email == user_data.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")

    role = user_data.role if user_data.role in ("owner", "admin", "staff") else "owner"
    new_user = models.User(
        email=user_data.email,
        hashed_password=auth.get_password_hash(user_data.password),
        business_name=user_data.business_name,
        role=role,
    )
    db.add(new_user)
    db.flush()

    # Create default settings
    db.add(models.UserSettings(user_id=new_user.id, delivery_cost_per_order=DELIVERY_CHARGE))
    db.commit()
    logger.info(f"New user registered: {user_data.email} ({role})")
    return {"message": "Account created successfully", "role": role}


@app.post("/auth/login", tags=["Auth"])
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.email == form_data.username).first()
    if not user or not auth.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect email or password")

    token = auth.create_access_token(
        data={"sub": user.email},
        expires_delta=timedelta(minutes=auth.ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return {"access_token": token, "token_type": "bearer", "business_name": user.business_name, "role": user.role}


# Legacy token endpoint — keeps old frontend working during migration
@app.post("/token", include_in_schema=False)
def token_legacy(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    return login(form_data, db)

# Legacy register endpoint
@app.post("/register", include_in_schema=False)
def register_legacy(user_data: auth.UserCreate, db: Session = Depends(get_db)):
    return register(user_data, db)


@app.get("/auth/me", tags=["Auth"])
def get_me(current_user: models.User = Depends(get_current_user)):
    return {
        "id": current_user.id,
        "email": current_user.email,
        "business_name": current_user.business_name,
        "role": current_user.role,
        "created_at": current_user.created_at.isoformat() if current_user.created_at else None,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# STAFF MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/staff", tags=["Staff"])
def list_staff(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if current_user.role != "owner":
        raise HTTPException(status_code=403, detail="Not authorized")
    
    staff_members = db.query(models.User).filter(
        models.User.parent_id == current_user.id,
        models.User.role == "staff"
    ).all()
    
    return [
        {
            "id": s.id,
            "name": s.full_name or s.email.split("@")[0],
            "email": s.email,
            "is_active": s.is_active,
        } for s in staff_members
    ]

@app.post("/staff", tags=["Staff"])
def create_staff(
    data: StaffCreate,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if current_user.role != "owner":
        raise HTTPException(status_code=403, detail="Not authorized")
        
    if db.query(models.User).filter(models.User.email == data.email).first():
        raise HTTPException(status_code=400, detail="Email already exists")
        
    new_staff = models.User(
        email=data.email,
        full_name=data.name,
        hashed_password=auth.get_password_hash(data.password),
        business_name=current_user.business_name,
        role="staff",
        parent_id=current_user.id,
        is_active=True
    )
    db.add(new_staff)
    db.commit()
    db.refresh(new_staff)
    _log_audit(
        db=db,
        user=current_user,
        action="create",
        entity="staff",
        entity_id=new_staff.id,
        summary=f"Added staff member {new_staff.email}",
        details={"new": {"email": data.email, "name": data.name}},
    )
    return {"id": new_staff.id, "email": new_staff.email}

@app.delete("/staff/{staff_id}", tags=["Staff"])
def delete_staff(
    staff_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if current_user.role != "owner":
        raise HTTPException(status_code=403, detail="Not authorized")
        
    staff = db.query(models.User).filter(
        models.User.id == staff_id,
        models.User.parent_id == current_user.id,
        models.User.role == "staff"
    ).first()
    
    if not staff:
        raise HTTPException(status_code=404, detail="Staff not found")
        
    email = staff.email
    db.delete(staff)
    db.commit()
    _log_audit(
        db=db,
        user=current_user,
        action="delete",
        entity="staff",
        entity_id=staff_id,
        summary=f"Removed staff member {email}",
        details={"old": {"email": email}},
    )
    return {"message": "Staff removed"}



# ═══════════════════════════════════════════════════════════════════════════════
# ORDERS — CRUD
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/orders", tags=["Orders"])
def list_orders(
    search: Optional[str] = None,
    status: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=50, ge=1, le=200),
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    q = db.query(models.Order).filter(models.Order.owner_id == current_user.shop_owner_id)

    if search:
        like = f"%{search}%"
        q = q.filter(
            models.Order.customer_name.ilike(like)
            | models.Order.customer_phone.ilike(like)
            | models.Order.order_id.ilike(like)
            | models.Order.product_name.ilike(like)
            | models.Order.product_id.ilike(like)
            | models.Order.consignment_id.ilike(like)
        )
    if status:
        q = q.filter(models.Order.status == status)
    if start_date:
        q = q.filter(models.Order.date >= start_date)
    if end_date:
        q = q.filter(models.Order.date <= end_date)

    total = q.count()
    items = q.order_by(models.Order.date.desc(), models.Order.id.desc()).offset((page - 1) * limit).limit(limit).all()

    return {"items": items, "total": total, "page": page, "pages": max(1, (total + limit - 1) // limit)}


@app.post("/orders", tags=["Orders"])
def create_order(
    data: OrderCreate,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    dc = data.delivery_cost if data.delivery_cost is not None else get_user_dc(current_user, db)
    product_cost = data.product_cost or 0
    if product_cost == 0 and data.product_name:
        prod = db.query(models.Product).filter(
            models.Product.owner_id == current_user.shop_owner_id,
            models.Product.name == data.product_name
        ).first()
        if prod and prod.unit_cost:
            product_cost = prod.unit_cost * (data.quantity or 1)
            
    order_id = (data.order_id or "").strip() or _generate_order_id(db, current_user.shop_owner_id)
    existing = db.query(models.Order.id).filter(
        models.Order.owner_id == current_user.shop_owner_id,
        models.Order.order_id == order_id,
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail="Order ID already exists")

    if _is_stock_consuming_status(data.status):
        _consume_product_stock(db, current_user.shop_owner_id, data.product_name, data.quantity)

    order = models.Order(
        owner_id=current_user.shop_owner_id,
        date=data.date,
        consignment_id=data.consignment_id or None,
        order_id=order_id,
        customer_name=data.customer_name or None,
        customer_phone=data.customer_phone or None,
        product_name=data.product_name or None,
        product_id=data.product_id or None,
        quantity=data.quantity,
        amount=data.amount,
        delivery_cost=dc,
        product_cost=product_cost,
        profit=_profit(data.amount, dc, product_cost),
        status=data.status or "delivered",
        notes=data.notes or None,
    )
    db.add(order)
    _log_audit(
        db, owner_id=current_user.shop_owner_id, user=current_user,
        action="create", entity="order", entity_id=None,
        summary=f"Created order {order_id}",
    )
    db.commit()
    db.refresh(order)
    _sync_shop_sheet_for_user(current_user.shop_owner_id, db)
    return order


@app.put("/orders/{order_id}", tags=["Orders"])
def update_order(
    order_id: int,
    data: OrderUpdate,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    order = db.query(models.Order).filter(
        models.Order.id == order_id, models.Order.owner_id == current_user.shop_owner_id
    ).first()
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")

    if current_user.role == "staff" and order.date != datetime.now().strftime("%Y-%m-%d"):
        raise HTTPException(status_code=403, detail="Staff may only edit today's orders")

    updated_order_id = order.order_id
    if data.order_id is not None:
        updated_order_id = data.order_id.strip() or order.order_id or _generate_order_id(db, current_user.shop_owner_id)
        if updated_order_id != order.order_id:
            duplicate = db.query(models.Order.id).filter(
                models.Order.owner_id == current_user.shop_owner_id,
                models.Order.order_id == updated_order_id,
                models.Order.id != order.id,
            ).first()
            if duplicate:
                raise HTTPException(status_code=400, detail="Order ID already exists")

    next_product_name = data.product_name if data.product_name is not None else order.product_name
    next_status = data.status if data.status is not None else order.status
    next_quantity = data.quantity if data.quantity is not None else order.quantity

    current_effective_product = _normalize_text(order.product_name) if _is_stock_consuming_status(order.status) else ""
    next_effective_product = _normalize_text(next_product_name) if _is_stock_consuming_status(next_status) else ""

    if current_effective_product != next_effective_product or order.quantity != next_quantity:
        if current_effective_product:
            _restore_product_stock(db, current_user.shop_owner_id, order.product_name, order.quantity)
        if next_effective_product:
            _consume_product_stock(db, current_user.shop_owner_id, next_product_name, next_quantity)

    changed = {k: v for k, v in data.dict(exclude_unset=True).items()}
    
    # Recalculate product cost if it's 0 or missing, and we have a product name
    if (changed.get("product_cost") in (None, 0, "")) and next_product_name:
        prod = db.query(models.Product).filter(
            models.Product.owner_id == current_user.shop_owner_id,
            models.Product.name == next_product_name
        ).first()
        if prod and prod.unit_cost:
            changed["product_cost"] = prod.unit_cost * (next_quantity or 1)
            
    old_vals = {k: getattr(order, k) for k in changed}
    for field, value in changed.items():
        setattr(order, field, value)

    order.order_id = updated_order_id

    order.profit = _profit(order.amount, order.delivery_cost, order.product_cost)
    order.updated_at = datetime.utcnow()
    _log_audit(
        db, owner_id=current_user.shop_owner_id, user=current_user,
        action="update", entity="order", entity_id=order.id,
        summary=f"Updated order {order.order_id or order.id}",
        details={"old": old_vals, "new": changed},
    )
    db.commit()
    db.refresh(order)
    _sync_shop_sheet_for_user(current_user.shop_owner_id, db)
    return order


@app.delete("/orders/{order_id}", tags=["Orders"])
def delete_order(
    order_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if current_user.role == "staff":
        raise HTTPException(status_code=403, detail="Staff cannot delete orders")

    order = db.query(models.Order).filter(
        models.Order.id == order_id, models.Order.owner_id == current_user.shop_owner_id
    ).first()
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")

    if _is_stock_consuming_status(order.status):
        _restore_product_stock(db, current_user.shop_owner_id, order.product_name)

    _log_audit(
        db, owner_id=current_user.shop_owner_id, user=current_user,
        action="delete", entity="order", entity_id=order.id,
        summary=f"Deleted order {order.order_id or order.id}",
    )
    db.delete(order)
    db.commit()
    _sync_shop_sheet_for_user(current_user.shop_owner_id, db)
    return {"message": "Order deleted"}


# ─── Bulk upload ──────────────────────────────────────────────────────────────

def _clean_amount(val) -> float:
    if pd.isna(val):
        return 0.0
    try:
        return float(str(val).replace("৳", "").replace(",", "").strip())
    except (ValueError, TypeError):
        return 0.0


@app.post("/orders/bulk-upload", tags=["Orders"])
async def bulk_upload(
    file: UploadFile = File(...),
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not file.filename.lower().endswith((".xlsx", ".xls", ".csv")):
        raise HTTPException(status_code=400, detail="Only Excel (.xlsx/.xls) or CSV files are accepted")

    content = await file.read()
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {MAX_UPLOAD_MB:.0f} MB limit")

    try:
        df = (
            pd.read_csv(io.BytesIO(content))
            if file.filename.lower().endswith(".csv")
            else pd.read_excel(io.BytesIO(content))
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not parse file: {exc}")

    if "Amount" not in df.columns:
        raise HTTPException(status_code=422, detail="File must contain an 'Amount' column")
    if "Date" not in df.columns:
        raise HTTPException(status_code=422, detail="File must contain a 'Date' column")

    df["Amount"] = df["Amount"].apply(_clean_amount)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df[df["Amount"] > 0].dropna(subset=["Date"]).copy()

    dc = get_user_dc(current_user, db)
    imported = 0

    for _, row in df.iterrows():
        date_str = row["Date"].strftime("%Y-%m-%d")
        amount = float(row["Amount"])

        cname = str(row.get("Customer Name", "")).replace("Name: ", "").strip()
        cphone = str(row.get("Customer Phone", "")).strip()
        product = str(row.get("Product Name", row.get("Product", ""))).strip()
        product_id = str(row.get("Product ID", row.get("SKU", ""))).strip()
        cid = str(row.get("Consignment ID", "")).replace("#", "").strip()
        oid = str(row.get("Order ID", "")).replace("#", "").strip()
        row_delivery = _clean_amount(row.get("Delivery Cost", dc))
        
        product_cost = _clean_amount(row.get("Product Cost", row.get("Cost", 0)))
        if product_cost == 0 and (product or product_id):
            matched_product = _find_product_for_order(db, current_user.shop_owner_id, product, product_id)
            if matched_product:
                product_cost = matched_product.unit_cost or 0
                
        row_profit = _clean_amount(row.get("Per Order Profit", row.get("Profit", None)))
        profit = row_profit if row_profit else _profit(amount, row_delivery, product_cost)

        order = models.Order(
            owner_id=current_user.shop_owner_id,
            date=date_str,
            consignment_id=cid or None,
            order_id=oid or None,
            customer_name=cname or None,
            customer_phone=cphone or None,
            product_name=product or None,
            product_id=product_id or None,
            amount=amount,
            delivery_cost=row_delivery,
            product_cost=product_cost,
            profit=profit,
            status="delivered",
        )
        db.add(order)
        imported += 1

    _log_audit(
        db, owner_id=current_user.shop_owner_id, user=current_user,
        action="create", entity="order", entity_id=None,
        summary=f"Bulk uploaded {imported} orders",
    )
    db.commit()
    _sync_shop_sheet_for_user(current_user.shop_owner_id, db)

    # Build analytics context for AI
    all_orders = db.query(models.Order).filter(models.Order.owner_id == current_user.shop_owner_id).all()
    total_rev = sum(_order_sign(o) * (o.amount or 0) for o in all_orders)
    total_profit = sum(_order_sign(o) * (o.profit or 0) for o in all_orders)
    n_orders = len(all_orders)
    unique_phones = {o.customer_phone for o in all_orders if o.customer_phone}
    aov = total_rev / n_orders if n_orders else 0

    cutoff = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    at_risk = sum(
        1 for p in unique_phones
        if any(o.customer_phone == p and o.date and o.date < cutoff for o in all_orders)
        and not any(o.customer_phone == p and o.date and o.date >= cutoff for o in all_orders)
    )

    top_customers = sorted(
        [
            {"phone": p, "orders": sum(1 for o in all_orders if o.customer_phone == p), "spent": sum(_order_sign(o) * (o.amount or 0) for o in all_orders if o.customer_phone == p)}
            for p in unique_phones
        ],
        key=lambda x: x["spent"],
        reverse=True,
    )[:3]

    s = db.query(models.UserSettings).filter(models.UserSettings.user_id == current_user.shop_owner_id).first()

    products = db.query(models.Product).filter(models.Product.owner_id == current_user.shop_owner_id).all()
    investments = db.query(models.Investment).filter(models.Investment.owner_id == current_user.shop_owner_id).all()
    ai_context = _build_ai_context(
        business_name=current_user.business_name,
        orders=all_orders,
        investments=investments,
        products=products,
    )

    ai_insights = ai_service.generate_insights(
        business_name=current_user.business_name,
        context_data=ai_context,
    )

    logger.info(f"Bulk upload: {imported} orders imported for user {current_user.email}")
    return {
        "imported": imported,
        "total_orders": n_orders,
        "total_revenue": round(total_rev, 2),
        "total_profit": round(total_profit, 2),
        "unique_customers": len(unique_phones),
        "top_customers": top_customers,
        "ai_insights": ai_insights,
        "at_risk_customers": at_risk,
    }


# ─── Legacy upload route ──────────────────────────────────────────────────────
@app.post("/upload/", include_in_schema=False)
async def upload_legacy(
    file: UploadFile = File(...),
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return await bulk_upload(file, current_user, db)


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════

from sqlalchemy import func

@app.get("/analytics/summary", tags=["Analytics"])
def analytics_summary(
    current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)
):
    owner_id = current_user.shop_owner_id
    
    orders = db.query(models.Order).filter(models.Order.owner_id == owner_id).all()
    investments = db.query(models.Investment).filter(models.Investment.owner_id == owner_id).all()
    return _dashboard_summary(orders, investments)


@app.get("/analytics/daily", tags=["Analytics"])
def analytics_daily(
    start: Optional[str] = None,
    end: Optional[str] = None,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    q = db.query(models.Order).filter(
        models.Order.owner_id == current_user.shop_owner_id,
    )
    if start:
        q = q.filter(models.Order.date >= start)
    if end:
        q = q.filter(models.Order.date <= end)
    orders = q.all()
    return _daily_aggregates(orders)


@app.get("/analytics/monthly", tags=["Analytics"])
def analytics_monthly(
    current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)
):
    orders = db.query(models.Order).filter(
        models.Order.owner_id == current_user.shop_owner_id,
        models.Order.date != None
    ).all()
    investments = db.query(models.Investment).filter(
        models.Investment.owner_id == current_user.shop_owner_id,
        models.Investment.date != None
    ).all()
    return _monthly_aggregates(orders, investments)


from pydantic import BaseModel, Field
from typing import List, Literal

class PricingRecommendation(BaseModel):
    product_id: int
    sku: str | None
    name: str
    current_price: float
    unit_cost: float
    suggested_price: float
    action: Literal["increase", "decrease"]
    reason: str

class PricingRecommendationList(BaseModel):
    recommendations: List[PricingRecommendation]

@app.get("/analytics/pricing-recommendations", tags=["Analytics"])
def analytics_pricing_recommendations(
    current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)
):
    import os
    import json
    import datetime
    from sqlalchemy import func
    from google import genai
    from google.genai import types
    from fastapi import HTTPException
    
    thirty_days_ago = (datetime.datetime.utcnow() - datetime.timedelta(days=30)).strftime("%Y-%m-%d")
    
    # Fetch recent sales velocities
    sales_data = db.query(
        models.Order.product_id,
        models.Order.product_name,
        func.sum(models.Order.quantity).label("units_sold")
    ).filter(
        models.Order.owner_id == current_user.shop_owner_id,
        models.Order.status == "delivered",
        models.Order.date >= thirty_days_ago
    ).group_by(models.Order.product_id, models.Order.product_name).all()
    
    velocity_by_sku = {r.product_id: r.units_sold for r in sales_data if r.product_id}
    velocity_by_name = {r.product_name.lower(): r.units_sold for r in sales_data if r.product_name}
    
    # Fetch all products
    products = db.query(models.Product).filter(models.Product.owner_id == current_user.shop_owner_id).all()
    
    catalog_data = []
    for p in products:
        units_sold = 0
        if p.sku and p.sku in velocity_by_sku:
            units_sold = velocity_by_sku[p.sku]
        elif p.name and p.name.lower() in velocity_by_name:
            units_sold = velocity_by_name[p.name.lower()]
            
        if p.sell_price and p.sell_price > 0:
            catalog_data.append({
                "product_id": p.id,
                "sku": p.sku,
                "name": p.name,
                "current_price": p.sell_price,
                "unit_cost": p.unit_cost or 0.0,
                "current_stock": p.current_stock or 0,
                "units_sold_last_30_days": units_sold
            })

    if not catalog_data:
        return []

    # Get API key
    api_key = os.getenv("GOOGLE_AI_STUDIO_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GOOGLE_AI_STUDIO_KEY is not set in the environment. Please add it to your .env file.")
        
    client = genai.Client(api_key=api_key)
    
    prompt = f"""
    You are an expert AI retail pricing strategist.
    I will provide you with the inventory and recent sales data of my catalog.
    Analyze the data and recommend pricing changes (increase or decrease) for products that need attention.
    
    Rules for your analysis:
    - Increase price (+5% to +10%) if a product has high demand (high units_sold) but low stock. This maximizes margin before a stockout.
    - Decrease price (-10% to -20%) if a product is dead stock (0 or very low sales) but has a high stock buffer. This liquidates capital.
    - DO NOT recommend a decrease if the new price would be below the unit_cost (always maintain at least a 5% gross margin).
    - Return ONLY recommendations for products that genuinely need a change. Do not return recommendations if the price is fine.
    - Limit your response to the top 10 most critical recommendations.
    
    IMPORTANT: You must return a strict JSON object with a single key 'recommendations' that contains a list of objects.
    Each object must have the following keys: product_id (int), sku (string or null), name (string), current_price (float), unit_cost (float), suggested_price (float), action ("increase" or "decrease"), reason (string).
    
    Catalog Data:
    {json.dumps(catalog_data, default=str)}
    """
    
    try:
        model_name = os.getenv("GEMMA_MODEL", "gemma-2-27b-it")
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2,
            ),
        )
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
            
        result_data = json.loads(text.strip())
        return result_data.get("recommendations", [])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM Error: {str(e)}")


@app.get("/analytics/restock-plan", tags=["Analytics"])
def analytics_restock_plan(
    horizon_days: int = Query(default=14, ge=7, le=60),
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Weekly purchase plan: sales velocity + current stock + reorder level.

    All quantities are computed deterministically in Python; Gemma 4 only
    writes the advisory summary (same KPI-first approach as /ai/insights).
    """
    import math

    thirty_days_ago = (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d")

    sales_data = db.query(
        models.Order.product_id,
        models.Order.product_name,
        func.sum(models.Order.quantity).label("units_sold"),
    ).filter(
        models.Order.owner_id == current_user.shop_owner_id,
        models.Order.status == "delivered",
        models.Order.date >= thirty_days_ago,
    ).group_by(models.Order.product_id, models.Order.product_name).all()

    velocity_by_sku = {r.product_id: r.units_sold for r in sales_data if r.product_id}
    velocity_by_name = {r.product_name.lower(): r.units_sold for r in sales_data if r.product_name}

    products = db.query(models.Product).filter(
        models.Product.owner_id == current_user.shop_owner_id
    ).all()

    if not products:
        return {"plan": [], "totals": {}, "ai_summary": "No products found. Add products with stock levels to generate a restock plan."}

    today = datetime.utcnow().date()
    plan = []
    for p in products:
        units_30d = 0
        if p.sku and p.sku in velocity_by_sku:
            units_30d = velocity_by_sku[p.sku]
        elif p.name and p.name.lower() in velocity_by_name:
            units_30d = velocity_by_name[p.name.lower()]

        stock = p.current_stock or 0
        reorder = p.reorder_level or 0
        daily_velocity = units_30d / 30.0

        if daily_velocity > 0:
            days_left = stock / daily_velocity
            stockout_date = (today + timedelta(days=int(days_left))).strftime("%Y-%m-%d")
        else:
            days_left = None
            stockout_date = None

        predicted_demand = math.ceil(daily_velocity * horizon_days)
        # Cover the horizon's demand plus the safety buffer, minus what's on hand.
        suggested_qty = max(0, predicted_demand + reorder - stock)

        if daily_velocity > 0 and (stock == 0 or days_left <= 3):
            urgency = "critical"
        elif daily_velocity > 0 and (days_left <= 7 or stock <= reorder):
            urgency = "soon"
        elif daily_velocity == 0 and stock > reorder:
            urgency = "overstocked"
        else:
            urgency = "healthy"

        plan.append({
            "product_id": p.id,
            "sku": p.sku,
            "name": p.name,
            "supplier": p.supplier,
            "current_stock": stock,
            "reorder_level": reorder,
            "units_sold_30d": units_30d,
            "daily_velocity": round(daily_velocity, 2),
            "days_of_stock_left": round(days_left, 1) if days_left is not None else None,
            "stockout_date": stockout_date,
            "predicted_demand": predicted_demand,
            "suggested_order_qty": suggested_qty,
            "estimated_cost": _round_money(suggested_qty * (p.unit_cost or 0)),
            "urgency": urgency,
        })

    urgency_rank = {"critical": 0, "soon": 1, "healthy": 2, "overstocked": 3}
    plan.sort(key=lambda i: (urgency_rank[i["urgency"]], -(i["suggested_order_qty"] or 0)))

    buy_items = [i for i in plan if i["suggested_order_qty"] > 0]
    totals = {
        "horizon_days": horizon_days,
        "items_to_restock": len(buy_items),
        "total_units": sum(i["suggested_order_qty"] for i in buy_items),
        "total_cost": _round_money(sum(i["estimated_cost"] for i in buy_items)),
    }

    s = db.query(models.UserSettings).filter(
        models.UserSettings.user_id == current_user.shop_owner_id
    ).first()
    api_key = s.gemini_api_key if s else None

    # Keep the Gemma context compact: urgent/buy items in full, the rest summarized.
    ai_context = {
        "horizon_days": horizon_days,
        "plan": [
            {k: i[k] for k in ("name", "current_stock", "units_sold_30d", "days_of_stock_left", "stockout_date", "suggested_order_qty", "estimated_cost", "urgency")}
            for i in plan
        ][:20],
        "totals": totals,
    }
    ai_summary = ai_service.generate_restock_plan(
        current_user.business_name,
        ai_context,
        api_key=api_key,
    )

    return {"plan": plan, "totals": totals, "ai_summary": ai_summary}


@app.get("/analytics/investments-by-category", tags=["Analytics"])
def analytics_investments_by_category(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    items = db.query(models.Investment).filter(models.Investment.owner_id == current_user.shop_owner_id).all()
    categories: dict = defaultdict(lambda: {"category": "", "amount": 0.0, "entries": 0})
    for item in items:
        key = item.category or "Other"
        categories[key]["category"] = key
        categories[key]["amount"] += item.amount or 0
        categories[key]["entries"] += 1
    return [
        {**row, "amount": _round_money(row["amount"])}
        for row in sorted(categories.values(), key=lambda value: value["amount"], reverse=True)
    ]


@app.get("/analytics/top-customers", tags=["Analytics"])
def analytics_top_customers(
    limit: int = Query(default=10, ge=1, le=50),
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    orders = db.query(models.Order).filter(
        models.Order.owner_id == current_user.shop_owner_id,
        models.Order.customer_phone.isnot(None),
    ).all()

    return [
        {
            **customer,
            "orders": customer["total_orders"],
            "spent": customer["total_revenue"],
        }
        for customer in _customer_aggregates(orders)[:limit]
    ]


@app.get("/analytics/forecast", tags=["Analytics"])
def analytics_forecast(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    cutoff_90 = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
    orders = db.query(models.Order).filter(
        models.Order.owner_id == current_user.shop_owner_id,
        models.Order.date >= cutoff_90,
        models.Order.status.notin_(["cancelled", "returned"])
    ).all()
    
    if not orders:
        return {"daily_forecast": [], "top_products_forecast": [], "insights": "Not enough data for a forecast."}
    
    # 1. Daily Forecast (Simple Moving Average)
    daily_data = defaultdict(lambda: {"revenue": 0.0, "orders": 0})
    for o in orders:
        daily_data[o.date]["revenue"] += (o.amount or 0)
        daily_data[o.date]["orders"] += 1
        
    df = pd.DataFrame.from_dict(daily_data, orient="index").reset_index()
    df = df.rename(columns={"index": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    
    start_date = df["date"].min()
    end_date = datetime.now()
    full_range = pd.date_range(start=start_date, end=end_date)
    df = df.set_index("date").reindex(full_range, fill_value=0).reset_index()
    df = df.rename(columns={"index": "date"})
    
    df["revenue_ma"] = df["revenue"].rolling(window=7, min_periods=1).mean()
    df["orders_ma"] = df["orders"].rolling(window=7, min_periods=1).mean()
    
    last_rev_ma = df["revenue_ma"].iloc[-1] if not df.empty else 0
    last_ord_ma = df["orders_ma"].iloc[-1] if not df.empty else 0
    
    forecast_dates = pd.date_range(start=end_date + timedelta(days=1), periods=14)
    daily_forecast = []
    for d in forecast_dates:
        daily_forecast.append({
            "date": d.strftime("%Y-%m-%d"),
            "predicted_revenue": _round_money(last_rev_ma),
            "predicted_orders": max(1, int(last_ord_ma))
        })
        
    # 2. Product Level Forecast
    product_stats = defaultdict(lambda: {"units_30d": 0})
    cutoff_30 = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    for o in orders:
        if o.date >= cutoff_30 and o.product_name:
            product_stats[o.product_name]["units_30d"] += o.quantity
            
    top_products_forecast = []
    for p_name, stats in sorted(product_stats.items(), key=lambda x: x[1]["units_30d"], reverse=True)[:10]:
        predicted_units = max(1, int(stats["units_30d"] * (14 / 30)))
        top_products_forecast.append({
            "product_name": p_name,
            "predicted_demand_14d": predicted_units
        })
        
    forecast_payload = {
        "daily_forecast": daily_forecast,
        "top_products_forecast": top_products_forecast
    }
    
    s = db.query(models.UserSettings).filter(models.UserSettings.user_id == current_user.shop_owner_id).first()
    api_key = s.gemini_api_key if s else None
    
    insights = ai_service.generate_forecast_insights(
        current_user.business_name, 
        forecast_payload, 
        api_key=api_key
    )
    
    return {
        **forecast_payload,
        "insights": insights
    }


@app.get("/analytics/monthly-report", tags=["Analytics"])
def analytics_monthly_report(
    month: str = Query(..., description="YYYY-MM"),
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Generate a data payload for a specific month to be used in the PDF report."""
    # Find all orders and investments in this month
    start_date = f"{month}-01"
    
    # Simple way to get next month for end_date
    y, m = map(int, month.split('-'))
    if m == 12:
        end_date = f"{y+1}-01-01"
    else:
        end_date = f"{y}-{m+1:02d}-01"
        
    orders = db.query(models.Order).filter(
        models.Order.owner_id == current_user.shop_owner_id,
        models.Order.date >= start_date,
        models.Order.date < end_date
    ).all()
    
    investments = db.query(models.Investment).filter(
        models.Investment.owner_id == current_user.shop_owner_id,
        models.Investment.date >= start_date,
        models.Investment.date < end_date
    ).all()
    
    total_revenue = sum(_order_sign(o) * (o.amount or 0) for o in orders)
    total_orders = len(orders)
    
    total_investment = sum(i.amount for i in investments if i.amount)
    net_profit = total_revenue - total_investment
    
    # Daily breakdown for charting
    daily_map = defaultdict(float)
    for o in orders:
        daily_map[o.date] += _order_sign(o) * (o.amount or 0)
    
    daily_revenue = [{"date": k, "revenue": v} for k, v in sorted(daily_map.items())]
    
    # Top Products
    prod_map = defaultdict(lambda: {"qty": 0, "revenue": 0.0})
    for o in orders:
        sign = _order_sign(o)
        pname = o.product_name or "Unknown"
        qty = o.quantity or 1
        price = o.amount or 0
        prod_map[pname]["qty"] += sign * qty
        prod_map[pname]["revenue"] += sign * price
    
    top_products = [{"name": k, **v} for k, v in sorted(prod_map.items(), key=lambda x: x[1]["revenue"], reverse=True)[:5]]
    
    report_payload = {
        "month": month,
        "total_revenue": total_revenue,
        "total_orders": total_orders,
        "total_investment": total_investment,
        "net_profit": net_profit,
        "daily_revenue": daily_revenue,
        "top_products": top_products
    }
    
    # Generate AI executive summary
    s = db.query(models.UserSettings).filter(models.UserSettings.user_id == current_user.shop_owner_id).first()
    api_key = s.gemini_api_key if s else None
    
    summary = ai_service.generate_monthly_executive_summary(
        business_name=current_user.business_name,
        month_str=month,
        report_data=report_payload,
        api_key=api_key
    )
    
    report_payload["executive_summary"] = summary
    return report_payload


@app.get("/analytics/compare", tags=["Analytics"])
def analytics_compare(
    p1_start: str = Query(...),
    p1_end: str = Query(...),
    p2_start: str = Query(...),
    p2_end: str = Query(...),
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    uid = current_user.shop_owner_id

    # Fetch ALL orders for both periods
    all_orders_p1 = db.query(models.Order).filter(
        models.Order.owner_id == uid,
        models.Order.date >= p1_start,
        models.Order.date <= p1_end,
    ).all()
    all_orders_p2 = db.query(models.Order).filter(
        models.Order.owner_id == uid,
        models.Order.date >= p2_start,
        models.Order.date <= p2_end,
    ).all()

    # Investments
    inv_p1 = db.query(models.Investment).filter(
        models.Investment.owner_id == uid,
        models.Investment.date >= p1_start,
        models.Investment.date <= p1_end,
    ).all()
    inv_p2 = db.query(models.Investment).filter(
        models.Investment.owner_id == uid,
        models.Investment.date >= p2_start,
        models.Investment.date <= p2_end,
    ).all()

    def calc_metrics(orders, investments):
        rev = sum(_order_sign(o) * ((o.amount or 0) + (o.delivery_cost or 0)) for o in orders)
        cnt = len(orders)
        prof = sum(_order_sign(o) * ((o.amount or 0) - (o.product_cost or 0)) for o in orders)
        delivery = sum(_order_sign(o) * (o.delivery_cost or 0) for o in orders)
        prod_cost = sum(_order_sign(o) * (o.product_cost or 0) for o in orders)
        customers = len({o.customer_phone for o in orders if o.customer_phone})
        avg_order = _round_money(rev / cnt) if cnt else 0
        inv_total = sum(i.amount or 0 for i in investments)
        margin = round((prof / rev) * 100, 1) if rev else 0
        return {
            "revenue": _round_money(rev),
            "orders": cnt,
            "profit": _round_money(prof),
            "cogs": _round_money(prod_cost),
            "delivery": _round_money(delivery),
            "customers": customers,
            "avg_order_value": avg_order,
            "investment": _round_money(inv_total),
            "profit_margin": margin,
        }

    def top_products(orders, limit=5):
        products = defaultdict(lambda: {"name": "", "revenue": 0, "orders": 0})
        for o in orders:
            sign = _order_sign(o)
            key = o.product_name or "Unknown"
            products[key]["name"] = key
            products[key]["revenue"] += sign * (o.amount or 0)
            products[key]["orders"] += 1
        return sorted(products.values(), key=lambda x: x["revenue"], reverse=True)[:limit]

    def top_customers(orders, limit=5):
        custs = defaultdict(lambda: {"name": "", "phone": "", "revenue": 0, "orders": 0})
        for o in orders:
            sign = _order_sign(o)
            key = o.customer_phone or o.customer_name or "Unknown"
            custs[key]["name"] = o.customer_name or key
            custs[key]["phone"] = o.customer_phone or ""
            custs[key]["revenue"] += sign * (o.amount or 0)
            custs[key]["orders"] += 1
        return sorted(custs.values(), key=lambda x: x["revenue"], reverse=True)[:limit]

    def status_breakdown(all_orders):
        breakdown = defaultdict(int)
        for o in all_orders:
            breakdown[o.status or "unknown"] += 1
        return dict(breakdown)

    def daily_revenue(orders):
        daily = defaultdict(float)
        for o in orders:
            if o.date:
                daily[o.date] += _order_sign(o) * ((o.amount or 0) + (o.delivery_cost or 0))
        return [{"date": d, "revenue": _round_money(v)} for d, v in sorted(daily.items())]

    m1 = calc_metrics(all_orders_p1, inv_p1)
    m2 = calc_metrics(all_orders_p2, inv_p2)

    # Auto-generate insights
    def pct_change(curr, prev):
        if prev == 0:
            return 100.0 if curr > 0 else 0.0
        return round(((curr - prev) / abs(prev)) * 100, 1)

    insights = []
    metrics_config = [
        ("Revenue", "revenue", True),
        ("Profit", "profit", True),
        ("Orders", "orders", True),
        ("Customers", "customers", True),
        ("Average order value", "avg_order_value", True),
        ("Investment spending", "investment", False),
        ("Delivery cost", "delivery", False),
        ("Profit margin", "profit_margin", True),
    ]
    for label, key, higher_is_better in metrics_config:
        curr_val = m1[key]
        prev_val = m2[key]
        change = pct_change(curr_val, prev_val)
        if abs(change) < 0.1:
            insights.append({"text": f"{label} remained stable.", "type": "neutral"})
        else:
            direction = "increased" if change > 0 else "decreased"
            is_good = (change > 0) == higher_is_better
            suffix = "%" if key == "profit_margin" else ""
            insights.append({
                "text": f"{label} {direction} by {abs(change)}% (from {prev_val}{suffix} to {curr_val}{suffix}).",
                "type": "positive" if is_good else "negative",
            })

    return {
        "period1": {"label": f"{p1_start} to {p1_end}", **m1},
        "period2": {"label": f"{p2_start} to {p2_end}", **m2},
        "chartData": [
            {"metric": "Revenue", "Period A": m1["revenue"], "Period B": m2["revenue"]},
            {"metric": "Profit", "Period A": m1["profit"], "Period B": m2["profit"]},
            {"metric": "Orders", "Period A": m1["orders"], "Period B": m2["orders"]},
            {"metric": "Avg Order", "Period A": m1["avg_order_value"], "Period B": m2["avg_order_value"]},
            {"metric": "Investment", "Period A": m1["investment"], "Period B": m2["investment"]},
            {"metric": "Delivery", "Period A": m1["delivery"], "Period B": m2["delivery"]},
        ],
        "topProducts": {"period1": top_products(all_orders_p1), "period2": top_products(all_orders_p2)},
        "topCustomers": {"period1": top_customers(all_orders_p1), "period2": top_customers(all_orders_p2)},
        "statusBreakdown": {"period1": status_breakdown(all_orders_p1), "period2": status_breakdown(all_orders_p2)},
        "dailyTrend": {"period1": daily_revenue(all_orders_p1), "period2": daily_revenue(all_orders_p2)},
        "insights": insights,
    }



@app.get("/analytics/pl-statement", tags=["Analytics"])
def analytics_pl_statement(
    month: str = Query(..., description="YYYY-MM"),
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Generate a Profit & Loss statement for a given month."""
    start_date = f"{month}-01"
    
    y, m = map(int, month.split('-'))
    if m == 12:
        end_date = f"{y+1}-01-01"
    else:
        end_date = f"{y}-{m+1:02d}-01"
        
    orders = db.query(models.Order).filter(
        models.Order.owner_id == current_user.shop_owner_id,
        models.Order.date >= start_date,
        models.Order.date < end_date
    ).all()
    
    investments = db.query(models.Investment).filter(
        models.Investment.owner_id == current_user.shop_owner_id,
        models.Investment.date >= start_date,
        models.Investment.date < end_date
    ).all()
    
    # Separate cancelled/returned orders for the deduction line
    cancelled_returned = [o for o in orders if (o.status or "").strip().lower() in ("cancelled", "canceled", "returned")]
    cancelled_returned_amount = sum((o.amount or 0) + (o.delivery_cost or 0) for o in cancelled_returned)
    cancelled_returned_count = len(cancelled_returned)

    # Gross revenue uses _order_sign: delivered adds, cancelled/returned subtracts
    gross_revenue = sum(_order_sign(o) * ((o.amount or 0) + (o.delivery_cost or 0)) for o in orders)
    total_cogs = sum(_order_sign(o) * (o.product_cost or 0) for o in orders)
    delivery_costs = sum(_order_sign(o) * (o.delivery_cost or 0) for o in orders)
    
    gross_profit = gross_revenue - total_cogs - delivery_costs
    
    # Group OPEX by category
    opex_groups = {}
    total_opex = 0
    for i in investments:
        amt = i.amount or 0
        cat = i.category or "Other"
        opex_groups[cat] = opex_groups.get(cat, 0) + amt
        total_opex += amt
        
    operating_expenses = [{"category": k, "amount": v} for k, v in opex_groups.items()]
    
    net_profit = gross_profit - total_opex
    
    s = db.query(models.UserSettings).filter(models.UserSettings.user_id == current_user.shop_owner_id).first()
    api_key = s.gemini_api_key if s else None
    
    pl_data = {
        "period": month,
        "gross_revenue": gross_revenue,
        "total_cogs": total_cogs,
        "delivery_costs": delivery_costs,
        "gross_profit": gross_profit,
        "total_opex": total_opex,
        "net_profit": net_profit,
        "cancelled_returned_amount": _round_money(cancelled_returned_amount),
        "cancelled_returned_count": cancelled_returned_count,
    }
    
    summary = ai_service.generate_pl_summary(
        current_user.business_name,
        pl_data,
        api_key=api_key
    )
    
    return {
        **pl_data,
        "operating_expenses": operating_expenses,
        "summary": summary
    }


@app.get("/customers", tags=["Customers"])
def list_customers(
    inactive_days: int = Query(default=30, ge=1),
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    orders = db.query(models.Order).filter(models.Order.owner_id == current_user.shop_owner_id).all()
    return _customer_aggregates(orders, inactive_days)


# ----------------------------------------------------------------------------
# INVESTMENTS / EXPENSES
# ----------------------------------------------------------------------------

@app.get("/investments", tags=["Investments"])
def list_investments(
    category: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=50, ge=1, le=1000),
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    q = db.query(models.Investment).filter(models.Investment.owner_id == current_user.shop_owner_id)
    if category:
        q = q.filter(models.Investment.category == category)
    if start_date:
        q = q.filter(models.Investment.date >= start_date)
    if end_date:
        q = q.filter(models.Investment.date <= end_date)

    total = q.count()
    items = (
        q.order_by(models.Investment.date.desc(), models.Investment.id.desc())
        .offset((page - 1) * limit)
        .limit(limit)
        .all()
    )
    return {
        "items": [_investment_to_dict(item) for item in items],
        "total": total,
        "page": page,
        "pages": max(1, (total + limit - 1) // limit),
    }


@app.post("/investments", tags=["Investments"])
def create_investment(
    data: InvestmentCreate,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    item = models.Investment(
        owner_id=current_user.shop_owner_id,
        date=data.date,
        category=data.category or "Other",
        amount=data.amount,
        notes=_investment_notes(data),
    )
    db.add(item)
    _log_audit(
        db, owner_id=current_user.shop_owner_id, user=current_user,
        action="create", entity="investment", entity_id=None,
        summary=f"Created investment \u09f3{data.amount:.0f} ({data.category})",
    )
    db.commit()
    db.refresh(item)
    _sync_shop_sheet_for_user(current_user.shop_owner_id, db)
    return _investment_to_dict(item)


@app.put("/investments/{investment_id}", tags=["Investments"])
def update_investment(
    investment_id: int,
    data: InvestmentUpdate,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    item = db.query(models.Investment).filter(
        models.Investment.id == investment_id,
        models.Investment.owner_id == current_user.shop_owner_id,
    ).first()
    if not item:
        raise HTTPException(status_code=404, detail="Investment entry not found")

    changed = {k: v for k, v in data.dict(exclude_unset=True).items()}
    old_vals = {k: getattr(item, k if k != 'notes_quantity' else 'notes') for k in changed}
    for field, value in changed.items():
        if field == "notes_quantity":
            if data.notes is None:
                item.notes = value
        elif field == "notes":
            item.notes = value
        else:
            setattr(item, field, value)

    item.updated_at = datetime.utcnow()
    _log_audit(
        db, owner_id=current_user.shop_owner_id, user=current_user,
        action="update", entity="investment", entity_id=item.id,
        summary=f"Updated investment #{item.id}",
        details={"old": old_vals, "new": changed},
    )
    db.commit()
    db.refresh(item)
    _sync_shop_sheet_for_user(current_user.shop_owner_id, db)
    return _investment_to_dict(item)


@app.delete("/investments/{investment_id}", tags=["Investments"])
def delete_investment(
    investment_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    item = db.query(models.Investment).filter(
        models.Investment.id == investment_id,
        models.Investment.owner_id == current_user.shop_owner_id,
    ).first()
    if not item:
        raise HTTPException(status_code=404, detail="Investment entry not found")

    _log_audit(
        db, owner_id=current_user.shop_owner_id, user=current_user,
        action="delete", entity="investment", entity_id=item.id,
        summary=f"Deleted investment #{item.id} (\u09f3{item.amount:.0f})",
    )
    db.delete(item)
    db.commit()
    _sync_shop_sheet_for_user(current_user.shop_owner_id, db)
    return {"message": "Investment entry deleted"}


# ----------------------------------------------------------------------------
# PRODUCTS / INVENTORY
# ----------------------------------------------------------------------------

@app.get("/products", tags=["Products"])
def list_products(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    products = (
        db.query(models.Product)
        .filter(models.Product.owner_id == current_user.shop_owner_id)
        .order_by(models.Product.name.asc())
        .all()
    )
    return [_product_to_dict(product) for product in products]


@app.post("/products", tags=["Products"])
def create_product(
    data: ProductCreate,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if data.sku:
        existing = db.query(models.Product).filter(
            models.Product.owner_id == current_user.shop_owner_id,
            models.Product.sku == data.sku,
        ).first()
        if existing:
            raise HTTPException(status_code=400, detail="SKU already exists")

    product = models.Product(
        owner_id=current_user.shop_owner_id,
        sku=data.sku or None,
        name=data.name,
        category=data.category or None,
        supplier=data.supplier or None,
        unit_cost=data.unit_cost or 0,
        sell_price=data.sell_price or 0,
        current_stock=data.current_stock or 0,
        reorder_level=data.reorder_level or 0,
        notes=data.notes or None,
    )
    db.add(product)
    _log_audit(
        db, owner_id=current_user.shop_owner_id, user=current_user,
        action="create", entity="product", entity_id=None,
        summary=f"Created product {data.name}",
    )
    db.commit()
    db.refresh(product)
    _sync_shop_sheet_for_user(current_user.shop_owner_id, db)
    return _product_to_dict(product)


@app.put("/products/{product_id}", tags=["Products"])
def update_product(
    product_id: int,
    data: ProductUpdate,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    product = db.query(models.Product).filter(
        models.Product.id == product_id,
        models.Product.owner_id == current_user.shop_owner_id,
    ).first()
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    if data.sku:
        duplicate = db.query(models.Product).filter(
            models.Product.owner_id == current_user.shop_owner_id,
            models.Product.sku == data.sku,
            models.Product.id != product_id,
        ).first()
        if duplicate:
            raise HTTPException(status_code=400, detail="SKU already exists")

    changed = {k: v for k, v in data.dict(exclude_unset=True).items()}
    old_vals = {k: getattr(product, k) for k in changed}
    for field, value in changed.items():
        setattr(product, field, value)

    product.updated_at = datetime.utcnow()
    _log_audit(
        db, owner_id=current_user.shop_owner_id, user=current_user,
        action="update", entity="product", entity_id=product.id,
        summary=f"Updated product {product.name}",
        details={"old": old_vals, "new": changed},
    )
    db.commit()
    db.refresh(product)
    _sync_shop_sheet_for_user(current_user.shop_owner_id, db)
    return _product_to_dict(product)


@app.delete("/products/{product_id}", tags=["Products"])
def delete_product(
    product_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    product = db.query(models.Product).filter(
        models.Product.id == product_id,
        models.Product.owner_id == current_user.shop_owner_id,
    ).first()
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    _log_audit(
        db, owner_id=current_user.shop_owner_id, user=current_user,
        action="delete", entity="product", entity_id=product.id,
        summary=f"Deleted product {product.name}",
    )
    db.delete(product)
    db.commit()
    _sync_shop_sheet_for_user(current_user.shop_owner_id, db)
    return {"message": "Product deleted"}


# ═══════════════════════════════════════════════════════════════════════════════
# AI ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

def _build_ai_context(
    business_name: str,
    orders: list[models.Order],
    investments: list[models.Investment],
    products: list[models.Product],
) -> dict:
    """Assemble cross-functional business data for AI query and report features."""
    total_rev = sum(_order_sign(o) * (o.amount or 0.0) for o in orders)
    total_profit = sum(_order_sign(o) * (o.profit or 0.0) for o in orders)
    total_delivery = sum(_order_sign(o) * (o.delivery_cost or 0.0) for o in orders)
    n = len(orders)
    unique_phones = {o.customer_phone for o in orders if o.customer_phone}

    today = datetime.now().strftime("%Y-%m-%d")
    today_orders = [o for o in orders if o.date == today]
    cutoff_30 = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    cutoff_7 = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    last_7_orders = [o for o in orders if o.date and o.date >= cutoff_7]

    at_risk = sum(
        1
        for phone in unique_phones
        if any(o.customer_phone == phone and o.date and o.date < cutoff_30 for o in orders)
        and not any(o.customer_phone == phone and o.date and o.date >= cutoff_30 for o in orders)
    )

    product_stats: dict[str, dict] = defaultdict(lambda: {"orders": 0, "revenue": 0.0})
    for order in orders:
        if not order.product_name:
            continue
        sign = _order_sign(order)
        row = product_stats[order.product_name]
        row["orders"] += 1
        row["revenue"] += sign * (order.amount or 0)
    top_products = [
        f"{name} ({stats['orders']} orders, ৳{_round_money(stats['revenue']):,.0f} revenue)"
        for name, stats in sorted(
            product_stats.items(), key=lambda item: item[1]["revenue"], reverse=True
        )[:5]
    ]

    top_customers = [
        f"{c['name']} ({c['total_orders']} orders, ৳{c['total_revenue']:,.0f} spent, last order {c['last_order'] or 'n/a'})"
        for c in _customer_aggregates(orders)[:5]
    ]

    investment_by_category: dict[str, float] = defaultdict(float)
    for item in investments:
        investment_by_category[item.category or "Other"] += item.amount or 0
    investment_breakdown = [
        f"{category}: ৳{_round_money(amount):,.0f}"
        for category, amount in sorted(
            investment_by_category.items(), key=lambda item: item[1], reverse=True
        )
    ]

    total_investment = sum((i.amount or 0.0) for i in investments)
    aov = total_rev / n if n else 0
    margin = (total_profit / total_rev * 100) if total_rev else 0
    dates = [o.date for o in orders if o.date]

    out_of_stock = [p.name for p in products if p.current_stock <= 0]
    low_stock = [p.name for p in products if 0 < p.current_stock <= p.reorder_level]
    total_stock_value = sum(p.current_stock * p.unit_cost for p in products if p.current_stock > 0)

    return {
        "business_name": business_name,
        "total_orders": n,
        "delivered_orders": len([o for o in orders if o.status == "delivered"]),
        "pending_orders": len([o for o in orders if o.status == "pending"]),
        "returned_orders": len([o for o in orders if o.status == "returned"]),
        "cancelled_orders": len([o for o in orders if o.status == "cancelled"]),
        "return_rate_pct": round(len([o for o in orders if o.status == "returned"]) / n * 100, 1)
        if n
        else 0,
        "total_revenue": _round_money(total_rev),
        "total_profit": _round_money(total_profit),
        "total_delivery_cost": _round_money(total_delivery),
        "avg_order_value": _round_money(aov),
        "profit_margin_pct": round(margin, 1),
        "net_cash_flow": _round_money(total_profit - total_investment),
        "unique_customers": len(unique_phones),
        "at_risk_customers": at_risk,
        "today_orders": len(today_orders),
        "today_revenue": _round_money(sum(_order_sign(o) * (o.amount or 0.0) for o in today_orders)),
        "today_profit": _round_money(sum(_order_sign(o) * (o.profit or 0.0) for o in today_orders)),
        "last_7_days_orders": len(last_7_orders),
        "last_7_days_revenue": _round_money(sum(_order_sign(o) * (o.amount or 0.0) for o in last_7_orders)),
        "last_7_days_profit": _round_money(sum(_order_sign(o) * (o.profit or 0.0) for o in last_7_orders)),
        "last_sale_date": max(dates) if dates else "None",
        "top_products": top_products or ["No product-level sales data tracked"],
        "top_customers": top_customers or ["No customer phone data tracked"],
        "total_products_in_inventory": len(products),
        "total_stock_value": _round_money(total_stock_value),
        "out_of_stock_products": ", ".join(out_of_stock) if out_of_stock else "None",
        "low_stock_products": ", ".join(low_stock) if low_stock else "None",
        "total_investment": _round_money(total_investment),
        "investment_by_category": investment_breakdown or ["No investment entries"],
        "data_gaps": (
            "CLV, CAC, churn rate, sentiment analysis, and per-channel acquisition costs "
            "are not tracked in this system."
        ),
    }


def _resolve_ai_api_key(user_id: int, db: Session) -> Optional[str]:
    """Prefer per-user key from settings, then fall back to server env."""
    settings = db.query(models.UserSettings).filter(
        models.UserSettings.user_id == user_id
    ).first()
    if settings and settings.gemini_api_key:
        return settings.gemini_api_key.strip()
    env_key = os.getenv("GOOGLE_AI_STUDIO_KEY") or os.getenv("GEMMA_API_KEY")
    return env_key.strip() if env_key else None


@app.post("/ai/insights", tags=["AI"])
def get_ai_insights(
    current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)
):
    orders = db.query(models.Order).filter(models.Order.owner_id == current_user.shop_owner_id).all()
    if not orders:
        return {"insights": "কোনো অর্ডার ডেটা পাওয়া যায়নি। প্রথমে কিছু অর্ডার যোগ করুন অথবা Excel ফাইল আপলোড করুন।"}

    products = db.query(models.Product).filter(models.Product.owner_id == current_user.shop_owner_id).all()
    investments = db.query(models.Investment).filter(models.Investment.owner_id == current_user.shop_owner_id).all()
    context = _build_ai_context(
        business_name=current_user.business_name,
        orders=orders,
        investments=investments,
        products=products,
    )

    api_key = _resolve_ai_api_key(current_user.id, db)
    logger.info(
        "AI insights requested by user=%s orders=%s api_key_present=%s",
        current_user.email,
        context["total_orders"],
        bool(api_key),
    )

    insights = ai_service.generate_insights(
        business_name=current_user.business_name,
        context_data=context,
        api_key=api_key,
    )
    return {"insights": insights}


@app.post("/ai/query", tags=["AI"])
def ai_query(
    req: AIQueryRequest,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    orders = db.query(models.Order).filter(models.Order.owner_id == current_user.shop_owner_id).all()
    products = db.query(models.Product).filter(models.Product.owner_id == current_user.shop_owner_id).all()
    investments = db.query(models.Investment).filter(models.Investment.owner_id == current_user.shop_owner_id).all()

    context = _build_ai_context(
        business_name=current_user.business_name,
        orders=orders,
        investments=investments,
        products=products,
    )

    api_key = _resolve_ai_api_key(current_user.id, db)
    logger.info(
        "AI query requested by user=%s question=%r today_revenue=%s today_orders=%s api_key_present=%s",
        current_user.email,
        req.question,
        context["today_revenue"],
        context["today_orders"],
        bool(api_key),
    )

    answer = ai_service.answer_query(
        business_name=current_user.business_name,
        question=req.question,
        context_data=context,
        api_key=api_key,
    )

    logger.info(
        "AI query answered for user=%s answer_preview=%s",
        current_user.email,
        (answer[:200] + "…") if len(answer) > 200 else answer,
    )
    return {"answer": answer, "context": context}


class WinBackPreviewRequest(BaseModel):
    discount_percent: int = 10
    inactive_days: int = 30
    language: str = "bn"
    custom_note: str = ""
    limit: int = 50
    customer_phones: Optional[list[str]] = None

@app.post("/customers/win-back/preview", tags=["Customers"])
def win_back_preview(
    req: WinBackPreviewRequest,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    orders = db.query(models.Order).filter(models.Order.owner_id == current_user.shop_owner_id).all()
    all_customers = _customer_aggregates(orders, req.inactive_days)
    
    # Filter to only at-risk
    at_risk = [c for c in all_customers if c.get("is_at_risk")]
    
    if req.customer_phones:
        # User selected specific phones
        selected = set(req.customer_phones)
        targets = [c for c in at_risk if c.get("phone") in selected]
    else:
        # Default to top 'limit' customers by revenue
        targets = at_risk[:req.limit]

    api_key = _resolve_ai_api_key(current_user.id, db)
    return ai_service.generate_winback_messages(
        business_name=current_user.business_name,
        campaign=req.dict(),
        target_customers=targets,
        api_key=api_key,
    )

# ═══════════════════════════════════════════════════════════════════════════════
# GOOGLE SHEETS
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/sheets/connect", tags=["Google Sheets"])
def sheets_connect(
    req: SheetsConnectRequest,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if current_user.role != "owner":
        raise HTTPException(status_code=403, detail="Only the owner can connect Google Sheets.")
    s = db.query(models.UserSettings).filter(models.UserSettings.user_id == current_user.shop_owner_id).first()
    if not s:
        s = models.UserSettings(user_id=current_user.shop_owner_id)
        db.add(s)
    s.google_sheet_id = req.sheet_id
    if req.credentials_json:
        s.google_credentials = req.credentials_json
    db.commit()
    return {"message": "Google Sheet connected successfully", "sheet_id": req.sheet_id}


@app.post("/sheets/sync", tags=["Google Sheets"])
def sheets_sync(
    current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)
):
    s = db.query(models.UserSettings).filter(models.UserSettings.user_id == current_user.shop_owner_id).first()
    if not s or not s.google_sheet_id:
        raise HTTPException(status_code=400, detail="No Google Sheet connected. Go to Settings first.")

    orders = db.query(models.Order).filter(models.Order.owner_id == current_user.shop_owner_id).all()
    investments = db.query(models.Investment).filter(models.Investment.owner_id == current_user.shop_owner_id).all()
    products = db.query(models.Product).filter(models.Product.owner_id == current_user.shop_owner_id).all()

    result = sheets_service.sync_shop_to_sheet(
        s.google_sheet_id,
        orders=[
            {
                "id": o.id,
                "date": o.date,
                "consignment_id": o.consignment_id,
                "order_id": o.order_id,
                "customer_name": o.customer_name,
                "customer_phone": o.customer_phone,
                "product_name": o.product_name,
                "amount": o.amount,
                "delivery_cost": o.delivery_cost,
                "product_cost": o.product_cost,
                "profit": o.profit,
                "status": o.status,
                "notes": o.notes,
            }
            for o in orders
        ],
        investments=[_investment_to_dict(item) for item in investments],
        customers=_customer_aggregates(orders),
        daily=_daily_aggregates(orders),
        products=[_product_to_dict(product) for product in products],
        credentials_json=s.google_credentials,
    )
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result.get("error", "Sync failed"))

    return {
        "synced_rows": result.get("synced_rows", 0),
        "sales_rows": result.get("sales_rows", 0),
        "investment_rows": result.get("investment_rows", 0),
        "inventory_rows": result.get("inventory_rows", 0),
        "message": (
            f"Synced {result.get('sales_rows', 0)} sales, "
            f"{result.get('investment_rows', 0)} investments, and "
            f"{result.get('inventory_rows', 0)} inventory rows to Google Sheets"
        ),
    }


@app.post("/sheets/import", tags=["Google Sheets"])
def sheets_import(
    current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)
):
    if current_user.role != "owner":
        raise HTTPException(status_code=403, detail="Only the owner can import from Google Sheets.")
    s = db.query(models.UserSettings).filter(models.UserSettings.user_id == current_user.shop_owner_id).first()
    if not s or not s.google_sheet_id:
        raise HTTPException(status_code=400, detail="No Google Sheet connected")

    result = sheets_service.import_shop_from_sheet(s.google_sheet_id, s.google_credentials)
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result.get("error", "Import failed"))

    imported_orders = 0
    updated_orders = 0
    imported_investments = 0
    imported_products = 0
    updated_products = 0

    for od in result["orders"]:
        existing = None
        if od.get("order_id"):
            existing = db.query(models.Order).filter(
                models.Order.owner_id == current_user.shop_owner_id,
                models.Order.order_id == od["order_id"],
            ).first()
        if not existing and od.get("consignment_id"):
            existing = db.query(models.Order).filter(
                models.Order.owner_id == current_user.shop_owner_id,
                models.Order.consignment_id == od["consignment_id"],
            ).first()
        if not existing:
            existing = db.query(models.Order).filter(
                models.Order.owner_id == current_user.shop_owner_id,
                models.Order.date == od.get("date"),
                models.Order.customer_phone == od.get("customer_phone"),
                models.Order.amount == od.get("amount", 0),
            ).first()

        payload = {
            "date": od.get("date"),
            "consignment_id": od.get("consignment_id") or None,
            "order_id": od.get("order_id") or None,
            "customer_name": od.get("customer_name") or None,
            "customer_phone": od.get("customer_phone") or None,
            "product_name": od.get("product_name") or None,
            "amount": od.get("amount", 0),
            "delivery_cost": od.get("delivery_cost", 0),
            "product_cost": od.get("product_cost", 0),
            "profit": od.get("profit", _profit(od.get("amount", 0), od.get("delivery_cost", 0), od.get("product_cost", 0))),
            "status": od.get("status", "delivered") or "delivered",
            "notes": od.get("notes") or None,
        }

        if existing:
            for field, value in payload.items():
                setattr(existing, field, value)
            existing.updated_at = datetime.utcnow()
            updated_orders += 1
        else:
            db.add(models.Order(owner_id=current_user.shop_owner_id, **payload))
            imported_orders += 1

    for item in result["investments"]:
        existing = db.query(models.Investment).filter(
            models.Investment.owner_id == current_user.shop_owner_id,
            models.Investment.date == item.get("date"),
            models.Investment.category == item.get("category", "Other"),
            models.Investment.amount == item.get("amount", 0),
            models.Investment.notes == (item.get("notes") or None),
        ).first()
        if existing:
            continue
        db.add(
            models.Investment(
                owner_id=current_user.shop_owner_id,
                date=item.get("date"),
                category=item.get("category", "Other"),
                amount=item.get("amount", 0),
                notes=item.get("notes") or None,
            )
        )
        imported_investments += 1

    for product_data in result["products"]:
        existing = None
        if product_data.get("sku"):
            existing = db.query(models.Product).filter(
                models.Product.owner_id == current_user.shop_owner_id,
                models.Product.sku == product_data["sku"],
            ).first()
        if not existing:
            existing = db.query(models.Product).filter(
                models.Product.owner_id == current_user.shop_owner_id,
                models.Product.name == product_data.get("name"),
                models.Product.category == (product_data.get("category") or None),
            ).first()

        payload = {
            "sku": product_data.get("sku") or None,
            "name": product_data.get("name"),
            "category": product_data.get("category") or None,
            "supplier": product_data.get("supplier") or None,
            "unit_cost": product_data.get("unit_cost", 0),
            "sell_price": product_data.get("sell_price", 0),
            "current_stock": product_data.get("current_stock", 0),
            "reorder_level": product_data.get("reorder_level", 0),
            "notes": product_data.get("notes") or None,
        }

        if existing:
            for field, value in payload.items():
                setattr(existing, field, value)
            existing.updated_at = datetime.utcnow()
            updated_products += 1
        else:
            db.add(models.Product(owner_id=current_user.shop_owner_id, **payload))
            imported_products += 1

    db.commit()
    _sync_shop_sheet_for_user(current_user.shop_owner_id, db)
    return {
        "imported": imported_orders,
        "updated_orders": updated_orders,
        "imported_investments": imported_investments,
        "imported_products": imported_products,
        "updated_products": updated_products,
        "message": (
            f"Imported {imported_orders} sales, updated {updated_orders}, "
            f"added {imported_investments} investments and {imported_products} products"
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# AUDIT LOG
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/audit-log", tags=["Audit Log"])
def list_audit_log(
    entity: Optional[str] = None,
    action: Optional[str] = None,
    search: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=50, ge=1, le=200),
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if current_user.role != "owner":
        raise HTTPException(status_code=403, detail="Only the owner can view the audit log.")

    q = db.query(models.AuditLog).filter(models.AuditLog.owner_id == current_user.shop_owner_id)

    if entity:
        q = q.filter(models.AuditLog.entity == entity)
    if action:
        q = q.filter(models.AuditLog.action == action)
    if search:
        like = f"%{search}%"
        q = q.filter(
            models.AuditLog.summary.ilike(like)
            | models.AuditLog.user_email.ilike(like)
        )
    if start_date:
        q = q.filter(models.AuditLog.created_at >= start_date)
    if end_date:
        q = q.filter(models.AuditLog.created_at <= end_date + " 23:59:59")

    total = q.count()
    items = (
        q.order_by(models.AuditLog.created_at.desc())
        .offset((page - 1) * limit)
        .limit(limit)
        .all()
    )

    return {
        "items": [
            {
                "id": entry.id,
                "user_email": entry.user_email,
                "action": entry.action,
                "entity": entry.entity,
                "entity_id": entry.entity_id,
                "summary": entry.summary,
                "details": json.loads(entry.details) if entry.details else None,
                "created_at": entry.created_at.isoformat() if entry.created_at else None,
            }
            for entry in items
        ],
        "total": total,
        "page": page,
        "pages": max(1, (total + limit - 1) // limit),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/settings", tags=["Settings"])
def get_settings(
    current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)
):
    s = db.query(models.UserSettings).filter(models.UserSettings.user_id == current_user.shop_owner_id).first()
    return {
        "delivery_cost_per_order": s.delivery_cost_per_order if s else DELIVERY_CHARGE,
        "notification_email": (s and s.notification_email) or current_user.email,
        "google_sheet_id": s.google_sheet_id if s else None,
        "google_credentials_configured": bool(s and s.google_credentials),
        "report_time": s.report_time if (s and s.report_time) else "20:00",
    }


@app.put("/settings", tags=["Settings"])
def update_settings(
    data: SettingsUpdate,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if current_user.role != "owner":
        raise HTTPException(status_code=403, detail="Only the owner can update settings.")
    s = db.query(models.UserSettings).filter(models.UserSettings.user_id == current_user.shop_owner_id).first()
    if not s:
        s = models.UserSettings(user_id=current_user.shop_owner_id)
        db.add(s)
    if data.delivery_cost_per_order is not None:
        s.delivery_cost_per_order = data.delivery_cost_per_order
    if data.notification_email is not None:
        s.notification_email = data.notification_email
    if data.google_sheet_id is not None:
        s.google_sheet_id = data.google_sheet_id
    if data.report_time is not None:
        s.report_time = data.report_time
    _log_audit(
        db, owner_id=current_user.shop_owner_id, user=current_user,
        action="update", entity="settings", entity_id=None,
        summary="Updated settings",
        details={k: v for k, v in data.dict(exclude_unset=True).items()},
    )
    db.commit()
    return {"message": "Settings updated"}


# ─── Legacy insight route ─────────────────────────────────────────────────────
@app.get("/insights/", include_in_schema=False)
def insights_legacy(current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    return analytics_summary(current_user, db)


@app.get("/", tags=["Health"])
def health_check():
    return {"status": "ok", "app": "myShop AI", "version": "2.0.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
@app.post("/investments/bulk-upload", tags=["Investments"])
async def bulk_upload_investments(
    file: UploadFile = File(...),
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not file.filename.lower().endswith((".xlsx", ".xls", ".csv")):
        raise HTTPException(status_code=400, detail="Only Excel (.xlsx/.xls) or CSV files are accepted")

    content = await file.read()
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {MAX_UPLOAD_MB:.0f} MB limit")

    try:
        df = (
            pd.read_csv(io.BytesIO(content))
            if file.filename.lower().endswith(".csv")
            else pd.read_excel(io.BytesIO(content))
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not parse file: {exc}")

    if "Amount" not in df.columns:
        raise HTTPException(status_code=422, detail="File must contain an 'Amount' column")
    if "Date" not in df.columns:
        raise HTTPException(status_code=422, detail="File must contain a 'Date' column")
    if "Category" not in df.columns:
        raise HTTPException(status_code=422, detail="File must contain a 'Category' column")

    df["Amount"] = df["Amount"].apply(_clean_amount)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df[df["Amount"] > 0].dropna(subset=["Date", "Category"]).copy()

    imported = 0

    for _, row in df.iterrows():
        date_str = row["Date"].strftime("%Y-%m-%d")
        amount = float(row["Amount"])
        category = str(row["Category"]).strip()
        notes = str(row.get("Notes", row.get("Notes / Quantity", ""))).strip()

        investment = models.Investment(
            owner_id=current_user.shop_owner_id,
            date=date_str,
            category=category,
            amount=amount,
            notes=notes or None,
        )
        db.add(investment)
        imported += 1

    db.commit()
    _sync_shop_sheet_for_user(current_user.shop_owner_id, db)
    return {"message": f"Successfully imported {imported} investments."}


@app.post("/products/bulk-upload", tags=["Products"])
async def bulk_upload_products(
    file: UploadFile = File(...),
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not file.filename.lower().endswith((".xlsx", ".xls", ".csv")):
        raise HTTPException(status_code=400, detail="Only Excel (.xlsx/.xls) or CSV files are accepted")

    content = await file.read()
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {MAX_UPLOAD_MB:.0f} MB limit")

    try:
        df = (
            pd.read_csv(io.BytesIO(content))
            if file.filename.lower().endswith(".csv")
            else pd.read_excel(io.BytesIO(content))
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not parse file: {exc}")

    if "Name" not in df.columns:
        raise HTTPException(status_code=422, detail="File must contain a 'Name' column")

    df = df.dropna(subset=["Name"]).copy()
    imported = 0
    updated = 0

    for _, row in df.iterrows():
        name = str(row["Name"]).strip()
        if not name or str(name).lower() == 'nan':
            continue
            
        sku = str(row.get("SKU", "")).strip()
        if str(sku).lower() == 'nan': sku = ""
        category = str(row.get("Category", "")).strip()
        if str(category).lower() == 'nan': category = ""
        supplier = str(row.get("Supplier", "")).strip()
        if str(supplier).lower() == 'nan': supplier = ""
        unit_cost = _clean_amount(row.get("Unit Cost", 0))
        sell_price = _clean_amount(row.get("Sell Price", 0))
        current_stock = int(_clean_amount(row.get("Current Stock", row.get("Stock", 0))))
        reorder_level = int(_clean_amount(row.get("Reorder Level", 0)))
        notes = str(row.get("Notes", "")).strip()
        if str(notes).lower() == 'nan': notes = ""

        existing = None
        if sku:
            existing = db.query(models.Product).filter(
                models.Product.owner_id == current_user.shop_owner_id,
                models.Product.sku == sku
            ).first()
        
        if not existing:
            existing = db.query(models.Product).filter(
                models.Product.owner_id == current_user.shop_owner_id,
                func.lower(models.Product.name) == name.lower()
            ).first()

        if existing:
            if sku: existing.sku = sku
            if category: existing.category = category
            if supplier: existing.supplier = supplier
            existing.unit_cost = unit_cost
            existing.sell_price = sell_price
            existing.current_stock = current_stock
            existing.reorder_level = reorder_level
            if notes: existing.notes = notes
            existing.updated_at = datetime.utcnow()
            updated += 1
        else:
            product = models.Product(
                owner_id=current_user.shop_owner_id,
                sku=sku or None,
                name=name,
                category=category or None,
                supplier=supplier or None,
                unit_cost=unit_cost,
                sell_price=sell_price,
                current_stock=current_stock,
                reorder_level=reorder_level,
                notes=notes or None,
            )
            db.add(product)
            imported += 1

    db.commit()
    _sync_shop_sheet_for_user(current_user.shop_owner_id, db)
    return {"message": f"Successfully added {imported} and updated {updated} products."}


# ═══════════════════════════════════════════════════════════════════════════════
# PLATFORM ADMIN
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/admin/shops", tags=["Admin"])
def get_all_shops(
    current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Only platform admins can view shops")
    
    users = db.query(models.User).all()
    shops = []
    for u in users:
        # Avoid sending password hash
        shops.append({
            "id": u.id,
            "email": u.email,
            "business_name": u.business_name,
            "role": u.role,
            "is_active": u.is_active,
            "created_at": u.created_at.isoformat() if u.created_at else None
        })
    return shops

@app.delete("/admin/shops/{user_id}", tags=["Admin"])
def delete_shop(
    user_id: int,
    current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Only platform admins can delete shops")
    
    if user_id == current_user.id:
        raise HTTPException(status_code=400, detail="You cannot delete yourself")
        
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Shop not found")
        
    db.delete(user)
    db.commit()
    return {"message": f"Shop {user.business_name} deleted successfully"}


@app.post("/test-email", tags=["Test"])
def test_email(
    to_email: str,
    subject: str,
    body: str,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    import email_service
    
    success = email_service.send_email(to_email, subject, body)
    if success:
        return {"message": f"Email sent successfully to {to_email}!"}
    else:
        raise HTTPException(status_code=500, detail="Failed to send email. Check logs.")
