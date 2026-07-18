import logging
from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
import auth as auth_utils
import models
from database import get_db
from dependencies import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Auth"])

# We define this here or import it. In main.py it was 60.0
DELIVERY_CHARGE = 60.0

@router.post("/register")
def register(user_data: auth_utils.UserCreate, db: Session = Depends(get_db)):
    if db.query(models.User).filter(models.User.email == user_data.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")

    role = user_data.role if user_data.role in ("owner", "admin", "staff") else "owner"
    new_user = models.User(
        email=user_data.email,
        hashed_password=auth_utils.get_password_hash(user_data.password),
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


@router.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.email == form_data.username).first()
    if not user or not auth_utils.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect email or password")

    token = auth_utils.create_access_token(
        data={"sub": user.email},
        expires_delta=timedelta(minutes=auth_utils.ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return {"access_token": token, "token_type": "bearer", "business_name": user.business_name, "role": user.role}


@router.get("/me")
def get_me(current_user: models.User = Depends(get_current_user)):
    return {
        "id": current_user.id,
        "email": current_user.email,
        "business_name": current_user.business_name,
        "role": current_user.role,
        "shop_owner_id": current_user.shop_owner_id,
        "created_at": current_user.created_at,
    }

