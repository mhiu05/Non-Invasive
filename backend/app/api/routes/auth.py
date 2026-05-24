import uuid
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from app.schemas.auth import UserCreate, UserResponse, Token, UserLogin
from app.core.security import get_password_hash, verify_password, create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES, get_current_user_required
from app.services.history_store import get_user_by_email, create_user

router = APIRouter()

@router.post("/register", response_model=UserResponse)
def register_user(user_in: UserCreate):
    existing_user = get_user_by_email(user_in.email)
    if existing_user:
        raise HTTPException(
            status_code=400,
            detail="The user with this email already exists in the system.",
        )
    user_id = str(uuid.uuid4())
    user_data = {
        "id": user_id,
        "username": user_in.username,
        "email": user_in.email,
        "hashed_password": get_password_hash(user_in.password),
        "created_at": datetime.utcnow().isoformat(),
    }
    user = create_user(user_data)
    return user


@router.post("/login", response_model=Token)
def login_access_token(form_data: UserLogin):
    user = get_user_by_email(form_data.email)
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(user["id"]), "username": user["username"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/me", response_model=UserResponse)
def read_current_user(current_user: dict = Depends(get_current_user_required)):
    # We could fetch from DB, but token has enough info or we can fetch to get email
    # Let's fetch from DB to get the full user email
    # wait, we don't have get_user_by_id. Let's just use what's in token, or add get_user_by_id
    from app.services.history_store import _get_conn, _row_to_dict
    conn = _get_conn()
    row = conn.execute("SELECT * FROM users WHERE id = ?", (current_user["id"],)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="User not found")
    
    return dict(row)
