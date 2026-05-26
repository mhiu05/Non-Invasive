import os
from jose import JWTError, jwt
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET", "")
ALGORITHM = "HS256"

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login", auto_error=False)

def get_current_user(token: str = Depends(oauth2_scheme)):
    if not token or not SUPABASE_JWT_SECRET:
        return None
    try:
        payload = jwt.decode(
            token, 
            SUPABASE_JWT_SECRET, 
            algorithms=[ALGORITHM],
            audience="authenticated"
        )
        user_id: str = payload.get("sub")
        email: str = payload.get("email")
        if user_id is None:
            return None
        return {"id": user_id, "email": email}
    except JWTError as e:
        return None

def get_current_user_required(token: str = Depends(oauth2_scheme)):
    user = get_current_user(token)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user
