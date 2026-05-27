import os
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from supabase import create_client, Client

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login", auto_error=False)

def get_current_user(token: str = Depends(oauth2_scheme)):
    if not token or not SUPABASE_URL or not SUPABASE_ANON_KEY:
        return None
        
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
        response = supabase.auth.get_user(token)
        
        if response and response.user:
            return {"id": response.user.id, "email": response.user.email}
        return None
    except Exception as e:
        print(f"Auth Error: {e}")
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
