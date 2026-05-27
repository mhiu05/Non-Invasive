"""
security.py — JWT authentication and user validation.

Responsibilities:
- Verify access tokens sent from the frontend using Supabase Auth.
- Provide dependencies (Depends) to protect FastAPI routes.
"""

import logging
import os
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from supabase import create_client, Client

logger = logging.getLogger(__name__)

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip('"').strip("'")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "").strip('"').strip("'")

# Global Supabase client for auth verification to avoid recreating it per request
supabase_client: Client | None = None
if SUPABASE_URL and SUPABASE_ANON_KEY:
    supabase_client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login-username", auto_error=False)

def get_current_user(token: str = Depends(oauth2_scheme)):
    if not token or not supabase_client:
        return None
        
    try:
        response = supabase_client.auth.get_user(token)
        
        if response and response.user:
            return {"id": response.user.id, "email": response.user.email}
        return None
    except Exception as e:
        logger.error("Auth Error: %s", e)
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
