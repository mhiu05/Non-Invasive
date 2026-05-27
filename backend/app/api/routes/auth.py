"""
auth.py — FastAPI endpoints for authentication.

POST /auth/login-username  →  { username, password } → { access_token, refresh_token, user }
"""

from fastapi import APIRouter, HTTPException, status
from supabase import create_client, Client
from app.schemas.auth import LoginUsernameRequest
import os
from dotenv import load_dotenv

load_dotenv()

router = APIRouter(prefix="/auth", tags=["Auth"])

SUPABASE_URL = os.environ.get("VITE_SUPABASE_URL") or os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY") # Service key to have permission to query the profiles table bypassing all Row Level Security (RLS) rules

@router.post("/login-username")
async def login_username(req: LoginUsernameRequest):
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise HTTPException(status_code=500, detail="Missing Supabase Backend configuration.")

    # 1. Initialize Admin Client
    admin_client: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

    # 2. Find user_id by username in profiles table
    response = admin_client.table("profiles").select("id, username").eq("username", req.username).execute()
    data = response.data
    if not data or len(data) == 0:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Username does not exist."
        )
    user_id = data[0]["id"]

    # 3. Get user's real email
    try:
        user_response = admin_client.auth.admin.get_user_by_id(user_id)
        email = user_response.user.email
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not retrieve email information for this account."
        )
    
    # 4. Perform login and issue token
    anon_key = os.environ.get("VITE_SUPABASE_ANON_KEY") or os.environ.get("SUPABASE_ANON_KEY")
    if not anon_key:
        raise HTTPException(status_code=500, detail="Missing ANON KEY configuration.")
    client: Client = create_client(SUPABASE_URL, anon_key)
    try:
        auth_response = client.auth.sign_in_with_password({"email": email, "password": req.password})
        session = auth_response.session
        return {
            "access_token": session.access_token,
            "refresh_token": session.refresh_token,
            "user": {
                "id": auth_response.user.id,
                "email": auth_response.user.email
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password."
        )
