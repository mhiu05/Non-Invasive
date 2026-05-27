from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from supabase import create_client, Client
import os
from dotenv import load_dotenv

load_dotenv()

router = APIRouter(prefix="/auth", tags=["Auth"])

SUPABASE_URL = os.environ.get("VITE_SUPABASE_URL") or os.environ.get("SUPABASE_URL")
# Dùng service key để có quyền query bảng profiles bất chấp RLS
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")

class LoginUsernameRequest(BaseModel):
    username: str
    password: str

@router.post("/login-username")
async def login_username(req: LoginUsernameRequest):
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise HTTPException(status_code=500, detail="Thiếu cấu hình Supabase Backend.")

    # 1. Khởi tạo Supabase client với Service Key để bypass RLS
    admin_client: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

    # 2. Truy vấn email từ username trong bảng profiles
    response = admin_client.table("profiles").select("id, username").eq("username", req.username).execute()
    data = response.data
    
    if not data or len(data) == 0:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Tên đăng nhập không tồn tại."
        )
    
    user_id = data[0]["id"]

    # 3. Sử dụng API Admin (hoặc rpc) để lấy email thực sự của user này?
    # Bảng profiles không lưu email để bảo mật. Chúng ta phải lấy email từ auth.users.
    # supabase-py cung cấp admin.get_user_by_id.
    try:
        user_response = admin_client.auth.admin.get_user_by_id(user_id)
        email = user_response.user.email
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Không lấy được thông tin email của tài khoản này."
        )
    
    # 4. Xác thực password thông qua signInWithPassword
    # (Vì Supabase Service Key không gọi được sign_in_with_password để cấp token thông thường,
    # ta phải khởi tạo một client anon bình thường).
    anon_key = os.environ.get("VITE_SUPABASE_ANON_KEY") or os.environ.get("SUPABASE_ANON_KEY")
    if not anon_key:
        raise HTTPException(status_code=500, detail="Thiếu cấu hình ANON KEY.")
    
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
            detail="Tên đăng nhập hoặc mật khẩu không đúng."
        )
