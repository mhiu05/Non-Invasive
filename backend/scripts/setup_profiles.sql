-- ========================================================
-- SCRIPT THIẾT LẬP BẢNG PROFILES VÀ BẢO MẬT GOOGLE OAUTH
-- ========================================================

-- 1. Tạo bảng profiles để đồng bộ thông tin người dùng
CREATE TABLE IF NOT EXISTS public.profiles (
  id uuid REFERENCES auth.users ON DELETE CASCADE PRIMARY KEY,
  username text UNIQUE NOT NULL,
  full_name text,
  gender text,
  dob date,
  created_at timestamp with time zone DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- 2. Bật Row Level Security (RLS) cho bảng profiles
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;

-- Policy: Bất kỳ ai cũng có thể xem hồ sơ công khai (cần thiết để backend phân giải username)
CREATE POLICY "Public profiles are viewable by everyone." 
  ON public.profiles FOR SELECT 
  USING (true);

-- Policy: Users chỉ có thể sửa hồ sơ của chính mình
CREATE POLICY "Users can update own profile." 
  ON public.profiles FOR UPDATE 
  USING (auth.uid() = id);

-- 3. Tạo Trigger tự động đồng bộ dữ liệu từ auth.users sang public.profiles
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS trigger AS $$
BEGIN
  -- CHẶN TẠO MỚI BẰNG GOOGLE TỪ NHỮNG USER CHƯA ĐĂNG KÝ FORM
  -- Supabase lưu thông tin provider trong raw_app_meta_data
  IF new.raw_app_meta_data->>'provider' = 'google' THEN
    RAISE EXCEPTION 'Vui lòng đăng ký tài khoản bằng Form trước khi sử dụng Đăng nhập bằng Google.';
  END IF;

  -- Nếu đi qua được đây nghĩa là user đăng ký bằng Form (email/password).
  -- Ta lấy thông tin từ raw_user_meta_data (được gửi từ frontend signUp).
  INSERT INTO public.profiles (id, username, full_name, gender, dob)
  VALUES (
    new.id,
    new.raw_user_meta_data->>'username',
    new.raw_user_meta_data->>'full_name',
    new.raw_user_meta_data->>'gender',
    NULLIF(new.raw_user_meta_data->>'dob', '')::date
  );

  RETURN new;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Gắn Trigger vào bảng auth.users
DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE PROCEDURE public.handle_new_user();
