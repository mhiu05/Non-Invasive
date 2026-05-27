from pydantic import BaseModel

class LoginUsernameRequest(BaseModel):
    username: str
    password: str
