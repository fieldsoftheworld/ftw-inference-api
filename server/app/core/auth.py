from datetime import datetime, timedelta

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from app.core.config import get_settings

# Create two security schemes:
# 1. When auth is enabled: Required bearer token
# 2. When auth is disabled: Optional bearer token (for testing)
security = HTTPBearer(auto_error=not get_settings().auth_disabled)


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    """
    Create a JWT access token
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=get_settings().access_token_expire_minutes
        )
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode, get_settings().secret_key, algorithm=get_settings().algorithm
    )
    return encoded_jwt


def verify_token(token: str) -> dict:
    """
    Verify a JWT token and return the payload
    """
    try:
        payload = jwt.decode(
            token, get_settings().secret_key, algorithms=[get_settings().algorithm]
        )
        return payload
    except JWTError as err:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        ) from err


async def verify_auth(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> dict:
    """
    Dependency to verify JWT token from Authorization header
    If authentication is disabled in config, returns a default user without verification
    """
    # Check if authentication is disabled in config
    if get_settings().auth_disabled:
        return {"sub": "guest_user"}

    # At this point, auth is enabled, and we should have credentials
    # If not, it means auto_error=False, so we need to raise the error manually
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Normal authentication flow with token verification
    return verify_token(credentials.credentials)
