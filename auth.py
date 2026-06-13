import os
import hashlib
import secrets
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from dotenv import load_dotenv
from fastapi import HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

load_dotenv()

API_KEYS_RAW = os.getenv("API_KEYS", "")
RATE_LIMIT_DEFAULT = os.getenv("RATE_LIMIT_DEFAULT", "100/minute")
RATE_LIMIT_UPLOAD = os.getenv("RATE_LIMIT_UPLOAD", "10/minute")
RATE_LIMIT_ASK = os.getenv("RATE_LIMIT_ASK", "30/minute")

security = HTTPBearer(auto_error=False)


@dataclass
class APIKeyInfo:
    key_hash: str
    name: str
    rate_limit: str = RATE_LIMIT_DEFAULT
    quota: int = 10000
    used: int = 0
    enabled: bool = True


class AuthManager:
    def __init__(self):
        self.keys: Dict[str, APIKeyInfo] = {}
        self._load_keys()
        self.limiter = Limiter(key_func=self._get_rate_limit_key)
    
    def _load_keys(self):
        if not API_KEYS_RAW:
            return
        for entry in API_KEYS_RAW.split(","):
            entry = entry.strip()
            if not entry:
                continue
            parts = entry.split(":")
            if len(parts) >= 2:
                name, key = parts[0], parts[1]
                rate_limit = parts[2] if len(parts) > 2 else RATE_LIMIT_DEFAULT
                quota = int(parts[3]) if len(parts) > 3 else 10000
                key_hash = hashlib.sha256(key.encode()).hexdigest()
                self.keys[key_hash] = APIKeyInfo(
                    key_hash=key_hash,
                    name=name,
                    rate_limit=rate_limit,
                    quota=quota,
                    used=0,
                    enabled=True
                )
    
    def _get_rate_limit_key(self, request: Request) -> str:
        api_key = self._extract_api_key(request)
        if api_key:
            return f"apikey:{api_key}"
        return get_remote_address(request)
    
    def _extract_api_key(self, request: Request) -> Optional[str]:
        auth = request.headers.get("Authorization")
        if auth and auth.startswith("Bearer "):
            return auth[7:]
        return request.query_params.get("api_key")
    
    def verify_key(self, credentials: HTTPAuthorizationCredentials = Depends(security)) -> APIKeyInfo:
        if not credentials:
            raise HTTPException(status_code=401, detail="Missing API key. Provide via Authorization: Bearer <key> or ?api_key=<key>")
        
        key_hash = hashlib.sha256(credentials.credentials.encode()).hexdigest()
        key_info = self.keys.get(key_hash)
        
        if not key_info or not key_info.enabled:
            raise HTTPException(status_code=401, detail="Invalid or disabled API key")
        
        if key_info.used >= key_info.quota:
            raise HTTPException(status_code=429, detail=f"Quota exceeded for key '{key_info.name}'")
        
        key_info.used += 1
        return key_info
    
    def verify_key_optional(self, credentials: HTTPAuthorizationCredentials = Depends(security)) -> Optional[APIKeyInfo]:
        if not credentials:
            return None
        key_hash = hashlib.sha256(credentials.credentials.encode()).hexdigest()
        return self.keys.get(key_hash)
    
    def generate_key(self, name: str, rate_limit: str = RATE_LIMIT_DEFAULT, quota: int = 10000) -> str:
        key = secrets.token_urlsafe(32)
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        self.keys[key_hash] = APIKeyInfo(
            key_hash=key_hash,
            name=name,
            rate_limit=rate_limit,
            quota=quota,
            used=0,
            enabled=True
        )
        return key
    
    def revoke_key(self, name: str) -> bool:
        for key_hash, info in self.keys.items():
            if info.name == name:
                info.enabled = False
                return True
        return False
    
    def get_key_stats(self, name: str) -> Optional[Dict[str, Any]]:
        for info in self.keys.values():
            if info.name == name:
                return {
                    "name": info.name,
                    "rate_limit": info.rate_limit,
                    "quota": info.quota,
                    "used": info.used,
                    "remaining": info.quota - info.used,
                    "enabled": info.enabled
                }
        return None
    
    def list_keys(self) -> list:
        return [
            {
                "name": info.name,
                "rate_limit": info.rate_limit,
                "quota": info.quota,
                "used": info.used,
                "remaining": info.quota - info.used,
                "enabled": info.enabled
            }
            for info in self.keys.values()
        ]


auth_manager = AuthManager()
limiter = auth_manager.limiter