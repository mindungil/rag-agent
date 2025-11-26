"""
API 키 인증 모듈
X-API-Key 헤더 및 Authorization: Bearer 토큰 모두 지원
"""

import os
from typing import Optional
from fastapi import Security, HTTPException, status, Request
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 환경변수에서 API 키 가져오기
API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise ValueError("API_KEY가 .env 파일에 설정되지 않았습니다.")

# API 키 헤더 정의 (X-API-Key 헤더 사용)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Bearer 토큰 정의 (Authorization: Bearer 형식)
bearer_scheme = HTTPBearer(auto_error=False)


async def verify_api_key(
    api_key_header_value: Optional[str] = Security(api_key_header),
    bearer_token: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme)
) -> str:
    """
    API 키 검증 함수
    두 가지 형식을 모두 지원:
    1. X-API-Key: your-api-key
    2. Authorization: Bearer your-api-key

    Args:
        api_key_header_value: X-API-Key 헤더 값
        bearer_token: Authorization Bearer 토큰

    Returns:
        검증된 API 키

    Raises:
        HTTPException: API 키가 유효하지 않을 경우
    """
    import logging
    logger = logging.getLogger(__name__)

    # 두 가지 형식 중 하나에서 API 키 추출
    provided_key = None

    if api_key_header_value:
        provided_key = api_key_header_value
    elif bearer_token:
        provided_key = bearer_token.credentials

    # API 키가 제공되지 않음
    if not provided_key:
        logger.warning("API 키가 제공되지 않음")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API 키가 필요합니다. X-API-Key 헤더 또는 Authorization: Bearer 토큰을 사용하세요.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # API 키 검증
    if provided_key != API_KEY:
        logger.warning("API 키 인증 실패")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="유효하지 않은 API 키입니다.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return provided_key
