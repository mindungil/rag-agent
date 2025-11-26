"""
Pydantic 모델 정의
OpenAPI 형식의 요청/응답 스키마를 정의합니다.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class QueryRequest(BaseModel):
    """질의 요청 모델"""
    prompt: str = Field(..., description="사용자 질문", min_length=1)
    top_k: int = Field(default=5, description="검색할 문서 개수", ge=1, le=20)
    temperature: float = Field(default=0.7, description="생성 온도", ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, description="최대 토큰 수", ge=100, le=4096)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "prompt": "휴가 신청은 어떻게 하나요?",
                    "top_k": 5,
                    "temperature": 0.7,
                    "max_tokens": 1024
                }
            ]
        }
    }


class DocumentReference(BaseModel):
    """참조 문서 정보"""
    filename: str = Field(..., description="파일명")
    download_url: str = Field(..., description="문서 다운로드 링크")


class QueryResponse(BaseModel):
    """질의 응답 모델"""
    answer: str = Field(..., description="생성된 답변")
    references: List[DocumentReference] = Field(
        default_factory=list,
        description="참조 문서 목록"
    )
    prompt: str = Field(..., description="원본 질문")
    retrieved_count: int = Field(..., description="검색된 문서 개수")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "answer": "휴가 신청은 다음과 같이 진행됩니다...",
                    "references": [
                        {
                            "filename": "휴가관리규정.pdf",
                            "download_url": "https://ai.jb.go.kr/files/download?path=documents/static/휴가관리규정.pdf"
                        }
                    ],
                    "prompt": "휴가 신청은 어떻게 하나요?",
                    "retrieved_count": 5
                }
            ]
        }
    }


class HealthResponse(BaseModel):
    """헬스 체크 응답 모델"""
    status: str = Field(..., description="서비스 상태 (healthy/unhealthy)")
    qdrant_connected: bool = Field(..., description="Qdrant 연결 상태")
    collection_name: str = Field(..., description="사용 중인 컬렉션 이름")
    collection_vector_count: int = Field(..., description="컬렉션 벡터 개수")
    embedding_dimension: int = Field(..., description="임베딩 벡터 차원")
    model_name: str = Field(..., description="사용 중인 임베딩 모델")
    device: str = Field(..., description="사용 중인 디바이스 (cuda/cpu)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "healthy",
                    "qdrant_connected": True,
                    "collection_name": "v1",
                    "collection_vector_count": 1500,
                    "embedding_dimension": 1024,
                    "model_name": "intfloat/multilingual-e5-large-instruct",
                    "device": "cuda"
                }
            ]
        }
    }
