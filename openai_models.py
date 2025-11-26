"""
OpenAI 호환 API 모델 정의
VLLM, SGLang 등과 동일한 형식으로 요청/응답을 처리합니다.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Union
from datetime import datetime


# ============= Chat Completion Models =============

class ChatMessage(BaseModel):
    """채팅 메시지"""
    role: Literal["system", "user", "assistant"] = Field(..., description="메시지 역할")
    content: str = Field(..., description="메시지 내용")


class ChatCompletionRequest(BaseModel):
    """OpenAI Chat Completion 요청 형식"""
    model: str = Field(default="rag-model", description="사용할 모델 이름")
    messages: List[ChatMessage] = Field(..., description="대화 메시지 목록", min_length=1)
    temperature: float = Field(default=0.7, description="생성 온도", ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=1024, description="최대 토큰 수", ge=100, le=4096)
    top_k: Optional[int] = Field(default=5, description="검색할 문서 개수 (RAG 전용)", ge=1, le=20)
    stream: bool = Field(default=False, description="스트리밍 여부 (현재 미지원)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "model": "rag-model",
                    "messages": [
                        {"role": "user", "content": "휴가 신청은 어떻게 하나요?"}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 1024,
                    "top_k": 5
                }
            ]
        }
    }


class ChatCompletionChoice(BaseModel):
    """응답 선택지"""
    index: int = Field(..., description="선택지 인덱스")
    message: ChatMessage = Field(..., description="응답 메시지")
    finish_reason: Literal["stop", "length"] = Field(..., description="완료 이유")


# ============= Streaming Models =============

class ChatCompletionChunkDelta(BaseModel):
    """스트리밍 델타 (변경사항)"""
    role: Optional[Literal["assistant"]] = Field(None, description="역할 (첫 청크에만)")
    content: Optional[str] = Field(None, description="내용 조각")


class ChatCompletionChunkChoice(BaseModel):
    """스트리밍 응답 선택지"""
    index: int = Field(..., description="선택지 인덱스")
    delta: ChatCompletionChunkDelta = Field(..., description="변경사항")
    finish_reason: Optional[Literal["stop", "length"]] = Field(None, description="완료 이유")


class ChatCompletionUsage(BaseModel):
    """토큰 사용량"""
    prompt_tokens: int = Field(..., description="입력 토큰 수")
    completion_tokens: int = Field(..., description="출력 토큰 수")
    total_tokens: int = Field(..., description="전체 토큰 수")


class ChatCompletionResponse(BaseModel):
    """OpenAI Chat Completion 응답 형식"""
    id: str = Field(..., description="응답 고유 ID")
    object: Literal["chat.completion"] = Field(default="chat.completion", description="객체 타입")
    created: int = Field(..., description="생성 시간 (Unix timestamp)")
    model: str = Field(..., description="사용된 모델 이름")
    choices: List[ChatCompletionChoice] = Field(..., description="응답 선택지 목록")
    usage: ChatCompletionUsage = Field(..., description="토큰 사용량")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": "chatcmpl-123",
                    "object": "chat.completion",
                    "created": 1677652288,
                    "model": "rag-model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "휴가 신청은 다음과 같이 진행됩니다..."
                            },
                            "finish_reason": "stop"
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 20,
                        "completion_tokens": 100,
                        "total_tokens": 120
                    }
                }
            ]
        }
    }


# ============= Models List =============

class ModelInfo(BaseModel):
    """모델 정보"""
    id: str = Field(..., description="모델 ID")
    object: Literal["model"] = Field(default="model", description="객체 타입")
    created: int = Field(..., description="생성 시간 (Unix timestamp)")
    owned_by: str = Field(default="organization", description="소유자")


class ModelsResponse(BaseModel):
    """모델 목록 응답"""
    object: Literal["list"] = Field(default="list", description="객체 타입")
    data: List[ModelInfo] = Field(..., description="모델 목록")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "object": "list",
                    "data": [
                        {
                            "id": "rag-model",
                            "object": "model",
                            "created": 1677652288,
                            "owned_by": "organization"
                        }
                    ]
                }
            ]
        }
    }
