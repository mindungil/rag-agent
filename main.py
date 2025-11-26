"""
RAG 기반 업무편람 도우미 API 서버
FastAPI를 사용하여 OpenAPI 형식으로 요청을 받아 처리합니다.
"""

import os
import uuid
import time
import json
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import logging
from dotenv import load_dotenv

from models import QueryRequest, QueryResponse, HealthResponse
from openai_models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatCompletionUsage,
    ChatMessage,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ModelsResponse,
    ModelInfo
)
from rag_service import RAGService
from auth import verify_api_key

# .env 파일 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# RAG 서비스 인스턴스 (전역)
rag_service: RAGService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작/종료 시 실행되는 라이프사이클 관리"""
    global rag_service

    logger.info("RAG 서비스 초기화 중...")
    try:
        # 환경변수에서 설정 로드
        rag_service = RAGService(
            qdrant_host=os.getenv("QDRANT_HOST", "qdrant-rag"),
            qdrant_grpc_port=int(os.getenv("QDRANT_GRPC_PORT", "6334")),
            collection_name=os.getenv("COLLECTION_NAME", "v1"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large-instruct"),
            llm_api_url=os.getenv("LLM_API_URL"),
            llm_model=os.getenv("LLM_MODEL"),
            download_base_url=os.getenv(
                "DOWNLOAD_BASE_URL",
                "https://ai.jb.go.kr/files/download?path=documents/static/"
            )
        )
        await rag_service.initialize()
        logger.info("RAG 서비스 초기화 완료")
    except Exception as e:
        logger.error(f"RAG 서비스 초기화 실패: {e}")
        raise

    yield

    logger.info("RAG 서비스 종료 중...")
    if rag_service:
        await rag_service.close()


# FastAPI 앱 생성
app = FastAPI(
    title="업무편람 도우미 API",
    description="RAG 기반 업무편람 질의응답 서비스",
    version="1.0.0",
    lifespan=lifespan
)

# 정적 파일 서빙을 위한 설정 (문서 다운로드)
app.mount("/static/documents", StaticFiles(directory="/workspace/rag_agent/documents/static"), name="documents")

# 요청 로깅 미들웨어 - body를 캐싱하여 재사용 가능하게 함
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """모든 요청을 로깅"""
    if request.url.path == "/v1/chat/completions":
        # Body를 읽고 캐싱
        body = await request.body()

        logger.info("=" * 60)
        logger.info(f"📥 요청 수신: {request.method} {request.url.path}")
        logger.info(f"Content-Type: {request.headers.get('content-type')}")
        logger.info(f"Authorization: {request.headers.get('authorization', 'None')[:20]}...")
        logger.info(f"Body: {body.decode('utf-8', errors='replace')}")
        logger.info("=" * 60)

    response = await call_next(request)

    if request.url.path == "/v1/chat/completions" and response.status_code >= 400:
        logger.error(f"❌ 응답 상태: {response.status_code}")

    return response


# CORS 설정 - 프론트엔드 도메인 명시적 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://ai.jb.go.kr",
        "http://ai.jb.go.kr",
        "http://localhost:8080",
        "http://localhost:3000",
        "*"  # 개발 환경을 위해 모든 출처 허용
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=[
        "Content-Type",
        "Authorization",
        "X-API-Key",  # API 키 헤더 명시적 허용
        "Accept",
        "Origin",
        "X-Requested-With"
    ],
    expose_headers=["*"],
    max_age=3600,  # Preflight 캐시 시간 (1시간)
)

# 정적 파일 서빙 - 문서 다운로드용
app.mount("/static/documents", StaticFiles(directory="/workspace/rag_agent/documents/static"), name="documents")


# 커스텀 파일 다운로드 엔드포인트 (한글 파일명 지원)
@app.api_route("/download/{filename:path}", methods=["GET", "HEAD"])
async def download_file(filename: str):
    """
    파일 다운로드 엔드포인트
    한글 파일명을 포함한 모든 파일 다운로드 지원
    """
    from urllib.parse import unquote

    # URL 디코딩
    decoded_filename = unquote(filename)
    file_path = os.path.join("/workspace/rag_agent/documents/static/", decoded_filename)

    logger.info(f"📥 파일 다운로드 요청: {decoded_filename}")

    if os.path.exists(file_path) and os.path.isfile(file_path):
        # Content-Disposition을 attachment로 설정하여 강제 다운로드
        # RFC 5987에 따라 한글 파일명을 URL 인코딩
        from urllib.parse import quote
        encoded_filename = quote(decoded_filename)
        headers = {
            'Content-Disposition': f'attachment; filename*=UTF-8\'\'{encoded_filename}'
        }

        # 파일 확장자에 따른 MIME type 설정
        ext = decoded_filename.lower().split('.')[-1] if '.' in decoded_filename else ''
        mime_types = {
            'pdf': 'application/pdf',
            'hwp': 'application/octet-stream',  # 강제 다운로드
            'hwpx': 'application/octet-stream',  # 강제 다운로드
            'doc': 'application/msword',
            'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'xls': 'application/vnd.ms-excel',
            'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'zip': 'application/zip',
        }
        media_type = mime_types.get(ext, 'application/octet-stream')

        return FileResponse(
            path=file_path,
            filename=decoded_filename,
            media_type=media_type,
            headers=headers
        )

    logger.error(f"❌ 파일을 찾을 수 없음: {file_path}")
    raise HTTPException(status_code=404, detail="File not found")


# 유효성 검증 오류 상세 로깅
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Pydantic 유효성 검증 오류를 로깅하고 상세한 오류 메시지 반환
    """
    body = await request.body()
    logger.error("=" * 60)
    logger.error(f"❌ 요청 유효성 검증 실패")
    logger.error(f"URL: {request.url}")
    logger.error(f"Method: {request.method}")
    logger.error(f"Headers: {dict(request.headers)}")
    logger.error(f"Body: {body.decode() if body else 'Empty'}")
    logger.error(f"에러 상세:")
    for error in exc.errors():
        logger.error(f"  - 필드: {error['loc']}")
        logger.error(f"    타입: {error['type']}")
        logger.error(f"    메시지: {error['msg']}")
    logger.error("=" * 60)

    return JSONResponse(
        status_code=422,
        content={
            "detail": exc.errors(),
            "message": "요청 데이터 유효성 검증 실패"
        }
    )


# 일반 예외 핸들러
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """
    처리되지 않은 모든 예외를 로깅
    """
    logger.error("=" * 60)
    logger.error(f"❌ 예상치 못한 오류 발생")
    logger.error(f"URL: {request.url}")
    logger.error(f"Method: {request.method}")
    logger.error(f"오류: {str(exc)}")
    logger.error(f"오류 타입: {type(exc).__name__}")
    logger.error("=" * 60)
    import traceback
    logger.error(traceback.format_exc())

    return JSONResponse(
        status_code=500,
        content={"detail": "서버 내부 오류가 발생했습니다"}
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    헬스 체크 엔드포인트
    Qdrant 연결 상태 및 모델 상태를 점검합니다.
    """
    try:
        if rag_service is None:
            raise HTTPException(status_code=503, detail="RAG 서비스가 초기화되지 않았습니다")

        health_status = await rag_service.health_check()

        if not health_status["qdrant_connected"]:
            raise HTTPException(status_code=503, detail="Qdrant 연결 실패")

        return HealthResponse(**health_status)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"헬스 체크 오류: {e}")
        raise HTTPException(status_code=500, detail=f"헬스 체크 중 오류 발생: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest
):
    """
    RAG 기반 질의응답 엔드포인트
    사용자의 질문을 받아 관련 문서를 검색하고 답변을 생성합니다.

    **인증 필요**: X-API-Key 헤더에 유효한 API 키 필요
    """
    try:
        if rag_service is None:
            raise HTTPException(status_code=503, detail="RAG 서비스가 초기화되지 않았습니다")

        logger.info(f"질의 수신: {request.prompt[:100]}...")

        # RAG 처리
        result = await rag_service.process_query(
            prompt=request.prompt,
            top_k=request.top_k,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )

        logger.info(f"질의 처리 완료, 참조 문서 수: {len(result['references'])}")

        return QueryResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"질의 처리 오류: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"질의 처리 중 오류 발생: {str(e)}")


@app.get("/")
async def root():
    """루트 엔드포인트 - API 정보 반환"""
    return {
        "service": "업무편람 도우미 API",
        "version": "1.0.0",
        "endpoints": {
            "POST /query": "RAG 기반 질의응답 (커스텀 형식)",
            "POST /v1/chat/completions": "RAG 기반 질의응답 (OpenAI 호환)",
            "GET /v1/models": "사용 가능한 모델 목록 (OpenAI 호환)",
            "GET /v1/health": "서비스 헬스 체크 (OpenAI 호환)",
            "GET /health": "서비스 헬스 체크 (상세)",
            "GET /docs": "API 문서 (Swagger UI)",
            "GET /redoc": "API 문서 (ReDoc)"
        }
    }


@app.get("/v1/health")
async def v1_health_check():
    """
    OpenAI 호환 헬스 체크 엔드포인트
    /v1 경로 아래에서 서비스 상태를 확인합니다.
    """
    try:
        if rag_service is None:
            return {
                "status": "unavailable",
                "message": "RAG 서비스가 초기화되지 않았습니다"
            }

        health_status = await rag_service.health_check()

        if not health_status["qdrant_connected"]:
            return {
                "status": "degraded",
                "message": "Qdrant 연결 실패",
                "details": health_status
            }

        return {
            "status": "ok",
            "message": "서비스가 정상적으로 작동 중입니다",
            "details": health_status
        }

    except Exception as e:
        logger.error(f"V1 헬스 체크 오류: {e}")
        return {
            "status": "error",
            "message": str(e)
        }


@app.get("/v1/models", response_model=ModelsResponse)
async def list_models():
    """
    사용 가능한 모델 목록 반환 (OpenAI 호환)
    Open-webui 등에서 모델 선택 시 사용됩니다.
    """
    return ModelsResponse(
        object="list",
        data=[
            ModelInfo(
                id="rag-model",
                object="model",
                created=int(time.time()),
                owned_by="organization"
            )
        ]
    )


async def chat_completions_streaming(request: ChatCompletionRequest):
    """
    스트리밍 응답 생성기
    """
    async def generate():
        try:
            # 마지막 사용자 메시지 추출
            user_messages = [msg for msg in request.messages if msg.role == "user"]
            if not user_messages:
                raise HTTPException(status_code=400, detail="사용자 메시지가 없습니다")

            user_query = user_messages[-1].content
            logger.info(f"Streaming Chat Completion 질의 수신: {user_query[:100]}...")

            # RAG 처리 (스트리밍용)
            result = await rag_service.process_query_streaming(
                prompt=user_query,
                top_k=request.top_k or 5,
                temperature=request.temperature,
                max_tokens=request.max_tokens or 1024
            )

            chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

            # 첫 번째 청크: role 전송
            chunk = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(chunk)}\n\n"

            # 답변 스트리밍
            async for text_chunk in result["stream"]:
                chunk = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": text_chunk},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(chunk)}\n\n"

            # 참조 문서 추가 (한 번에 전체 내용 전송)
            if result.get("references"):
                # 모든 참조 문서를 하나의 문자열로 생성
                sources_md = "\n\n참고 문서:\n"
                seen = set()
                for ref in result["references"]:
                    key = (ref.filename, ref.download_url)
                    if ref.download_url and ref.filename and key not in seen:
                        sources_md += f"출처: [{ref.filename}]({ref.download_url})\n"
                        seen.add(key)

                # 참조 문서가 있으면 한 번에 전송
                if sources_md.strip() != "참고 문서:":
                    chunk = {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": sources_md},
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

            # 마지막 청크: finish_reason 전송
            chunk = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

            # 스트리밍 종료
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Streaming 오류: {e}", exc_info=True)
            error_chunk = {
                "error": {
                    "message": str(e),
                    "type": "internal_error"
                }
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest
):
    """
    OpenAI 호환 Chat Completion 엔드포인트
    VLLM, SGLang 등과 동일한 형식으로 요청을 받아 RAG 처리 후 응답합니다.

    **인증 필요**: X-API-Key 헤더에 유효한 API 키 필요
    **Open-webui 호환**: Open-webui에서 일반 LLM 모델처럼 사용 가능
    **스트리밍 지원**: stream=true 시 SSE 형식으로 응답
    """
    try:
        if rag_service is None:
            raise HTTPException(status_code=503, detail="RAG 서비스가 초기화되지 않았습니다")

        # 스트리밍 요청 처리
        if request.stream:
            return await chat_completions_streaming(request)

        # 마지막 사용자 메시지 추출
        user_messages = [msg for msg in request.messages if msg.role == "user"]
        if not user_messages:
            raise HTTPException(status_code=400, detail="사용자 메시지가 없습니다")

        user_query = user_messages[-1].content
        logger.info(f"Chat Completion 질의 수신: {user_query[:100]}...")

        # RAG 처리
        result = await rag_service.process_query(
            prompt=user_query,
            top_k=request.top_k or 5,
            temperature=request.temperature,
            max_tokens=request.max_tokens or 1024
        )

        # 응답 메시지 구성: 답변 + 참조 문서 (마크다운 링크 형식)
        answer_content = result["answer"]

        # 참조 문서가 있으면 마크다운 링크 추가 (중복 제거)
        if result["references"]:
            sources_md = "\n\n참고 문서:\n"
            seen = set()
            for ref in result["references"]:
                key = (ref.filename, ref.download_url)
                if ref.download_url and ref.filename and key not in seen:
                    sources_md += f"출처: [{ref.filename}]({ref.download_url})\n"
                    seen.add(key)

            if sources_md.strip() != "참고 문서:":
                answer_content += sources_md

        # 토큰 수 추정 (간단한 추정)
        prompt_tokens = len(user_query.split()) * 2
        completion_tokens = len(answer_content.split()) * 2

        # OpenAI 호환 응답 생성
        response = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            object="chat.completion",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=answer_content
                    ),
                    finish_reason="stop"
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        )

        logger.info(f"Chat Completion 처리 완료, 참조 문서 수: {len(result['references'])}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat Completion 처리 오류: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat Completion 처리 중 오류 발생: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
