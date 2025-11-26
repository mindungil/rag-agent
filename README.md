# RAG 기반 업무편람 도우미 API

FastAPI 기반의 RAG(Retrieval-Augmented Generation) 서비스로, 업무편람 관련 질의에 대해 관련 문서를 검색하고 AI 기반 답변을 생성합니다.

## 주요 기능

- **임베딩 기반 문서 검색**: `intfloat/multilingual-e5-large-instruct` 모델 사용 (1024차원)
- **Qdrant 벡터 DB 연동**: gRPC를 통한 고성능 검색, 컨테이너 직접 연결
- **AI 답변 생성**: OpenAI 호환 LLM API를 통한 답변 생성
- **문서 다운로드 링크 제공**: 참조 문서의 다운로드 URL 포함 (HTTPS)
- **OpenAI 호환 API**: `/v1/chat/completions` 엔드포인트로 VLLM, SGLang과 동일한 형식 지원
- **Open-webui 완벽 호환**: 일반 LLM 모델처럼 바로 사용 가능
- **API 키 인증**: X-API-Key 헤더를 통한 보안 인증
- **GPU 가속**: CUDA를 통한 고속 임베딩 생성

## 시스템 요구사항

- Docker & Docker Compose
- NVIDIA GPU (CUDA 12.1 이상)
- NVIDIA Container Toolkit
- 실행 중인 Qdrant 컨테이너 (`qdrant-rag`, 포트 6334 gRPC)
- OpenAI 호환 LLM API 서버

## 프로젝트 구조

```
rag-agent/
├── main.py              # FastAPI 메인 애플리케이션
├── models.py            # Pydantic 요청/응답 모델
├── rag_service.py       # RAG 처리 로직
├── Dockerfile           # 컨테이너 이미지 정의
├── docker-compose.yml   # 컨테이너 오케스트레이션
├── requirements.txt     # Python 의존성
└── README.md           # 프로젝트 문서
```

## 빠른 시작

### 1. 환경 변수 설정

`.env.example` 파일을 복사하여 `.env` 파일을 생성하고 API 키를 설정합니다:
```bash
cp .env.example .env
# .env 파일을 열어 API_KEY를 변경하세요
nano .env
```

**중요**: `API_KEY`를 안전한 값으로 반드시 변경하세요!

### 2. 전제 조건 확인

Qdrant가 실행 중인지 확인:
```bash
curl http://localhost:6333/collections
```

### 3. 서비스 빌드 및 실행

```bash
# 컨테이너 빌드 및 시작
docker compose up -d --build

# 로그 확인
docker compose logs -f rag-api

# 서비스 상태 확인
curl http://localhost:8000/health
```

### 4. API 사용

#### 헬스 체크 (인증 불필요)
```bash
curl http://localhost:8000/health
```

#### 방법 1: OpenAI 호환 형식 (권장 - Open-webui 호환)
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-api-key-change-this" \
  -d '{
    "model": "rag-model",
    "messages": [
      {"role": "user", "content": "휴가 신청은 어떻게 하나요?"}
    ],
    "temperature": 0.7,
    "max_tokens": 1024,
    "top_k": 5
  }'
```

#### 방법 2: 커스텀 형식 (기존 호환성)
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-api-key-change-this" \
  -d '{
    "prompt": "휴가 신청은 어떻게 하나요?",
    "top_k": 5,
    "temperature": 0.7,
    "max_tokens": 1024
  }'
```

**참고**: `X-API-Key` 헤더의 값은 `.env` 파일에 설정한 `API_KEY`와 동일해야 합니다.

## API 엔드포인트

### OpenAI 호환 엔드포인트 (권장)

#### GET `/v1/health`
서비스 상태를 확인합니다. (OpenAI 호환 형식)

**응답 예시:**
```json
{
  "status": "ok",
  "message": "서비스가 정상적으로 작동 중입니다",
  "details": {
    "status": "healthy",
    "qdrant_connected": true,
    "collection_name": "v1",
    "collection_vector_count": 24636,
    "embedding_dimension": 1024,
    "model_name": "intfloat/multilingual-e5-large-instruct",
    "device": "cuda"
  }
}
```

#### GET `/v1/models`
사용 가능한 모델 목록을 반환합니다. Open-webui에서 모델 선택 시 사용됩니다.

**응답 예시:**
```json
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
```

#### POST `/v1/chat/completions`
OpenAI Chat Completion API와 호환되는 형식으로 RAG 기반 질의응답을 수행합니다.

**인증**: `X-API-Key` 헤더 필요

**요청 헤더:**
- `Content-Type: application/json`
- `X-API-Key: <your-api-key>` (필수)

**요청 파라미터:**
- `model` (string): 모델 이름 (예: "rag-model")
- `messages` (array): 대화 메시지 목록
  - `role` (string): "system", "user", "assistant" 중 하나
  - `content` (string): 메시지 내용
- `temperature` (float, optional): 생성 온도 (기본값: 0.7, 범위: 0.0-2.0)
- `max_tokens` (integer, optional): 최대 토큰 수 (기본값: 1024)
- `top_k` (integer, optional): 검색할 문서 개수 (기본값: 5, RAG 전용 파라미터)

**응답 예시:**
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "rag-model",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "휴가 신청은 다음과 같이 진행됩니다...\n\n📚 **참조 문서:**\n\n1. **휴가관리규정.pdf** (유사도: 0.92)\n   🔗 다운로드: https://ai.jb.go.kr/files/download?path=documents/static/%ED%9C%B4%EA%B0%80%EA%B4%80%EB%A6%AC%EA%B7%9C%EC%A0%95.pdf\n"
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
```

### 커스텀 엔드포인트

#### GET `/health`
서비스 상태를 확인합니다.

**응답 예시:**
```json
{
  "status": "healthy",
  "qdrant_connected": true,
  "collection_name": "v1",
  "collection_vector_count": 1500,
  "embedding_dimension": 1024,
  "model_name": "intfloat/multilingual-e5-large-instruct",
  "device": "cuda"
}
```

### POST `/query`
RAG 기반 질의응답을 수행합니다.

**인증**: `X-API-Key` 헤더 필요

**요청 헤더:**
- `Content-Type: application/json`
- `X-API-Key: <your-api-key>` (필수)

**요청 파라미터:**
- `prompt` (string, required): 사용자 질문
- `top_k` (integer, optional): 검색할 문서 개수 (기본값: 5, 범위: 1-20)
- `temperature` (float, optional): 생성 온도 (기본값: 0.7, 범위: 0.0-2.0)
- `max_tokens` (integer, optional): 최대 토큰 수 (기본값: 1024, 범위: 100-4096)

**응답 예시:**
```json
{
  "answer": "휴가 신청은 다음과 같이 진행됩니다...",
  "references": [
    {
      "source": "휴가관리규정.pdf",
      "download_url": "https://ai.jb.go.kr/files/download?path=documents/static/%ED%9C%B4%EA%B0%80%EA%B4%80%EB%A6%AC%EA%B7%9C%EC%A0%95.pdf",
      "score": 0.92,
      "content_preview": "휴가 신청 절차는 다음과 같습니다. 1. 전자결재 시스템 접속..."
    }
  ],
  "prompt": "휴가 신청은 어떻게 하나요?",
  "retrieved_count": 5
}
```

## API 문서

서비스 실행 후 다음 URL에서 대화형 API 문서를 확인할 수 있습니다:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 보안

### API 키 관리

1. **API 키 생성**: `.env` 파일에서 `API_KEY`를 안전한 값으로 설정
   ```bash
   # 예시: 랜덤 문자열 생성
   openssl rand -hex 32
   ```

2. **API 키 사용**: 모든 `/query` 요청에 `X-API-Key` 헤더 포함
   ```bash
   curl -H "X-API-Key: your-api-key" ...
   ```

3. **보안 권장사항**:
   - `.env` 파일을 절대 Git에 커밋하지 마세요
   - API 키는 정기적으로 교체하세요
   - HTTPS를 사용하여 API 키가 암호화된 채널로 전송되도록 하세요

## Open-webui 연동

이 API는 OpenAI 호환 형식을 지원하므로 Open-webui에서 **일반 LLM 모델처럼 바로 사용**할 수 있습니다!

### 1. OpenAI 호환 모델로 등록 (권장 ⭐)

Open-webui 설정에서 외부 OpenAI 호환 API로 등록:

**설정 방법:**
1. Open-webui 관리자 페널 → **Settings** → **Connections**
2. **OpenAI API** 섹션에서 추가:
   - **API Base URL**: `http://your-server:8000/v1`
   - **API Key**: `.env` 파일에 설정한 `API_KEY` 값
3. 저장 후 모델 선택에서 **rag-model** 선택

**장점:**
- ✅ VLLM, SGLang으로 띄운 모델과 동일한 방식으로 사용
- ✅ 채팅 인터페이스에서 바로 대화 가능
- ✅ 답변에 참조 문서 다운로드 링크 자동 포함
- ✅ 일반 LLM 모델과 동일한 사용자 경험

### 2. Function Calling 방식 (대안)

Open-webui의 Functions 기능을 사용:
- Function URL: `http://your-server:8000/query`
- Method: POST
- Headers: `X-API-Key: <your-api-key>`

### 3. 사용자 경험
- 사용자가 질문하면 RAG 처리 후 답변 생성
- 답변 하단에 참조 문서 목록과 다운로드 링크 자동 표시
- 모든 다운로드 링크는 HTTPS 프로토콜 사용

**응답 예시:**
```
휴가 신청은 전자결재 시스템을 통해 진행됩니다...

📚 **참조 문서:**

1. **휴가관리규정.pdf** (유사도: 0.92)
   🔗 다운로드: https://ai.jb.go.kr/files/download?path=documents/static/휴가관리규정.pdf

2. **업무편람.pdf** (유사도: 0.85)
   🔗 다운로드: https://ai.jb.go.kr/files/download?path=documents/static/업무편람.pdf
```

## 설정 변경

모든 설정은 `.env` 파일에서 관리됩니다. 설정을 변경하려면:

1. `.env` 파일을 수정
2. 서비스를 재시작: `docker compose restart`

### 주요 설정 항목

#### Qdrant 연결 설정
```bash
QDRANT_HOST=qdrant-rag          # Qdrant 컨테이너 이름
QDRANT_GRPC_PORT=6334           # gRPC 포트
COLLECTION_NAME=v1              # 컬렉션 이름
```

#### LLM API 연결 설정
```bash
LLM_API_URL=http://your-llm-server:port/v1  # LLM API URL
LLM_MODEL=your-model-name                   # 모델 이름
```

#### 문서 다운로드 URL 설정
```bash
DOWNLOAD_BASE_URL=ai.jb.go.kr/files/download?path=documents/static/
```
**참고**: 프로토콜(https://)은 자동으로 추가됩니다.

#### 임베딩 모델 설정
```bash
EMBEDDING_MODEL=intfloat/multilingual-e5-large-instruct
```

## 컨테이너 관리

```bash
# 서비스 중지
docker compose down

# 서비스 재시작
docker compose restart

# 로그 확인
docker compose logs -f rag-api

# 컨테이너 내부 접속
docker compose exec rag-api bash
```

## 트러블슈팅

### GPU 인식 안 됨
NVIDIA Container Toolkit이 설치되어 있는지 확인:
```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Qdrant 연결 실패
1. Qdrant 컨테이너(`qdrant-rag`)가 실행 중인지 확인: `docker ps | grep qdrant`
2. Qdrant와 RAG API가 같은 네트워크에 있는지 확인
3. 네트워크에 연결: `docker network connect rag-network qdrant-rag`

### 임베딩 모델 다운로드 실패
인터넷 연결을 확인하고, Hugging Face에 접근 가능한지 확인합니다.

### LLM API 연결 실패
1. LLM API 서버가 실행 중인지 확인: `curl http://your-llm-server:port/v1/models`
2. 네트워크 연결 확인 (컨테이너에서 LLM API 서버 접근 가능해야 함)
3. .env 파일의 LLM_API_URL과 LLM_MODEL이 올바르게 설정되었는지 확인

## 성능 최적화

### 임베딩 모델 캐싱
첫 실행 시 모델 다운로드가 필요합니다. Dockerfile에서 사전 다운로드를 설정하여 초기 실행 시간을 단축할 수 있습니다.

### GPU 메모리 관리
대용량 모델 사용 시 GPU 메모리가 부족할 수 있습니다. 필요시 `device='cpu'`로 변경하거나 더 작은 모델을 사용하세요.

### 검색 성능
- `top_k` 값을 조정하여 검색 속도와 정확도의 균형을 맞추세요
- Qdrant의 HNSW 파라미터 (`ef_search`, `m`)는 컬렉션 생성 시 설정됩니다

## 라이선스

이 프로젝트는 내부 사용을 위한 것입니다.

## 문의

기술 지원이 필요한 경우 개발팀에 문의하세요.
