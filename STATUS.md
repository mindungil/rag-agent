# RAG 업무편람 도우미 API - 프로젝트 상태

**최종 업데이트**: 2025-11-26
**상태**: ✅ 코드 작성 완료, 빌드 대기 중

---

## 📋 프로젝트 개요

FastAPI 기반 RAG(Retrieval-Augmented Generation) 서버
- Open-WebUI에서 질문을 받아 Qdrant로 문서 검색 후 LLM으로 답변 생성
- 참조 문서 다운로드 링크 포함

---

## 🏗️ 아키텍처

```
Open-WebUI (채팅 UI)
    ↓ HTTP POST /query
RAG API 서버 (:8000) ← 이 프로젝트
    ↓ gRPC :6334
Qdrant (qdrant-rag 컨테이너)

RAG API 서버
    ↓ HTTP POST /chat/completions
LLM API (192.168.0.201:30020/v1)
```

---

## 📁 생성된 파일

### 핵심 코드
- **main.py** - FastAPI 메인 애플리케이션, 라이프사이클 관리
- **models.py** - Pydantic 모델 (QueryRequest, QueryResponse, HealthResponse 등)
- **rag_service.py** - RAG 로직 (임베딩, 검색, LLM 호출)

### 인프라
- **Dockerfile** - NVIDIA CUDA 12.1 기반, Python 3.10
- **docker-compose.yml** - GPU 지원, 볼륨 마운트, 네트워크 설정
- **requirements.txt** - Python 의존성

### 문서/설정
- **README.md** - 상세 사용 가이드
- **.gitignore** - Git 제외 파일
- **.dockerignore** - Docker 빌드 제외 파일
- **test_api.sh** - API 테스트 스크립트

---

## ⚙️ 주요 설정

### 1. Qdrant 연결
```python
qdrant_host="qdrant-rag"        # 컨테이너 이름으로 직접 연결
qdrant_grpc_port=6334           # gRPC 포트
collection_name="v1"            # 컬렉션 이름
```

**중요**: 실행 전 네트워크 연결 필요
```bash
docker network connect rag-network qdrant-rag
```

### 2. 임베딩 모델
```python
model = "intfloat/multilingual-e5-large-instruct"
dimension = 1024
normalize_embeddings = True
device = "cuda"  # GPU 자동 감지
```

### 3. LLM API (OpenAI 호환)
```python
llm_api_url = "http://192.168.0.201:30020/v1"
llm_model = "ChatGPT-oss-120B"
endpoint = "/chat/completions"
```

### 4. 파일 다운로드
```python
# 호스트 경로
host_path = "/mnt/ssd2/downloads/jmbadpt_20251121/file"

# 컨테이너 경로
container_path = "/workspace/rag_agent/documents/static"

# 다운로드 URL 형식
url = "ai.jb.go.kr/files/download?path=documents/static/{filename}"
```

---

## 🚀 실행 방법

### 1단계: 빌드 및 실행
```bash
cd /home/gil/rag-agent
docker compose up -d --build
```

### 2단계: Qdrant 네트워크 연결 (최초 1회만)
```bash
docker network connect rag-network qdrant-rag
```

### 3단계: 로그 확인
```bash
docker compose logs -f rag-api
```

### 4단계: 헬스 체크
```bash
curl http://localhost:8000/health
```

---

## 🔍 API 엔드포인트

### GET /health
서비스 상태 확인
```bash
curl http://localhost:8000/health
```

### POST /query
RAG 질의응답
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "휴가 신청은 어떻게 하나요?",
    "top_k": 5,
    "temperature": 0.7,
    "max_tokens": 1024
  }'
```

**응답 예시:**
```json
{
  "answer": "휴가 신청은 다음과 같이...",
  "references": [
    {
      "source": "휴가관리규정.pdf",
      "download_url": "ai.jb.go.kr/files/download?path=documents/static/휴가관리규정.pdf",
      "score": 0.92,
      "content_preview": "..."
    }
  ],
  "prompt": "휴가 신청은 어떻게 하나요?",
  "retrieved_count": 5
}
```

---

## ⚠️ 주의사항

### 필수 선행 조건
1. ✅ Qdrant 컨테이너 실행 중 (`qdrant-rag`)
2. ✅ LLM API 서버 실행 중 (`192.168.0.201:30020`)
3. ✅ NVIDIA GPU + Container Toolkit 설치
4. ✅ 문서 파일 경로 존재 (`/mnt/ssd2/downloads/jmbadpt_20251121/file`)

### 초기 실행 시
- 임베딩 모델 다운로드에 1-2분 소요 (약 1.3GB)
- Dockerfile에서 사전 다운로드 설정됨

---

## 🐛 트러블슈팅

### 문제 1: Qdrant 연결 실패
```
ERROR: Could not resolve qdrant-rag
```
**해결:**
```bash
docker network connect rag-network qdrant-rag
```

### 문제 2: LLM API 연결 실패
```
ERROR: LLM API 호출 실패
```
**확인:**
```bash
curl http://192.168.0.201:30020/v1/models
```

### 문제 3: GPU 인식 안 됨
```
device: cpu (expected: cuda)
```
**확인:**
```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

---

## 📝 TODO (남은 작업)

- [ ] **첫 실행 및 테스트**
  ```bash
  docker compose up -d --build
  docker network connect rag-network qdrant-rag
  curl http://localhost:8000/health
  ./test_api.sh
  ```

- [ ] **Open-WebUI 연동 테스트**
  - Open-WebUI에서 RAG API 엔드포인트 설정
  - 실제 질의응답 테스트
  - 다운로드 링크 동작 확인

- [ ] **성능 최적화 (선택)**
  - top_k 값 조정
  - temperature 튜닝
  - GPU 메모리 사용량 모니터링

---

## 📚 참고

- **API 문서**: http://localhost:8000/docs (Swagger UI)
- **ReDoc**: http://localhost:8000/redoc
- **README.md**: 상세 사용 가이드 참조

---

**프로젝트 디렉토리**: `/home/gil/rag-agent`
**컨테이너 이름**: `rag-agent-api`
**포트**: `8000`
