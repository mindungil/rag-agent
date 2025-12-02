# NVIDIA CUDA 지원 Python 베이스 이미지
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# 환경 변수 설정
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Seoul \
    PYTHONDONTWRITEBYTECODE=1 \
    LANG=ko_KR.UTF-8 \
    LC_ALL=ko_KR.UTF-8 \
    LANGUAGE=ko_KR:ko

# 시스템 패키지 업데이트 및 필수 패키지 설치
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    curl \
    git \
    locales \
    && rm -rf /var/lib/apt/lists/*

# 한글 로케일 생성 및 설정
RUN sed -i '/ko_KR.UTF-8/s/^# //g' /etc/locale.gen && \
    locale-gen ko_KR.UTF-8 && \
    update-locale LANG=ko_KR.UTF-8 LC_ALL=ko_KR.UTF-8

# 작업 디렉토리 생성
WORKDIR /workspace/rag_agent

# Python 의존성 파일 복사 및 설치
COPY requirements.txt .
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# 임베딩 모델 사전 다운로드
RUN python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-large-instruct')"

# 애플리케이션 코드 복사
COPY main.py models.py openai_models.py rag_service.py auth.py ./

# 문서 디렉토리 생성 (마운트 포인트)
RUN mkdir -p /workspace/rag_agent/documents/static

# 포트 노출
EXPOSE 8000

# 헬스체크 설정
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 서버 실행
CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

