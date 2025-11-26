#!/bin/bash

# RAG API 테스트 스크립트

echo "==================================="
echo "RAG API 서버 테스트"
echo "==================================="
echo ""

# 1. 헬스 체크
echo "[1] 헬스 체크"
echo "-----------------------------------"
curl -s http://localhost:8000/health | python3 -m json.tool
echo ""
echo ""

# 2. 루트 엔드포인트
echo "[2] 루트 엔드포인트"
echo "-----------------------------------"
curl -s http://localhost:8000/ | python3 -m json.tool
echo ""
echo ""

# 3. 질의응답 테스트
echo "[3] 질의응답 테스트"
echo "-----------------------------------"
echo "질문: 휴가 신청은 어떻게 하나요?"
echo ""
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "휴가 신청은 어떻게 하나요?",
    "top_k": 3,
    "temperature": 0.7,
    "max_tokens": 512
  }' | python3 -m json.tool
echo ""
echo ""

echo "==================================="
echo "테스트 완료"
echo "==================================="
