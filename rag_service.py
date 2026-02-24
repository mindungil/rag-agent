"""
RAG 서비스 로직
Qdrant 검색, 임베딩 생성, LLM 답변 생성을 처리합니다.
"""

import os
import logging
from typing import List, Dict, Any
from urllib.parse import quote

import torch
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, SearchRequest
import httpx

from models import DocumentReference

logger = logging.getLogger(__name__)


class RAGService:
    """RAG 처리를 위한 서비스 클래스"""

    SYSTEM_PROMPT = """당신은 업무편람 도우미입니다. 제공된 문서를 바탕으로 정확하고 상세한 답변을 제공하세요.
답변 시 다음 규칙을 따르세요:
1. 제공된 문서의 내용을 기반으로 답변합니다.
2. 문서에 없는 내용은 추측하지 않습니다.
3. 명확하고 구조화된 답변을 제공합니다.
4. 필요시 번호나 단계를 사용하여 설명합니다."""

    def __init__(
        self,
        qdrant_host: str,
        qdrant_grpc_port: int,
        collection_name: str,
        embedding_model: str,
        llm_api_url: str,
        llm_model: str,
        download_base_url: str,
    ):
        self.qdrant_host = qdrant_host
        self.qdrant_grpc_port = qdrant_grpc_port
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        self.llm_api_url = llm_api_url
        self.llm_model = llm_model
        self.download_base_url = download_base_url

        self.qdrant_client: QdrantClient = None
        self.embedding_model: SentenceTransformer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_dimension = 1024

        logger.info(f"디바이스: {self.device}")

    async def initialize(self):
        """서비스 초기화 - 모델 로드 및 Qdrant 연결"""
        try:
            # Qdrant 클라이언트 연결 (gRPC)
            logger.info(f"Qdrant 연결 중... (gRPC: {self.qdrant_host}:{self.qdrant_grpc_port})")
            self.qdrant_client = QdrantClient(
                host=self.qdrant_host,
                grpc_port=self.qdrant_grpc_port,
                prefer_grpc=True,
                timeout=30
            )

            # 컬렉션 존재 확인
            collections = self.qdrant_client.get_collections()
            collection_names = [c.name for c in collections.collections]

            if self.collection_name not in collection_names:
                raise ValueError(
                    f"컬렉션 '{self.collection_name}'이 존재하지 않습니다. "
                    f"사용 가능한 컬렉션: {collection_names}"
                )

            # 컬렉션 정보 확인
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            logger.info(f"컬렉션 '{self.collection_name}' 연결 성공")
            logger.info(f"벡터 개수: {collection_info.points_count}")
            logger.info(f"벡터 차원: {collection_info.config.params.vectors.size}")

            # 임베딩 모델 로드
            logger.info(f"임베딩 모델 로드 중... ({self.embedding_model_name})")
            self.embedding_model = SentenceTransformer(
                self.embedding_model_name,
                device=self.device
            )
            logger.info(f"임베딩 모델 로드 완료 (device: {self.device})")

        except Exception as e:
            logger.error(f"초기화 실패: {e}")
            raise

    async def close(self):
        """리소스 정리"""
        if self.qdrant_client:
            self.qdrant_client.close()
            logger.info("Qdrant 연결 종료")

    async def health_check(self) -> Dict[str, Any]:
        """헬스 체크 - Qdrant 및 모델 상태 확인"""
        try:
            # Qdrant 연결 확인
            collection_info = self.qdrant_client.get_collection(self.collection_name)

            return {
                "status": "healthy",
                "qdrant_connected": True,
                "collection_name": self.collection_name,
                "collection_vector_count": collection_info.points_count,
                "embedding_dimension": self.embedding_dimension,
                "model_name": self.embedding_model_name,
                "device": self.device
            }
        except Exception as e:
            logger.error(f"헬스 체크 실패: {e}")
            return {
                "status": "unhealthy",
                "qdrant_connected": False,
                "collection_name": self.collection_name,
                "collection_vector_count": 0,
                "embedding_dimension": self.embedding_dimension,
                "model_name": self.embedding_model_name,
                "device": self.device
            }

    def _create_query_embedding(self, text: str) -> List[float]:
        """
        텍스트를 임베딩 벡터로 변환
        - normalize_embeddings=True 사용
        """
        logger.info(f"임베딩 생성 중... (길이: {len(text)} 문자)")

        # SentenceTransformer로 임베딩 생성
        embedding = self.embedding_model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
            device=self.device
        )

        # numpy array를 list로 변환
        embedding_list = embedding.tolist()

        logger.info(f"임베딩 생성 완료 (차원: {len(embedding_list)})")
        return embedding_list

    async def _search_documents(
        self,
        query_embedding: List[float],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Qdrant에서 유사 문서 검색
        - search() 사용
        - distance metric: cosine
        - ef_search=128, m=64 설정 (HNSW 파라미터)
        """
        logger.info(f"Qdrant 검색 시작 (top_k={top_k})")

        try:
            # query를 사용하여 검색
            # HNSW 파라미터는 인덱스 생성 시 설정되어 있어야 함
            results = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=top_k,
                with_payload=True,
                with_vectors=False
            ).points

            logger.info(f"검색 완료: {len(results)}개 문서 발견")

            # 결과 포맷팅
            documents = []
            for hit in results:
                documents.append({
                    "id": hit.id,
                    "score": hit.score,
                    "payload": hit.payload
                })

            return documents

        except Exception as e:
            logger.error(f"Qdrant 검색 실패: {e}")
            raise

    def _create_download_url(self, source_path: str) -> str:
        """
        문서 소스 경로로부터 다운로드 URL 생성
        예: filename = "file1.pdf" -> "https://example.com/files/download?path=documents/static/file1.pdf"
        """
        # 파일명만 추출 (경로가 포함되어 있을 수 있음)
        filename = os.path.basename(source_path)

        # URL 인코딩
        encoded_filename = quote(filename, safe='')

        # 다운로드 URL 생성 (프로토콜 포함)
        download_url = f"https://{self.download_base_url}{encoded_filename}"

        return download_url

    def _create_context_from_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> tuple[str, List[DocumentReference]]:
        """
        검색된 문서들로부터 컨텍스트 생성 및 참조 정보 추출
        """
        context_parts = []
        references = []
        seen_files = set()  # 중복 제거를 위한 파일명 추적

        for idx, doc in enumerate(documents, 1):
            payload = doc["payload"]
            score = doc["score"]

            # 텍스트 추출 (page_content 필드 사용)
            text = payload.get("page_content", "")

            # 메타데이터에서 파일명 및 소스 정보 추출
            metadata = payload.get("metadata", {})
            if isinstance(metadata, dict):
                # source 필드에서 실제 파일 경로 추출 (우선순위: source > filename)
                source = metadata.get("source", "")
                # source가 있으면 basename 사용, 없으면 filename 사용
                if source:
                    filename = os.path.basename(source)
                else:
                    filename = metadata.get("filename", f"문서_{idx}")
            else:
                filename = f"문서_{idx}"
                source = ""

            # 부고 관련 문서 필터링
            if "부고" in filename or "부고" in text[:100]:
                continue

            # 컨텍스트에 추가
            context_parts.append(f"[문서 {idx}: {filename}]\n{text}\n")

            # 중복 제거: 같은 파일명이면 추가하지 않음
            if filename not in seen_files:
                seen_files.add(filename)

                # 참조 정보 생성 (파일명과 다운로드 링크만)
                # source에서 추출한 실제 파일명 사용
                download_url = self._create_download_url(filename)

                references.append(
                    DocumentReference(
                        filename=filename,
                        download_url=download_url
                    )
                )

        context = "\n".join(context_parts)
        return context, references

    async def _generate_answer(
        self,
        prompt: str,
        context: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """
        OpenAI 호환 API를 호출하여 답변 생성
        """
        logger.info("LLM 답변 생성 중...")

        user_prompt = f"""참고 문서:
{context}

질문: {prompt}

답변:"""

        # OpenAI 호환 API 호출
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.llm_api_url}/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.llm_model,
                        "messages": [
                            {"role": "system", "content": self.SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt}
                        ],
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "stream": False
                    }
                )
                response.raise_for_status()

                result = response.json()
                answer = result["choices"][0]["message"]["content"]

                logger.info(f"답변 생성 완료 (길이: {len(answer)} 문자)")
                return answer

        except httpx.HTTPError as e:
            logger.error(f"LLM API 호출 실패: {e}")
            raise Exception(f"답변 생성 실패: {str(e)}")

    async def process_query_streaming(
        self,
        prompt: str,
        top_k: int = 5,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> Dict[str, Any]:
        """
        스트리밍 방식으로 RAG 파이프라인 실행
        1. 임베딩 생성
        2. Qdrant 검색
        3. 컨텍스트 구성
        4. LLM 스트리밍 답변 생성
        5. 제너레이터 반환
        """
        try:
            # 1. 질의 임베딩 생성
            query_embedding = self._create_query_embedding(prompt)

            # 2. 유사 문서 검색
            documents = await self._search_documents(query_embedding, top_k)

            if not documents:
                # 문서가 없을 경우 단순 메시지 스트리밍
                async def empty_stream():
                    yield "관련 문서를 찾을 수 없습니다. 질문을 다시 확인해주세요."

                return {
                    "stream": empty_stream(),
                    "references": []
                }

            # 3. 컨텍스트 및 참조 정보 생성
            context, references = self._create_context_from_documents(documents)

            # 4. LLM 스트리밍 답변 생성
            async def stream_generator():
                logger.info("LLM 스트리밍 답변 생성 중...")

                user_prompt = f"""참고 문서:
{context}

질문: {prompt}

답변:"""

                try:
                    async with httpx.AsyncClient(timeout=120.0) as client:
                        async with client.stream(
                            "POST",
                            f"{self.llm_api_url}/chat/completions",
                            headers={"Content-Type": "application/json"},
                            json={
                                "model": self.llm_model,
                                "messages": [
                                    {"role": "system", "content": self.SYSTEM_PROMPT},
                                    {"role": "user", "content": user_prompt}
                                ],
                                "temperature": temperature,
                                "max_tokens": max_tokens,
                                "stream": True
                            }
                        ) as response:
                            response.raise_for_status()

                            # SSE 형식 파싱
                            async for line in response.aiter_lines():
                                if line.startswith("data: "):
                                    data = line[6:]  # "data: " 제거

                                    if data.strip() == "[DONE]":
                                        break

                                    try:
                                        import json
                                        chunk_data = json.loads(data)

                                        # delta에서 content 추출
                                        if "choices" in chunk_data and len(chunk_data["choices"]) > 0:
                                            delta = chunk_data["choices"][0].get("delta", {})
                                            content = delta.get("content", "")

                                            if content:
                                                yield content
                                    except json.JSONDecodeError:
                                        continue

                    logger.info("스트리밍 답변 생성 완료")

                except httpx.HTTPError as e:
                    logger.error(f"LLM 스트리밍 API 호출 실패: {e}")
                    yield f"[오류: 답변 생성 실패 - {str(e)}]"

            # 5. 스트리밍 제너레이터 및 참조 정보 반환
            return {
                "stream": stream_generator(),
                "references": references
            }

        except Exception as e:
            logger.error(f"RAG 스트리밍 처리 중 오류: {e}", exc_info=True)
            raise

    async def process_query(
        self,
        prompt: str,
        top_k: int = 5,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> Dict[str, Any]:
        """
        전체 RAG 파이프라인 실행
        1. 임베딩 생성
        2. Qdrant 검색
        3. 컨텍스트 구성
        4. LLM 답변 생성
        5. 결과 반환
        """
        try:
            # 1. 질의 임베딩 생성
            query_embedding = self._create_query_embedding(prompt)

            # 2. 유사 문서 검색
            documents = await self._search_documents(query_embedding, top_k)

            if not documents:
                return {
                    "answer": "관련 문서를 찾을 수 없습니다. 질문을 다시 확인해주세요.",
                    "references": [],
                    "prompt": prompt,
                    "retrieved_count": 0
                }

            # 3. 컨텍스트 및 참조 정보 생성
            context, references = self._create_context_from_documents(documents)

            # 4. LLM 답변 생성
            answer = await self._generate_answer(
                prompt=prompt,
                context=context,
                temperature=temperature,
                max_tokens=max_tokens
            )

            # 5. 결과 반환
            return {
                "answer": answer,
                "references": references,
                "prompt": prompt,
                "retrieved_count": len(documents)
            }

        except Exception as e:
            logger.error(f"RAG 처리 중 오류: {e}", exc_info=True)
            raise
