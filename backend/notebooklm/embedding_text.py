#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import json
import logging
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
import markdown
from bs4 import BeautifulSoup
import argparse  # <--- [수정] 이 줄이 추가되었습니다!
from tqdm import tqdm
from datetime import datetime
import hashlib
import uuid

# 💡 Sentence Transformers와 Weaviate 클라이언트 임포트
from shared_embedding import shared_embedding
import weaviate
from weaviate.util import generate_uuid5
from weaviate.exceptions import UnexpectedStatusCodeException

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# 💡 Weaviate 및 Config 설정
# =============================================================================

class RAGConfig:
    """Weaviate 연결 및 클래스 이름 설정"""
    def __init__(self):
        self.WEAVIATE_URL = "http://localhost:8080"
        self.WEAVIATE_TEXT_CLASS = "TextDocument"
        # 💡 참고: 이 파일에서 WEAVIATE_DOCUMENT_CLASS는 사용되지 않으므로 제거해도 무방합니다.

    def get_weaviate_client(self):
        """Weaviate 클라이언트를 초기화하고 반환합니다."""
        try:
            # v4 API 사용
            host = self.WEAVIATE_URL.replace('http://', '').replace('https://', '').split(':')[0]
            port = int(self.WEAVIATE_URL.split(':')[-1]) if ':' in self.WEAVIATE_URL.split('//')[-1] else 8080
            
            client = weaviate.connect_to_custom(
                http_host=host,
                http_port=port,
                http_secure=False,
                grpc_host=host,
                grpc_port=50051,
                grpc_secure=False
            )
            if client.is_ready():
                return client
            else:
                logger.error(f"Weaviate is not ready at {self.WEAVIATE_URL}")
                client.close()
                return None
        except Exception as e:
            logger.error(f"Weaviate 클라이언트 초기화 중 오류 발생: {str(e)}")
            return None

def create_schema(client):
    """필요한 Weaviate 스키마를 생성합니다. (v4 API)"""
    from weaviate.classes.config import Configure, Property, DataType
    
    config = RAGConfig()
    
    try:
        # v4 API: collection이 이미 존재하는지 확인
        if client.collections.exists(config.WEAVIATE_TEXT_CLASS):
            logger.info(f"Collection '{config.WEAVIATE_TEXT_CLASS}' already exists")
            return
        
        # v4 API로 collection 생성
        client.collections.create(
            name=config.WEAVIATE_TEXT_CLASS,
            description="Text chunks and their embeddings.",
            vectorizer_config=Configure.Vectorizer.none(),
            properties=[
                Property(name="text", data_type=DataType.TEXT),
                Property(name="title", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT),
                Property(name="page_num", data_type=DataType.INT),
                Property(name="chunk_id", data_type=DataType.TEXT),
                Property(name="document_uuid", data_type=DataType.TEXT),
                Property(name="metadata", data_type=DataType.OBJECT, nested_properties=[
                    Property(name="document_id", data_type=DataType.TEXT),
                    Property(name="content_type", data_type=DataType.TEXT),
                    Property(name="file_path", data_type=DataType.TEXT)
                ]),
            ]
        )
        logger.info(f"Created schema collection: {config.WEAVIATE_TEXT_CLASS}")
            
    except Exception as e:
        logger.error(f"Weaviate 스키마 생성 중 오류 발생: {str(e)}")
# -----------------------------------------------------------------------


# =============================================================================
# 데이터 구조 및 프로세서 클래스 (이전과 동일)
# =============================================================================

@dataclass
class ChunkInfo:
    """청크 정보 구조체"""
    chunk_id: str
    text: str
    raw_text: str
    start_pos: int
    end_pos: int
    token_start: int
    token_end: int
    chunk_index: int
    document_id: str
    content_type: str
    metadata: Dict[str, Any]

@dataclass
class VectorEntry:
    """벡터 저장 엔트리"""
    id: str
    document_id: str
    chunk_id: str
    text: str
    raw_text: str
    embedding: np.ndarray
    content_type: str
    metadata: Dict[str, Any]
    created_at: datetime

class MixedContentProcessor:
    def __init__(self):
        self.md_parser = markdown.Markdown(extensions=['fenced_code', 'tables'])
        self.latex_patterns = {
            'inline_paren': re.compile(r'\\\((.*?)\\\)', re.DOTALL),
            'inline_dollar': re.compile(r'\$([^$]+)\$'),
            'display_bracket': re.compile(r'\\\[(.*?)\\]', re.DOTALL),
            'display_dollar': re.compile(r'\$\$(.*?)\$\$', re.DOTALL),
        }
        self.table_pattern = re.compile(r'<table.*?</table>', re.DOTALL | re.IGNORECASE)
        self.code_pattern = re.compile(r'```.*?```', re.DOTALL)

    def process_markdown(self, file_path: str) -> Dict[str, Any]:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        html = self.md_parser.convert(content)
        soup = BeautifulSoup(html, 'html.parser')
        sections = []
        for element in soup.find_all(['p', 'li', 'pre', 'blockquote', 'table']):
            text = element.get_text().strip()
            if not text:
                continue
            raw_text = str(element)
            content_type = self._identify_content_type(raw_text)
            sections.append({
                'text': text,
                'raw_text': raw_text,
                'content_type': content_type,
                'metadata': {}
            })
        return {'sections': sections, 'raw_content': content}

    def _identify_content_type(self, text: str) -> str:
        if any(pattern.search(text) for pattern in self.latex_patterns.values()):
            return 'latex'
        if self.table_pattern.search(text):
            return 'table'
        if self.code_pattern.search(text):
            return 'code'
        return 'text'

class EnhancedSemanticChunker:
    def __init__(self, max_chunk_size: int = 1000, overlap_size: int = 100):
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size

    def chunk_by_content_type(self, processed_data: Dict[str, Any]) -> List[Dict]:
        sections = processed_data['sections']
        chunks = []
        current_chunk_text = ""
        current_chunk_raw = ""
        for section in sections:
            if len(current_chunk_text) + len(section['text']) > self.max_chunk_size:
                if current_chunk_text:
                    chunks.append({'text': current_chunk_text, 'raw_text': current_chunk_raw, 'content_type': 'mixed'})
                current_chunk_text = section['text']
                current_chunk_raw = section['raw_text']
            else:
                current_chunk_text += "\n\n" + section['text']
                current_chunk_raw += "\n\n" + section['raw_text']
        if current_chunk_text:
            chunks.append({'text': current_chunk_text, 'raw_text': current_chunk_raw, 'content_type': 'mixed'})
        return chunks

class WeaviateVectorStore:
    """Weaviate 기반 벡터 저장소 (배치 저장 로직 포함)"""
    def __init__(self, session_id: Optional[str] = None):
        self.config = RAGConfig()
        self.session_id = session_id
        self.client = None
        self._init_client()

    def _init_client(self):
        self.client = self.config.get_weaviate_client()
        if self.client:
            logger.info(f"Initialized Weaviate client: {self.config.WEAVIATE_URL}")
        else:
            logger.warning("Weaviate 클라이언트 초기화 실패. 데이터 저장이 불가능합니다.")

    def add_vectors(self, vectors: List[VectorEntry]):
        if not vectors or not self.client:
            logger.warning("Weaviate 클라이언트가 초기화되지 않았거나 벡터가 없습니다. 데이터베이스 저장을 건너뜁니다.")
            return
        
        # 1. 배치 API를 사용하여 청크 데이터 저장 (v4)
        first_vec = vectors[0]
        metadata = first_vec.metadata
        file_path = metadata.get('file_path', '')
        filename = os.path.basename(file_path) if file_path else f"doc_{first_vec.document_id[:8]}"
        
        document_uuid = generate_uuid5(first_vec.document_id)

        document_metadata = {
            "document_id": first_vec.document_id,
        }
        if file_path:
            document_metadata["file_path"] = file_path
        content_type = getattr(first_vec, "content_type", None) or metadata.get("content_type")
        if content_type:
            document_metadata["content_type"] = content_type

        text_class_name = self.config.WEAVIATE_TEXT_CLASS
        collection = self.client.collections.get(text_class_name)
        
        # v4 API: batch context manager 사용
        with collection.batch.dynamic() as batch:
            for vec in tqdm(vectors, desc="Adding chunks to Weaviate batch"):
                chunk_uuid = generate_uuid5(vec.chunk_id)
                chunk_metadata = {
                    "document_id": vec.document_id,
                }
                if vec.content_type:
                    chunk_metadata["content_type"] = vec.content_type

                properties = {
                    "text": vec.text,
                    "title": metadata.get('title', ''),
                    "source": file_path,
                    "page_num": metadata.get('page_num', 0),
                    "chunk_id": vec.chunk_id,
                    "document_uuid": str(document_uuid),
                    "session_id": self.session_id or "default",
                    "metadata": {
                        **document_metadata,
                        **chunk_metadata,
                        "session_id": self.session_id or "default"
                    },
                }
                
                batch.add_object(
                    properties=properties,
                    uuid=chunk_uuid,
                    vector=vec.embedding.tolist()
                )

        logger.info(
            "Successfully added %d chunk vectors to Weaviate for document %s (%s)",
            len(vectors),
            filename,
            document_uuid,
        )

# =============================================================================
# your-modelLateChunker 클래스
# =============================================================================

class your-modelLateChunker:
    def __init__(self, chunk_size=1000, overlap_size=100, session_id: Optional[str] = None):
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.embedding_model = shared_embedding
        self.vector_store = WeaviateVectorStore(session_id=session_id)
        self.document_metadata = {}
        self.session_id = session_id
    def process_markdown_file(self, file_path: str) -> str:
        start_time = datetime.now()
        document_id = self._generate_document_id(file_path)
        
        content_processor = MixedContentProcessor()
        processed_data = content_processor.process_markdown(file_path)
        
        chunker = EnhancedSemanticChunker(self.chunk_size, self.overlap_size)
        chunks = chunker.chunk_by_content_type(processed_data)
        
        chunk_infos = self._get_chunk_boundaries(file_path, processed_data['raw_content'], chunks, document_id)
        vector_entries = self._perform_late_chunking(chunk_infos)
        
        if vector_entries:
            self.vector_store.add_vectors(vector_entries)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        self.document_metadata[document_id] = {
            'filename': os.path.basename(file_path),
            'file_path': file_path,
            'processing_time': processing_time,
            'total_chunks': len(vector_entries),
            'created_at': datetime.now()
        }
        return document_id

    def _generate_document_id(self, file_path: str) -> str:
        return hashlib.md5(file_path.encode()).hexdigest()

    def _get_chunk_boundaries(self, file_path: str, full_text: str, chunks: List[Dict], document_id: str) -> List[ChunkInfo]:
        chunk_infos = []
        current_pos = 0
        
        for i, chunk in enumerate(chunks):
            chunk_text = chunk['text']
            chunk_start = full_text.find(chunk_text.strip(), current_pos)
            if chunk_start == -1:
                chunk_start = current_pos 
            
            chunk_end = chunk_start + len(chunk_text.strip())
            
            chunk_info = ChunkInfo(
                chunk_id=f"{document_id}_chunk_{i}", 
                text=chunk_text,
                raw_text=chunk.get('raw_text', chunk_text),
                start_pos=chunk_start,
                end_pos=chunk_end,
                token_start=0,
                token_end=0,
                chunk_index=i,
                document_id=document_id,
                content_type=chunk.get('content_type', 'text'),
                metadata={'file_path': file_path} 
            )
            chunk_infos.append(chunk_info)
            current_pos = chunk_end
        return chunk_infos

    def _perform_late_chunking(self, chunk_infos: List[ChunkInfo]) -> List[VectorEntry]:
        vector_entries = []
        
        # 임베딩 모델이 로드되지 않은 경우 명시적으로 로드
        if not self.embedding_model.is_loaded:
            logger.info("Embedding model not loaded, attempting to load...")
            self.embedding_model.load_model()
        
        if not self.embedding_model.is_loaded:
            logger.error("Shared embedding model is still not loaded after attempting to load.")
            return []

        for info in tqdm(chunk_infos, desc="Embedding chunks"):
            try:
                embedding = self.embedding_model.generate_embedding(info.text, normalize=True)
                
                # 임베딩 유효성 검사
                if embedding is None:
                    logger.error(f"Embedding generation failed for chunk {info.chunk_id}: embedding is None")
                    continue
                
            except Exception as e:
                logger.error(f"Error generating embedding for chunk {info.chunk_id}: {str(e)}")
                if self.embedding_model.vector_dimension:
                        embedding = np.ones(self.embedding_model.vector_dimension) 
                else:
                    logger.error("Embedding dimension unknown, skipping chunk.")
                    continue

            # 유효한 임베딩만 VectorEntry에 추가
            if embedding is not None:
                entry = VectorEntry(
                    id=info.chunk_id,
                    document_id=info.document_id,
                    chunk_id=info.chunk_id,
                    text=info.text,
                    raw_text=info.raw_text,
                    embedding=embedding,
                    content_type=info.content_type,
                    metadata=info.metadata,
                    created_at=datetime.now()
                )
                vector_entries.append(entry)
            else:
                logger.warning(f"Skipping chunk {info.chunk_id} due to invalid embedding")
        return vector_entries

    def get_document_summary(self, document_id: str) -> Dict[str, Any]:
        if document_id in self.document_metadata:
            summary = self.document_metadata[document_id].copy()
            if 'created_at' in summary and isinstance(summary['created_at'], datetime):
                summary['created_at'] = summary['created_at'].isoformat()
            return summary
        return {"document_id": document_id, "status": "not_found"}

# =============================================================================
# 벡터 검색 함수
# =============================================================================

def search_vectors(query: str, top_k: int = 5) -> List[Dict]:
    config = RAGConfig()
    client = config.get_weaviate_client()
    if not client:
        print("Weaviate 클라이언트를 초기화할 수 없습니다.")
        return []
    
    try:
        query_embedding = shared_embedding.generate_embedding(query)
        if query_embedding is None:
            raise ValueError("Query embedding generation failed.")
    except Exception as e:
        logger.error(f"Error generating query embedding: {e}")
        return []
    
    try:
        # v4 API: collection을 통한 검색
        collection = client.collections.get(config.WEAVIATE_TEXT_CLASS)
        
        response = collection.query.near_vector(
            near_vector=query_embedding.tolist(),
            limit=top_k,
            return_metadata=['distance']
        )
        
        results = []
        for obj in response.objects:
            distance = obj.metadata.distance if obj.metadata.distance is not None else 1.0
            similarity_score = 1.0 - distance if 0.0 <= distance <= 1.0 else 0.0
            
            metadata = obj.properties.get("metadata", {})
            
            results.append({
                'score': similarity_score,
                'document_id': metadata.get('document_id', ''),
                'document_uuid': obj.properties.get('document_uuid', ''),
                'text': obj.properties.get('text', ''),
                'content_type': metadata.get('content_type', 'text'),
            })
        
        return results
    finally:
        client.close()

# =============================================================================
# Main 함수 (스키마 삭제 및 생성 순서 수정 완료)
# =============================================================================

def process_markdown_files(client, file_paths: List[str], session_id: Optional[str] = None):
    """여러 마운트다운 파일을 인덱싱합니다."""
    chunker = your-modelLateChunker(session_id=session_id)
    # 기존 코드에서 your-modelLateChunker가 사용하는 client는 내부적으로 생성하므로 
    # client 인자를 넘기지 않고 chunker 내부의 vector_store.client를 사용하거나 
    # 필요시 chunker 구조를 개선해야 하지만, 현재 구조 유지를 위해 세션 ID만 전달합니다.
    for file_path in file_paths:
        try:
            chunker.process_markdown_file(file_path)
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")

def main(input_file=None, recreate_schema=False, search_mode=False, session_id=None):
    """
    Main function for chunking and indexing
    
    Args:
        input_file: Input markdown file path for processing
        recreate_schema: Whether to recreate Weaviate schema
        search_mode: Whether to enter interactive search mode
    """
    # 명령줄에서 실행될 때만 argparse 사용
    if input_file is None and not recreate_schema and not search_mode:
        parser = argparse.ArgumentParser(description="your-model Late Chunking and Weaviate Indexing")
        parser.add_argument('--input-file', type=str, help='Input markdown file path for processing.')
        parser.add_argument('--recreate-schema', action='store_true', help='Recreate Weaviate schema.')
        parser.add_argument('--search', action='store_true', help='Interactive search mode.')
        args = parser.parse_args()
        input_file = args.input_file
        recreate_schema = args.recreate_schema
        search_mode = args.search
    
    # 간단한 객체로 변환 (기존 코드와 호환성 유지)
    class Args:
        pass
    args = Args()
    args.input_file = input_file
    args.recreate_schema = recreate_schema
    args.search = search_mode

    # [수정] config.py에서 RAGConfig를 가져오도록 수정
    try:
        from config import RAGConfig
        logger.info("config.py에서 RAGConfig 로드 성공.")
    except ImportError:
        logger.error("config.py를 찾을 수 없습니다. model3_late_chunking.py의 로컬 RAGConfig를 사용합니다.")
        # config.py가 없으면 이 파일의 로컬 RAGConfig를 사용 (기존 방식)
        pass

    config = RAGConfig()
    client = config.get_weaviate_client()
    
    if args.recreate_schema:
        if client:
            logger.warning("Weaviate 스키마 재생성을 시작합니다 (모든 데이터 삭제).")
            
            # vvvv [수정] vvvv
            # weaviate_utils.py에 정의된 (recreate=True를 지원하는) 
            # create_schema 함수를 임포트하여 강제 재생성
            try:
                from weaviate_utils import create_schema as create_utils_schema
                
                # recreate=True로 호출하여 delete_all() 실행
                success = create_utils_schema(client, recreate=True) 
                
                if success:
                    logger.info("스키마 재생성이 완료되었습니다.")
                else:
                    logger.error("스키마 재생성 중 오류가 발생했습니다.")

            except ImportError:
                logger.error("weaviate_utils.py를 임포트할 수 없습니다. 스키마 재생성 실패.")
            except Exception as e:
                logger.error(f"스키마 재생성 중 예외 발생: {e}")
            # ^^^^ [수정] ^^^^
        else:
            logger.error("Weaviate 클라이언트가 없어 스키마를 재생성할 수 없습니다.")
        return # 스키마 재생성 후 종료

    if args.search:
        while True:
            try:
                query = input("\n검색 쿼리: ")
                if query.lower() in ['exit', 'quit']:
                    break
                results = search_vectors(query)
                print(json.dumps(results, indent=2, ensure_ascii=False))
            except (KeyboardInterrupt, EOFError):
                break
        return

    if not args.input_file:
        parser.error("--input-file is required.")
        return

    try:
        if client:
            # [수정] weaviate_utils에서 스키마 생성 함수를 임포트하여
            # TextDocument와 ImageDocument 스키마가 모두 존재하는지 확인
            try:
                from weaviate_utils import create_schema as ensure_all_schema
                ensure_all_schema(client, recreate=False) # recreate=False로 존재 여부만 확인
            except ImportError:
                logger.warning("weaviate_utils.py를 찾을 수 없어 로컬 create_schema만 사용합니다.")
                # 임포트 실패 시 로컬 create_schema 사용 (기존 방식)
                if 'create_schema' in locals():
                    create_schema(client)

            logger.info(f"Weaviate client ready at {config.WEAVIATE_URL}")
        else:
            logger.warning("Weaviate client not ready. Indexing will fail.")
        
        chunker = your-modelLateChunker() 
        document_id = chunker.process_markdown_file(args.input_file)
        summary = chunker.get_document_summary(document_id)
        print("\nDocument Summary:")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        logger.info(f"Indexing completed. Total chunks: {summary.get('total_chunks', 0)}")
    except Exception as e:
        logger.error(f"FATAL ERROR during chunker processing: {e}")

if __name__ == "__main__":
    main()
