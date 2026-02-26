import weaviate
import logging

from logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

# 설정
WEAVIATE_HOST = "SERVER ADRESS"

# 1. v4 클라이언트 연결 (필수 인자 추가)
client = weaviate.connect_to_custom(
    http_host=WEAVIATE_HOST,
    http_port=8080,
    http_secure=False,      # 추가
    grpc_host=WEAVIATE_HOST,
    grpc_port=50051,
    grpc_secure=False       # 추가
)

try:
    # 2. 현재 존재하는 모든 컬렉션 목록 확인
    collections = client.collections.list_all()
    existing_names = list(collections.keys())
    print(f"🔍 현재 DB에 존재하는 컬렉션: {existing_names}")

    # 3. 삭제할 그래프 관련 타겟 목록
    target_to_delete = ["GraphEntity", "GraphEvent", "GraphRelation"]

    for class_name in target_to_delete:
        if class_name in existing_names:
            client.collections.delete(class_name)
            print(f"✅ 그래프 전용 컬렉션 삭제 완료: {class_name}")
        else:
            print(f"ℹ️ 삭제 건너뜀 (존재하지 않음): {class_name}")

    # 4. TextDocument 보존 여부 확인
    if "TextDocument" in existing_names:
        print(f"🛡️ 안전 확인: 'TextDocument' 클래스는 삭제되지 않고 유지되었습니다.")

except Exception as e:
    logger.error(f"❌ 오류 발생: {e}")
finally:
    client.close()
    print("🔌 클라이언트 연결 종료")