import os
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Set, Tuple
from PIL import Image
import logging
import sys
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import fitz  # PyMuPDF
from pdf2image import convert_from_path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.data_pipeline.pipe.bootstrap import (
    ensure_backend_root,
    configure_logging,
    get_backend_dir,
    get_data_pipeline_dir,
)

ensure_backend_root()
configure_logging()
logger = logging.getLogger(__name__)

# =============================================================================
# 실행 설정 매개변수 (여기서 모든 설정을 컨트롤)
# =============================================================================

BACKEND_DIR = get_backend_dir()
YourCompany_DATA_DIR = get_data_pipeline_dir()

# 입력 PDF 파일 경로
PDF_PATH = YourCompany_DATA_DIR / "doc" / "원가회계데이터샘플-250401.pdf"

# 출력 디렉토리
OUTPUT_DIR = YourCompany_DATA_DIR / "pipe" / "e"

# 변환 설정
DPI = 72  # 변환 DPI
LAYOUT_LONG_SIDE = 1280  # PP-DocLayout_plus-L 권장 값
REMOVE_MARKS = True  # 인쇄 마크 제거 여부
MARGIN_PERCENT = 0.00  # 여백 제거 비율
SKIP_EXISTING = True  # 기존 파일 건너뛰기 여부
RESUME = True  # 이어하기 여부
MAX_WORKERS = 256  # 최대 작업자 수
FILTER_BY_LEVEL = True  # PDF 레벨에 따라 페이지 필터링 여부

# =============================================================================
# 기본 매개변수 설정 (하위 호환성)
# =============================================================================
DEFAULT_LAYOUT_LONG_SIDE = LAYOUT_LONG_SIDE
DEFAULT_DPI = DPI
DEFAULT_OUTPUT_DIR = OUTPUT_DIR
DEFAULT_MARGIN_PERCENT = MARGIN_PERCENT
DEFAULT_REMOVE_MARKS = REMOVE_MARKS
DEFAULT_SKIP_EXISTING = SKIP_EXISTING
DEFAULT_RESUME = RESUME
DEFAULT_MAX_WORKERS = MAX_WORKERS
DEFAULT_FILTER_BY_LEVEL = FILTER_BY_LEVEL

class PDF2PNGForLayout:
    """PDF를 300 DPI PNG로 변환한 뒤 레이아웃 감지에 맞게 리사이즈하는 변환기
    pdftoppm을 직접 활용하여 병렬 처리로 최적화된 성능 제공"""

    # def __init__(self, output_dir: str = "converted_images_300dpi"):
    #     self.output_dir = Path(output_dir)
    #     # self.output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # 내부 유틸
    # ---------------------------------------------------------------------

    @staticmethod
    def _resize_for_layout(img: Image.Image, long_side: int = DEFAULT_LAYOUT_LONG_SIDE) -> Image.Image:
        """긴 변이 long_side를 넘으면 비율 유지하며 축소"""
        w, h = img.size
        max_side = max(w, h)
        if max_side <= long_side:
            return img  # 리사이즈 불필요
        scale = long_side / float(max_side)
        new_size = (int(w * scale), int(h * scale))
        logger.debug(f"리사이즈: {w}x{h} → {new_size[0]}x{new_size[1]}")
        return img.resize(new_size, Image.LANCZOS)

    # ------------------------------------------------------------------
    # 공개 API
    # ------------------------------------------------------------------
    def convert_pdf(
        self,
        pdf_path: str,
        *,
        output_dir: Optional[str] = None,
        dpi: int = DEFAULT_DPI,
        long_side: int = DEFAULT_LAYOUT_LONG_SIDE,
        remove_marks: bool = DEFAULT_REMOVE_MARKS,
        margin_percent: float = DEFAULT_MARGIN_PERCENT,
        skip_existing: bool = DEFAULT_SKIP_EXISTING,
        resume: bool = DEFAULT_RESUME,
        max_workers: int = DEFAULT_MAX_WORKERS,
        filter_by_level: bool = DEFAULT_FILTER_BY_LEVEL
    ) -> List[str]:
        """PDF → PNG 변환 (300 DPI) + 레이아웃 감지용 리사이즈
        
        구조:
        1. PDF 모든 페이지에 level 확인하여 필터링
        2. 필터링된 페이지만 이미지로 변환
        3. 리사이징
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")
        
        output_dir = Path(output_dir) if output_dir else self.output_dir / pdf_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"PDF 변환 시작: {pdf_path}")
        logger.info(f"출력 디렉토리: {output_dir}")
        
        # 이미 변환된 페이지 확인 (resume)
        last_processed = self._find_last_page(output_dir) if resume else 0
        
        # =================================================================
        # 1단계: PDF 모든 페이지에 level 확인하여 필터링
        # =================================================================
        pages_to_convert = set()
        has_level_2_or_higher = False
        total_pages = 0
        
        # PDF 총 페이지 수 확인
        try:
            doc = fitz.open(str(pdf_path))
            total_pages = len(doc)
            doc.close()
            logger.info(f"PDF 총 페이지 수: {total_pages}")
        except Exception as e:
            logger.error(f"PDF 페이지 수 확인 오류: {e}")
            raise
        
        if filter_by_level:
            logger.info("✅ 1단계: PDF 구조 분석 시작 (레벨에 따른 페이지 필터링)")
            has_level_2_or_higher, pages_to_convert = self._analyze_pdf_structure(str(pdf_path))
            
            if has_level_2_or_higher:
                logger.info(f"✅ 필터링 결과: PDF에 레벨 2 이상이 존재합니다.")
                logger.info(f"✅ 레벨 1 페이지를 제외한 {len(pages_to_convert)}개 페이지만 변환합니다.")
                logger.info(f"✅ 변환할 페이지: {sorted(list(pages_to_convert))[:10]}{'...' if len(pages_to_convert) > 10 else ''}")
            else:
                logger.info("✅ 필터링 결과: PDF에 레벨 2 이상이 없습니다. 모든 페이지를 변환합니다.")
                pages_to_convert = set(range(total_pages))
        else:
            logger.info("✅ 1단계: 필터링 비활성화 - 모든 페이지를 변환합니다.")
            pages_to_convert = set(range(total_pages))
        
        # =================================================================
        # 2단계 & 3단계: 필터링된 페이지 변환 + 리사이징 (배치 병렬 처리)
        # =================================================================
        logger.info(f"✅ 2&3단계: 필터링된 {len(pages_to_convert)}개 페이지를 {max_workers}개씩 배치로 변환 + 리사이징 시작")
        
        # 필터링된 페이지만 변환
        filtered_pages = sorted(list(pages_to_convert))
        if not filtered_pages:
            logger.warning("⚠️ 변환할 페이지가 없습니다.")
            return []
        
        saved_files = []
        batch_size = max_workers  # 8개씩 배치 처리
        total_batches = (len(filtered_pages) + batch_size - 1) // batch_size
        
        logger.info(f"✅ 총 {len(filtered_pages)}개 페이지를 {total_batches}개 배치({batch_size}개씩)로 처리")
        
        # 배치별로 처리
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(filtered_pages))
            batch_pages = filtered_pages[start_idx:end_idx]
            
            logger.info(f"✅ 배치 {batch_idx + 1}/{total_batches}: 페이지 {len(batch_pages)}개 처리 시작")
            
            try:
                # 배치에 해당하는 페이지만 변환 (개별 페이지 지정)
                # pdf2image는 1부터 시작하므로 +1
                batch_page_numbers = [page + 1 for page in batch_pages]
                
                logger.info(f"✅ 배치 {batch_idx + 1} 변환 페이지: {batch_page_numbers[:5]}{'...' if len(batch_page_numbers) > 5 else ''} (DPI: {dpi})")
                
                # 각 페이지를 개별로 변환
                batch_images = []
                for page_num in batch_page_numbers:
                    try:
                        page_images = convert_from_path(
                            str(pdf_path),
                            dpi=dpi,
                            first_page=page_num,
                            last_page=page_num,
                            thread_count=1,  # 개별 페이지는 단일 스레드
                            use_pdftocairo=True
                        )
                        if page_images:
                            batch_images.extend(page_images)
                    except Exception as e:
                        logger.warning(f"⚠️ 페이지 {page_num} 변환 실패: {e}")
                        continue
                
                logger.info(f"✅ 배치 {batch_idx + 1} PDF 변환 완료: {len(batch_images)}개 페이지")
                
                # 배치 내에서 병렬 리사이징 처리
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    futures = {}
                    
                    for i, img in enumerate(batch_images):
                        # 실제 페이지 번호 계산
                        actual_page_idx = batch_pages[i]
                        page_num = actual_page_idx + 1  # 1부터 시작하는 페이지 번호
                        
                        # 이어하기 처리
                        if resume and page_num <= last_processed:
                            logger.debug(f"페이지 {page_num}: 이미 처리됨, 건너뛰")
                            continue
                        
                        # 출력 파일 경로 생성
                        output_path = output_dir / f"{page_num:03d}.png"
                        
                        # 기존 파일 건너뛰기 처리
                        if skip_existing and output_path.exists():
                            logger.debug(f"페이지 {page_num}: 기존 파일 존재, 건너뛰")
                            saved_files.append(str(output_path))
                            continue
                        
                        # 병렬 처리로 이미지 후처리 실행
                        futures[executor.submit(
                            self._process_pil_image,
                            img,
                            output_path,
                            remove_marks,
                            margin_percent,
                            long_side
                        )] = (page_num, output_path)
                    
                    # 병렬 처리 결과 수집
                    batch_completed = 0
                    for future in as_completed(futures):
                        page_num, output_path = futures[future]
                        try:
                            if future.result():
                                saved_files.append(str(output_path))
                                batch_completed += 1
                                logger.info(f"✅ 배치 {batch_idx + 1} - 페이지 {page_num} 처리 완료 ({batch_completed}/{len(futures)})")
                        except Exception as e:
                            logger.error(f"❌ 배치 {batch_idx + 1} - 페이지 {page_num} 처리 오류: {e}")
                
                logger.info(f"✅ 배치 {batch_idx + 1}/{total_batches} 완료: {len(batch_images)}개 페이지 처리")
                
            except Exception as e:
                logger.error(f"❌ 배치 {batch_idx + 1} 처리 중 오류 발생: {e}")
                continue  # 다음 배치로 계속
        
        logger.info(f"✅ 모든 배치 처리 완료: {len(saved_files)}개 PNG 생성")
        return saved_files
        
    def _process_image(self, png_file: Path, output_path: Path, remove_marks: bool, margin_percent: float, long_side: int) -> bool:
        """이미지 후처리 (병렬 처리용)"""
        try:
            # 이미지 로드
            img = Image.open(png_file)
            
            # 인쇄 마크 제거(옵션)
            if remove_marks:
                img = self._remove_print_marks(img, margin_percent)
            
            # 리사이즈
            img = self._resize_for_layout(img, long_side)
            
            # 저장
            img.save(output_path, "PNG", optimize=True, compress_level=6)  # 압축 레벨 조정 (9→6)
            return True
        except Exception as e:
            logger.error(f"이미지 처리 오류: {e}")
            return False
    
    def _process_pil_image(self, img: Image.Image, output_path: Path, remove_marks: bool, margin_percent: float, long_side: int) -> bool:
        """PIL Image 객체 후처리 (병렬 처리용)"""
        try:
            # 인쇄 마크 제거(옵션)
            if remove_marks:
                img = self._remove_print_marks(img, margin_percent)
            
            # 리사이즈
            img = self._resize_for_layout(img, long_side)
            
            # 저장
            img.save(output_path, "PNG", optimize=True, compress_level=6)
            return True
        except Exception as e:
            logger.error(f"이미지 처리 오류: {e}")
            return False

    # ------------------------------------------------------------------
    def _remove_print_marks(self, image: Image.Image, margin_percent: float) -> Image.Image:
        """PDF 인쇄 마크(테두리) 제거"""
        w, h = image.size
        mx = int(w * margin_percent)
        my = int(h * margin_percent)
        return image.crop((mx, my, w - mx, h - my))

    def _find_last_page(self, output_dir: Path) -> int:
        pngs = sorted(output_dir.glob("*.png"))
        if not pngs:
            return 0
        try:
            return max(int(p.stem) for p in pngs)
        except ValueError:
            return 0
            
    def _analyze_pdf_structure(self, pdf_path: str) -> Tuple[bool, Set[int]]:
        """PDF 구조를 분석하여 레벨 정보와 변환할 페이지 목록을 반환
        
        Args:
            pdf_path: PDF 파일 경로
            
        Returns:
            Tuple[bool, Set[int]]: 
                - 첫 번째 요소: 레벨 2 이상이 존재하는지 여부
                - 두 번째 요소: 변환할 페이지 번호 집합 (0-based)
        """
        try:
            # PDF 문서 열기
            doc = fitz.open(pdf_path)
            total_pages = doc.page_count
            
            # 문서의 목차(TOC) 가져오기 (test_pdf_level_analysis.py와 동일하게 simple=False 사용)
            toc = doc.get_toc(simple=False)
            
            if not toc:
                # TOC가 없으면 모든 페이지 변환
                doc.close()
                return False, set(range(total_pages))
            
            # 레벨 2 이상이 존재하는지 확인
            levels = [item[0] for item in toc]
            has_level_2_or_higher = any(level >= 2 for level in levels)
            
            # 변환할 페이지 결정
            pages_to_convert = set()
            
            if has_level_2_or_higher:
                # test_pdf_level_analysis.py와 동일한 로직 적용
                # TOC 기반 페이지 레벨 매핑
                page_levels = {}
                for item in toc:
                    level, title, page = item[0], item[1], item[2]
                    if page not in page_levels or level < page_levels[page]:
                        page_levels[page] = level
                
                # 모든 페이지에 대해 레벨 할당 및 필터링
                current_level = None
                for page_num in range(1, total_pages + 1):
                    if page_num in page_levels:
                        current_level = page_levels[page_num]
                    
                    # 레벨 1이 아닌 페이지만 변환 (0-based로 변환)
                    if current_level != 1:
                        pages_to_convert.add(page_num - 1)
                
                logger.info(f"레벨 2 이상이 존재하므로 레벨 1 페이지를 제외합니다.")
                level_1_pages = [item[2] for item in toc if item[0] == 1]
                logger.info(f"제외될 레벨 1 페이지: {level_1_pages}")
                logger.info(f"변환될 페이지 수: {len(pages_to_convert)}/{total_pages}")
            else:
                # 레벨 2 이상이 없으면 모든 페이지 변환
                pages_to_convert = set(range(total_pages))
                logger.info(f"레벨 2 이상이 없으므로 모든 페이지를 변환합니다.")
                logger.info(f"변환될 페이지 수: {total_pages}/{total_pages}")
            
            doc.close()
            return has_level_2_or_higher, pages_to_convert
        
        except Exception as e:
            logger.error(f"PDF 구조 분석 오류: {e}")
            # 오류 발생 시 모든 페이지 변환
            try:
                doc = fitz.open(pdf_path)
                total_pages = doc.page_count
                doc.close()
                return False, set(range(total_pages))
            except:
                return False, set()

# ----------------------------------------------------------------------
# 메인 실행 함수
# ----------------------------------------------------------------------
def main():
    """메인 실행 함수 - 상단 매개변수를 사용하여 실행"""
    logger.info("=" * 80)
    logger.info("📝 PDF to PNG 변환기 시작")
    logger.info("=" * 80)
    
    # 상단 매개변수 로깅
    logger.info(f"📁 입력 PDF: {PDF_PATH}")
    logger.info(f"📂 출력 디렉토리: {OUTPUT_DIR}")
    logger.info(f"⚙️ DPI: {DPI}")
    logger.info(f"⚙️ 레이아웃 긴 변: {LAYOUT_LONG_SIDE}px")
    logger.info(f"⚙️ 인쇄 마크 제거: {REMOVE_MARKS}")
    logger.info(f"⚙️ 여백 비율: {MARGIN_PERCENT}")
    logger.info(f"⚙️ 기존 파일 건너뛰기: {SKIP_EXISTING}")
    logger.info(f"⚙️ 이어하기: {RESUME}")
    logger.info(f"⚙️ 최대 작업자 수: {MAX_WORKERS}")
    logger.info(f"⚙️ 레벨 필터링: {FILTER_BY_LEVEL}")
    logger.info("=" * 80)
    
    try:
        # PDF2PNGForLayout 인스턴스 생성
        converter = PDF2PNGForLayout(OUTPUT_DIR)
        
        # PDF 변환 실행
        result_files = converter.convert_pdf(
            PDF_PATH,
            output_dir=OUTPUT_DIR,
            dpi=DPI,
            long_side=LAYOUT_LONG_SIDE,
            remove_marks=REMOVE_MARKS,
            margin_percent=MARGIN_PERCENT,
            skip_existing=SKIP_EXISTING,
            resume=RESUME,
            max_workers=MAX_WORKERS,
            filter_by_level=FILTER_BY_LEVEL
        )
        
        logger.info("=" * 80)
        logger.info(f"✅ 변환 작업이 성공적으로 완료되었습니다!")
        logger.info(f"✅ 생성된 파일 수: {len(result_files)}개")
        if result_files:
            logger.info(f"✅ 출력 디렉토리: {OUTPUT_DIR}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error("=" * 80)
        logger.error(f"❌ 변환 중 오류 발생: {e}")
        logger.error("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()
