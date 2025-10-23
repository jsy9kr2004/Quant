"""
Parquet storage module with automatic validation and sample generation
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Optional, List, Union
from .data_validator import DataValidator


class ParquetStorage:
    """
    개선된 Parquet 저장소 (검증 기능 포함)

    Features:
    - Automatic validation after saving
    - Sample CSV generation for quick inspection
    - Efficient compression with configurable algorithms
    - Metadata tracking
    """

    def __init__(self, root_path: str, auto_validate: bool = True):
        """
        Initialize ParquetStorage

        Args:
            root_path: 루트 데이터 경로
            auto_validate: 저장 후 자동 검증 여부
        """
        self.root_path = Path(root_path)
        self.parquet_path = self.root_path / "parquet"
        self.view_path = self.root_path / "VIEW"
        self.sample_path = self.root_path / "samples"

        # 디렉토리 생성
        self.parquet_path.mkdir(parents=True, exist_ok=True)
        self.view_path.mkdir(parents=True, exist_ok=True)
        self.sample_path.mkdir(parents=True, exist_ok=True)

        self.auto_validate = auto_validate
        self.validator = DataValidator()

        logging.info(f"ParquetStorage initialized at: {self.root_path}")

    def save_parquet(self,
                    df: pd.DataFrame,
                    name: str,
                    compression: str = 'snappy',
                    save_sample: bool = True,
                    sample_size: int = 100,
                    partition_cols: Optional[List[str]] = None) -> bool:
        """
        Parquet 저장 + 검증 + 샘플 CSV 생성

        Args:
            df: 저장할 DataFrame
            name: 테이블 이름
            compression: 압축 알고리즘 ('snappy', 'gzip', 'zstd', 'brotli')
            save_sample: 샘플 CSV 생성 여부
            sample_size: 샘플 행 수
            partition_cols: 파티션 컬럼 (예: ['year'] - 연도별 분할)

        Returns:
            성공 여부
        """
        file_path = self.parquet_path / f"{name}.parquet"

        try:
            # 1. Parquet 저장
            logging.info(f"Saving {name}.parquet...")

            if partition_cols:
                # 파티셔닝 저장 (디렉토리 구조)
                df.to_parquet(
                    self.parquet_path / name,
                    engine='pyarrow',
                    compression=compression,
                    partition_cols=partition_cols,
                    index=False
                )
                file_path = self.parquet_path / name
            else:
                # 단일 파일 저장
                df.to_parquet(
                    file_path,
                    engine='pyarrow',
                    compression=compression,
                    index=False
                )

            # 2. 저장 정보 로깅
            if file_path.is_file():
                file_size = file_path.stat().st_size / 1024**2
            else:
                # 파티션된 경우 디렉토리 크기 계산
                file_size = sum(f.stat().st_size for f in file_path.rglob('*.parquet')) / 1024**2

            memory_size = df.memory_usage(deep=True).sum() / 1024**2
            compression_ratio = memory_size / file_size if file_size > 0 else 0

            logging.info(f"✅ Saved: {name}.parquet")
            logging.info(f"   📊 Rows: {len(df):,} | Columns: {len(df.columns)}")
            logging.info(f"   📁 File size: {file_size:.2f} MB")
            logging.info(f"   💾 Memory size: {memory_size:.2f} MB")
            logging.info(f"   🗜️  Compression ratio: {compression_ratio:.1f}x")

            # 3. 샘플 CSV 생성 (빠른 확인용)
            if save_sample:
                sample_file = self.sample_path / f"{name}_sample.csv"
                df.head(sample_size).to_csv(sample_file, index=False)
                sample_size_kb = sample_file.stat().st_size / 1024
                logging.info(f"   📄 Sample saved: {sample_file.name} ({sample_size_kb:.1f} KB)")

            # 4. 자동 검증
            if self.auto_validate and not partition_cols:  # 파티션은 검증 스킵
                logging.info(f"   🔍 Validating {name}...")
                result = self.validator.validate_file(str(file_path), name)

                if not result['passed']:
                    logging.error(f"   ❌ Validation failed for {name}")
                    for error in result['errors']:
                        logging.error(f"      {error}")
                    return False
                else:
                    logging.info(f"   ✅ Validation passed")

                    # 경고가 있으면 출력
                    for warning in result['warnings']:
                        logging.warning(f"      ⚠️  {warning}")

            return True

        except Exception as e:
            logging.error(f"❌ Failed to save {name}: {e}")
            return False

    def load_parquet(self,
                    name: str,
                    columns: Optional[List[str]] = None,
                    filters: Optional[List] = None) -> pd.DataFrame:
        """
        Parquet 파일 로드

        Args:
            name: 테이블 이름
            columns: 로드할 컬럼 리스트 (None이면 전체)
            filters: PyArrow 필터 (예: [('year', '=', 2020)])

        Returns:
            DataFrame
        """
        file_path = self.parquet_path / f"{name}.parquet"

        if not file_path.exists():
            # 파티션 디렉토리인지 확인
            partition_path = self.parquet_path / name
            if partition_path.exists() and partition_path.is_dir():
                file_path = partition_path
            else:
                raise FileNotFoundError(f"Parquet file not found: {file_path}")

        logging.info(f"Loading {name}.parquet...")

        df = pd.read_parquet(
            file_path,
            columns=columns,
            filters=filters,
            engine='pyarrow'
        )

        logging.info(f"✅ Loaded: {len(df):,} rows, {len(df.columns)} columns")
        return df

    def get_info(self, name: str) -> dict:
        """
        Parquet 파일 정보 조회 (메타데이터만, 데이터 로드 없음)

        Args:
            name: 테이블 이름

        Returns:
            파일 정보 딕셔너리
        """
        file_path = self.parquet_path / f"{name}.parquet"

        if not file_path.exists():
            return {'error': 'File not found'}

        # PyArrow로 메타데이터만 읽기 (빠름)
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(file_path)
        metadata = parquet_file.metadata
        schema = parquet_file.schema_arrow

        return {
            'name': name,
            'rows': metadata.num_rows,
            'row_groups': metadata.num_row_groups,
            'columns': len(schema),
            'column_names': schema.names,
            'dtypes': {field.name: str(field.type) for field in schema},
            'size_mb': file_path.stat().st_size / 1024**2,
            'created': metadata.created_by if hasattr(metadata, 'created_by') else None
        }

    def list_tables(self) -> List[str]:
        """저장된 모든 테이블 이름 반환"""
        tables = []

        # 파일
        for file_path in self.parquet_path.glob("*.parquet"):
            tables.append(file_path.stem)

        # 파티션 디렉토리
        for dir_path in self.parquet_path.iterdir():
            if dir_path.is_dir() and list(dir_path.glob("*.parquet")):
                tables.append(dir_path.name)

        return sorted(tables)

    def delete_table(self, name: str) -> bool:
        """테이블 삭제"""
        file_path = self.parquet_path / f"{name}.parquet"
        partition_path = self.parquet_path / name

        try:
            if file_path.exists():
                file_path.unlink()
                logging.info(f"🗑️  Deleted: {name}.parquet")
            elif partition_path.exists() and partition_path.is_dir():
                import shutil
                shutil.rmtree(partition_path)
                logging.info(f"🗑️  Deleted: {name}/ (partitioned)")
            else:
                logging.warning(f"Table not found: {name}")
                return False

            # 샘플 파일도 삭제
            sample_file = self.sample_path / f"{name}_sample.csv"
            if sample_file.exists():
                sample_file.unlink()

            return True

        except Exception as e:
            logging.error(f"Failed to delete {name}: {e}")
            return False

    def validate_all_tables(self) -> dict:
        """모든 테이블 검증"""
        return self.validator.validate_all(str(self.parquet_path))

    def generate_validation_report(self, output_file: str = "validation_report.txt") -> str:
        """검증 리포트 생성"""
        results = self.validate_all_tables()
        return self.validator.generate_report(results, output_file)

    def get_statistics(self) -> dict:
        """저장소 전체 통계"""
        tables = self.list_tables()
        total_size = 0
        total_rows = 0

        stats = {
            'total_tables': len(tables),
            'tables': []
        }

        for table in tables:
            try:
                info = self.get_info(table)
                total_size += info.get('size_mb', 0)
                total_rows += info.get('rows', 0)
                stats['tables'].append({
                    'name': table,
                    'rows': info.get('rows', 0),
                    'size_mb': info.get('size_mb', 0)
                })
            except Exception as e:
                logging.warning(f"Failed to get info for {table}: {e}")

        stats['total_size_mb'] = total_size
        stats['total_rows'] = total_rows

        return stats
