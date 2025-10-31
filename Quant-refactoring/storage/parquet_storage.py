"""자동 검증 및 샘플 생성 기능이 있는 Parquet 저장소 모듈입니다.

이 모듈은 내장 검증, 샘플 생성, 메타데이터 추적 기능이 있는 Parquet 파일 관리를 위한
포괄적인 ParquetStorage 클래스를 제공합니다. 자동 압축, 파티셔닝, 데이터 품질 검사를
지원합니다.

ParquetStorage 클래스는 PyArrow Parquet 작업을 래핑하고 다음을 추가합니다:
    - 저장 후 자동 데이터 검증
    - 빠른 검사를 위한 샘플 CSV 생성
    - 압축 통계 및 리포팅
    - 전체 데이터셋 로드 없이 메타데이터 쿼리
    - 저장된 모든 테이블의 배치 검증

사용 예시:
    자동 검증을 사용한 기본 사용법::

        from storage import ParquetStorage
        import pandas as pd

        # 자동 검증이 활성화된 저장소 초기화
        storage = ParquetStorage(
            root_path="/data/stocks",
            auto_validate=True
        )

        # 압축과 함께 DataFrame 저장
        df = pd.DataFrame({
            'symbol': ['AAPL', 'GOOGL', 'MSFT'],
            'price': [150.0, 2800.0, 300.0]
        })

        success = storage.save_parquet(
            df=df,
            name='stock_prices',
            compression='snappy',
            save_sample=True
        )

        # 컬럼 필터링으로 데이터 로드
        df_loaded = storage.load_parquet(
            name='stock_prices',
            columns=['symbol', 'price']
        )

        # 데이터 로드 없이 파일 정보 가져오기
        info = storage.get_info('stock_prices')
        print(f"Rows: {info['rows']}, Size: {info['size_mb']:.2f} MB")

Attributes:
    Module-level exports: ParquetStorage, DataValidator
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Optional, List, Union, Dict, Any
from .data_validator import DataValidator


class ParquetStorage:
    """검증 및 샘플 생성 기능이 강화된 Parquet 저장소 관리자입니다.

    이 클래스는 자동 데이터 검증, 샘플 CSV 생성, 포괄적인 메타데이터 추적이 포함된
    Parquet 파일 저장 및 검색을 위한 고수준 인터페이스를 제공합니다. 디렉토리 구조
    생성, 압축, 파티셔닝, 품질 검사를 처리합니다.

    저장소는 세 가지 주요 디렉토리를 생성합니다:
        - parquet/: Parquet 파일을 위한 메인 저장소
        - VIEW/: 처리된 데이터를 위한 뷰 테이블
        - samples/: 빠른 검사를 위한 CSV 샘플

    Attributes:
        root_path (Path): 모든 저장소 작업을 위한 루트 디렉토리.
        parquet_path (Path): Parquet 파일이 저장되는 디렉토리.
        view_path (Path): 뷰 테이블을 위한 디렉토리.
        sample_path (Path): 샘플 CSV 파일을 위한 디렉토리.
        auto_validate (bool): 저장 후 자동으로 데이터를 검증할지 여부.
        validator (DataValidator): 데이터 품질 검사를 위한 검증기 인스턴스.

    사용 예시:
        파티셔닝과 함께 저장소 초기화 및 데이터 저장::

            storage = ParquetStorage(
                root_path="/data/financial",
                auto_validate=True
            )

            # 연도 파티셔닝과 함께 저장
            df = pd.DataFrame({
                'date': pd.date_range('2020-01-01', periods=1000),
                'symbol': ['AAPL'] * 1000,
                'price': range(1000),
                'year': [2020] * 1000
            })

            storage.save_parquet(
                df=df,
                name='prices',
                partition_cols=['year'],
                compression='zstd'
            )

            # 저장된 모든 테이블 나열
            tables = storage.list_tables()
            print(f"Stored tables: {tables}")

            # 모든 테이블 검증
            results = storage.validate_all_tables()
    """

    def __init__(self, root_path: str, auto_validate: bool = True) -> None:
        """디렉토리 구조와 함께 ParquetStorage를 초기화합니다.

        Parquet 저장소, 뷰, 샘플을 위한 필요한 디렉토리를 생성합니다.
        품질 검사를 위한 데이터 검증기를 초기화합니다.

        Args:
            root_path (str): 모든 데이터 저장을 위한 루트 디렉토리 경로.
                존재하지 않으면 생성됩니다.
            auto_validate (bool, optional): True인 경우, 저장 후 자동으로
                데이터를 검증합니다. 기본값은 True.

        사용 예시:
            커스텀 검증 설정으로 저장소 초기화::

                # 자동 검증 사용
                storage = ParquetStorage(
                    root_path="/data/stocks",
                    auto_validate=True
                )

                # 자동 검증 없이 (신뢰할 수 있는 데이터에 대해 더 빠름)
                storage_fast = ParquetStorage(
                    root_path="/data/stocks",
                    auto_validate=False
                )
        """
        self.root_path = Path(root_path)
        self.parquet_path = self.root_path / "parquet"
        self.view_path = self.root_path / "VIEW"
        self.sample_path = self.root_path / "samples"

        # Create directory structure
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
        """검증 및 샘플 생성과 함께 DataFrame을 Parquet으로 저장합니다.

        선택적 압축, 파티셔닝, 자동 검증과 함께 DataFrame을 Parquet 형식으로
        저장합니다. 또한 빠른 검사를 위한 샘플 CSV를 생성합니다. 파일 크기,
        압축 비율, 검증 결과를 포함한 상세 통계를 로깅합니다.

        Args:
            df (pd.DataFrame): 저장할 DataFrame. 비어있으면 안 됩니다.
            name (str): 테이블 이름 (.parquet 확장자 제외). 파일명 또는
                파티션된 경우 디렉토리 이름으로 사용됩니다.
            compression (str, optional): 압축 알고리즘. 지원되는 값:
                'snappy' (빠름, 기본값), 'gzip' (더 작음), 'zstd' (균형),
                'brotli' (가장 작음). 기본값은 'snappy'.
            save_sample (bool, optional): True인 경우, 빠른 검사를 위한
                샘플 CSV를 저장합니다. 기본값은 True.
            sample_size (int, optional): 샘플 CSV의 행 수. 기본값은 100.
            partition_cols (Optional[List[str]], optional): 파티셔닝을 위한
                컬럼 이름. 파티션 값으로 디렉토리 구조를 생성합니다.
                예: ['year']는 year=2020/, year=2021/ 하위 디렉토리를 생성합니다.
                기본값은 None.

        Returns:
            bool: 저장 및 검증이 성공하면 True, 그렇지 않으면 False.

        Raises:
            Exception: 저장 작업이 실패하면 예외를 로깅하고 False를 반환합니다.

        사용 예시:
            다양한 압축 알고리즘으로 저장::

                # 빠른 압축 (기본값)
                storage.save_parquet(df, 'fast_data', compression='snappy')

                # 최대 압축
                storage.save_parquet(df, 'archive_data', compression='brotli')

                # 연도별 파티셔닝
                storage.save_parquet(
                    df=df,
                    name='historical_prices',
                    partition_cols=['year'],
                    compression='zstd'
                )

                # 샘플 생성 없이 저장
                storage.save_parquet(
                    df=large_df,
                    name='huge_dataset',
                    save_sample=False
                )

        Note:
            - 파티션된 데이터셋은 자동 검증을 건너뜁니다
            - 압축 비율은 memory_size / file_size로 계산됩니다
            - 검증 오류는 로깅되지만 예외를 발생시키지 않습니다
        """
        file_path = self.parquet_path / f"{name}.parquet"

        try:
            # 1. Save to Parquet format
            logging.info(f"Saving {name}.parquet...")

            if partition_cols:
                # Partitioned save (directory structure)
                df.to_parquet(
                    self.parquet_path / name,
                    engine='pyarrow',
                    compression=compression,
                    partition_cols=partition_cols,
                    index=False
                )
                file_path = self.parquet_path / name
            else:
                # Single file save
                df.to_parquet(
                    file_path,
                    engine='pyarrow',
                    compression=compression,
                    index=False
                )

            # 2. Calculate and log storage statistics
            if file_path.is_file():
                file_size = file_path.stat().st_size / 1024**2
            else:
                # For partitioned datasets, calculate total directory size
                file_size = sum(f.stat().st_size for f in file_path.rglob('*.parquet')) / 1024**2

            memory_size = df.memory_usage(deep=True).sum() / 1024**2
            compression_ratio = memory_size / file_size if file_size > 0 else 0

            logging.info(f"✅ Saved: {name}.parquet")
            logging.info(f"   📊 Rows: {len(df):,} | Columns: {len(df.columns)}")
            logging.info(f"   📁 File size: {file_size:.2f} MB")
            logging.info(f"   💾 Memory size: {memory_size:.2f} MB")
            logging.info(f"   🗜️  Compression ratio: {compression_ratio:.1f}x")

            # 3. Generate sample CSV for quick inspection
            if save_sample:
                sample_file = self.sample_path / f"{name}_sample.csv"
                df.head(sample_size).to_csv(sample_file, index=False)
                sample_size_kb = sample_file.stat().st_size / 1024
                logging.info(f"   📄 Sample saved: {sample_file.name} ({sample_size_kb:.1f} KB)")

            # 4. Automatic validation (skip for partitioned datasets)
            if self.auto_validate and not partition_cols:
                logging.info(f"   🔍 Validating {name}...")
                result = self.validator.validate_file(str(file_path), name)

                if not result['passed']:
                    logging.error(f"   ❌ Validation failed for {name}")
                    for error in result['errors']:
                        logging.error(f"      {error}")
                    return False
                else:
                    logging.info(f"   ✅ Validation passed")

                    # Log warnings if present
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
        """Parquet 파일 또는 파티션된 데이터셋을 DataFrame으로 로드합니다.

        선택적 컬럼 선택 및 행 필터링과 함께 Parquet 데이터를 효율적으로 로드합니다.
        단일 파일 및 파티션된 데이터셋을 모두 지원합니다. 빠른 I/O 작업을 위해
        PyArrow를 사용합니다.

        Args:
            name (str): 테이블 이름 (.parquet 확장자 제외).
            columns (Optional[List[str]], optional): 로드할 컬럼 리스트.
                None인 경우, 모든 컬럼을 로드합니다. 컬럼 필터링은 메모리 사용량을
                줄입니다. 기본값은 None.
            filters (Optional[List], optional): 행 필터링을 위한 PyArrow 필터 표현식.
                형식: [('column', 'operator', value)].
                예: [('year', '=', 2020), ('price', '>', 100)].
                기본값은 None.

        Returns:
            pd.DataFrame: 지정된 컬럼과 필터가 적용된 로드된 DataFrame.

        Raises:
            FileNotFoundError: 지정된 Parquet 파일 또는 파티션이 존재하지 않는 경우.

        사용 예시:
            다양한 필터링 옵션으로 로드::

                # 모든 데이터 로드
                df = storage.load_parquet('stock_prices')

                # 특정 컬럼만 로드
                df = storage.load_parquet(
                    'stock_prices',
                    columns=['symbol', 'close', 'volume']
                )

                # 행 필터링과 함께 로드
                df = storage.load_parquet(
                    'stock_prices',
                    filters=[('date', '>=', '2020-01-01')]
                )

                # 컬럼 선택과 필터링 결합
                df = storage.load_parquet(
                    'stock_prices',
                    columns=['symbol', 'close'],
                    filters=[('symbol', 'in', ['AAPL', 'GOOGL'])]
                )

        Note:
            - 컬럼 필터링은 읽기 시점에 발생하여 I/O를 줄입니다
            - PyArrow 필터는 다음을 지원합니다: =, !=, <, <=, >, >=, in, not in
            - 파티션된 데이터셋의 경우, 필터는 파티션 컬럼을 사용할 수 있습니다
        """
        file_path = self.parquet_path / f"{name}.parquet"

        if not file_path.exists():
            # Check if it's a partitioned directory
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

    def get_info(self, name: str) -> Dict[str, Any]:
        """데이터를 로드하지 않고 Parquet 파일 메타데이터를 가져옵니다.

        실제 데이터를 읽지 않고 스키마, 행 수, 파일 크기를 포함한 파일 메타데이터를
        검색합니다. 이것은 전체 데이터셋을 로드하는 것보다 훨씬 빠릅니다.
        PyArrow의 메타데이터 읽기 기능을 사용합니다.

        Args:
            name (str): 테이블 이름 (.parquet 확장자 제외).

        Returns:
            Dict[str, Any]: 다음 키를 포함하는 메타데이터 딕셔너리:
                - name (str): 테이블 이름
                - rows (int): 행 수
                - row_groups (int): 행 그룹 수
                - columns (int): 컬럼 수
                - column_names (List[str]): 컬럼 이름 리스트
                - dtypes (Dict[str, str]): 컬럼 이름을 데이터 타입에 매핑
                - size_mb (float): 파일 크기 (MB)
                - created (Optional[str]): 사용 가능한 경우 생성 메타데이터

                파일이 존재하지 않으면 {'error': 'File not found'}를 반환합니다.

        사용 예시:
            데이터를 로드하지 않고 메타데이터 쿼리::

                # 기본 정보 가져오기
                info = storage.get_info('stock_prices')
                print(f"Rows: {info['rows']:,}")
                print(f"Columns: {info['column_names']}")
                print(f"Size: {info['size_mb']:.2f} MB")

                # 로드하기 전에 스키마 확인
                info = storage.get_info('financial_data')
                if 'revenue' in info['column_names']:
                    df = storage.load_parquet('financial_data')

                # 모든 테이블 크기 가져오기
                for table in storage.list_tables():
                    info = storage.get_info(table)
                    print(f"{table}: {info['size_mb']:.2f} MB")

        Note:
            - 파일 메타데이터만 읽고, 실제 데이터는 읽지 않습니다
            - 전체 DataFrame을 로드하는 것보다 훨씬 빠릅니다
            - 파티션된 데이터셋(디렉토리 구조)에서는 작동하지 않습니다
        """
        file_path = self.parquet_path / f"{name}.parquet"

        if not file_path.exists():
            return {'error': 'File not found'}

        # Read metadata only using PyArrow (fast operation)
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
        """저장된 모든 테이블 이름을 나열합니다.

        단일 파일 테이블과 파티션된 데이터셋(Parquet 파일을 포함하는 디렉토리)을
        모두 위해 Parquet 디렉토리를 스캔합니다. 파일 확장자 없이 정렬된
        테이블 이름 리스트를 반환합니다.

        Returns:
            List[str]: 테이블 이름의 정렬된 리스트 (.parquet 확장자 제외).

        사용 예시:
            모든 테이블 나열 및 반복::

                # 모든 테이블 이름 가져오기
                tables = storage.list_tables()
                print(f"Found {len(tables)} tables: {tables}")

                # 모든 테이블 처리
                for table_name in storage.list_tables():
                    df = storage.load_parquet(table_name)
                    print(f"{table_name}: {len(df)} rows")

                # 테이블 존재 확인
                if 'stock_prices' in storage.list_tables():
                    df = storage.load_parquet('stock_prices')
        """
        tables = []

        # Find single Parquet files
        for file_path in self.parquet_path.glob("*.parquet"):
            tables.append(file_path.stem)

        # Find partitioned directories
        for dir_path in self.parquet_path.iterdir():
            if dir_path.is_dir() and list(dir_path.glob("*.parquet")):
                tables.append(dir_path.name)

        return sorted(tables)

    def delete_table(self, name: str) -> bool:
        """테이블과 관련 파일을 삭제합니다.

        Parquet 파일 또는 파티션된 디렉토리와 관련 샘플 CSV 파일을 제거합니다.
        단일 파일 및 파티션된 데이터셋을 모두 처리합니다.

        Args:
            name (str): 삭제할 테이블 이름 (.parquet 확장자 제외).

        Returns:
            bool: 삭제가 성공하면 True, 테이블을 찾지 못하거나 오류가 발생하면 False.

        사용 예시:
            확인과 함께 테이블 삭제::

                # 단일 테이블 삭제
                if storage.delete_table('old_data'):
                    print("Table deleted successfully")

                # 여러 테이블 삭제
                tables_to_delete = ['temp1', 'temp2', 'test_data']
                for table in tables_to_delete:
                    storage.delete_table(table)

                # 존재 확인과 함께 삭제
                if 'deprecated_table' in storage.list_tables():
                    storage.delete_table('deprecated_table')

        Note:
            - 관련 샘플 CSV 파일도 삭제합니다
            - 파티션된 데이터셋의 경우, 전체 디렉토리 트리를 제거합니다
            - 테이블을 찾지 못한 경우 경고를 로깅합니다
        """
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

            # Delete associated sample file
            sample_file = self.sample_path / f"{name}_sample.csv"
            if sample_file.exists():
                sample_file.unlink()

            return True

        except Exception as e:
            logging.error(f"Failed to delete {name}: {e}")
            return False

    def validate_all_tables(self) -> Dict[str, Dict[str, Any]]:
        """저장된 모든 Parquet 테이블을 검증합니다.

        저장소 디렉토리의 모든 테이블에 대해 포괄적인 검증을 실행합니다.
        데이터 품질, 스키마 일관성, 필수 컬럼, 데이터 타입을 검사합니다.
        상세 검증 규칙을 위해 DataValidator를 사용합니다.

        Returns:
            Dict[str, Dict[str, Any]]: 테이블 이름을 검증 결과에 매핑하는
                중첩 딕셔너리. 각 결과는 다음을 포함합니다:
                - passed (bool): 전체 검증 상태
                - errors (List[str]): 검증 오류 리스트
                - warnings (List[str]): 경고 리스트
                - info (Dict): 메타데이터 및 통계

        사용 예시:
            모든 테이블 검증 및 결과 확인::

                # 모든 테이블에 대해 검증 실행
                results = storage.validate_all_tables()

                # 어떤 테이블이 통과했는지 확인
                for table_name, result in results.items():
                    if result['passed']:
                        print(f"✅ {table_name}: OK")
                    else:
                        print(f"❌ {table_name}: FAILED")
                        for error in result['errors']:
                            print(f"   - {error}")

                # 검증 문제 수 계산
                failed = sum(1 for r in results.values() if not r['passed'])
                print(f"{failed} tables failed validation")

        See Also:
            generate_validation_report: 검증 결과에서 텍스트 보고서 생성.
        """
        return self.validator.validate_all(str(self.parquet_path))

    def generate_validation_report(self, output_file: str = "validation_report.txt") -> str:
        """검증 결과의 텍스트 보고서를 생성합니다.

        모든 테이블을 검증하고 상세한 보고서를 텍스트 파일에 작성합니다.
        보고서는 각 테이블에 대한 검증 상태, 오류, 경고, 통계를 사람이 읽기
        쉬운 형식으로 포함합니다.

        Args:
            output_file (str, optional): 출력 보고서 파일 경로. 절대 또는
                상대 경로일 수 있습니다. 기본값은 "validation_report.txt".

        Returns:
            str: 생성된 보고서 파일 경로.

        사용 예시:
            검증 보고서 생성 및 검토::

                # 기본 이름으로 보고서 생성
                report_path = storage.generate_validation_report()
                print(f"Report saved to: {report_path}")

                # 커스텀 이름으로 생성
                report_path = storage.generate_validation_report(
                    output_file="/reports/parquet_validation_2024.txt"
                )

                # 보고서 읽기 및 출력
                with open(report_path, 'r') as f:
                    print(f.read())

        Note:
            - 보고서는 모든 테이블에 대한 요약 통계를 포함합니다
            - 모든 오류와 경고를 상세히 나열합니다
            - 출력 파일이 존재하면 덮어씌워집니다
        """
        results = self.validate_all_tables()
        return self.validator.generate_report(results, output_file)

    def get_statistics(self) -> Dict[str, Any]:
        """전체 저장소에 대한 포괄적인 통계를 가져옵니다.

        총 크기, 행 수, 테이블별 통계를 포함하여 저장된 모든 테이블에 대한
        정보를 집계합니다. 저장소 사용량 및 데이터 볼륨 모니터링에 유용합니다.

        Returns:
            Dict[str, Any]: 다음을 포함하는 딕셔너리:
                - total_tables (int): 저장된 테이블 수
                - total_size_mb (float): 총 저장소 크기 (MB)
                - total_rows (int): 모든 테이블의 총 행 수
                - tables (List[Dict]): 다음 키를 가진 테이블별 통계:
                    - name (str): 테이블 이름
                    - rows (int): 행 수
                    - size_mb (float): 파일 크기 (MB)

        사용 예시:
            저장소 사용량 모니터링::

                # 전체 통계 가져오기
                stats = storage.get_statistics()
                print(f"Total tables: {stats['total_tables']}")
                print(f"Total size: {stats['total_size_mb']:.2f} MB")
                print(f"Total rows: {stats['total_rows']:,}")

                # 가장 큰 테이블 찾기
                tables = sorted(
                    stats['tables'],
                    key=lambda x: x['size_mb'],
                    reverse=True
                )
                print("\nTop 5 largest tables:")
                for table in tables[:5]:
                    print(f"  {table['name']}: {table['size_mb']:.2f} MB")

                # 평균 테이블 크기 계산
                avg_size = stats['total_size_mb'] / stats['total_tables']
                print(f"Average table size: {avg_size:.2f} MB")

        Note:
            - 많은 대용량 테이블의 경우 통계 계산이 느릴 수 있습니다
            - 읽을 수 없는 테이블은 건너뜁니다 (경고 로깅)
            - 크기 계산에 샘플 CSV 파일은 포함되지 않습니다
        """
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
