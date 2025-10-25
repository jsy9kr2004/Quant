# 통합 로깅 시스템 가이드

## 개요

멀티프로세싱 환경에서 안전하게 작동하는 통합 로깅 시스템입니다.

### 주요 기능

- ✅ **멀티프로세싱 안전**: QueueHandler를 사용한 프로세스 간 안전한 로깅
- ✅ **컬러 출력**: 콘솔에서 로그 레벨별 색상 구분
- ✅ **자동 로테이션**: 로그 파일 크기 기반 자동 백업 (기본: 10MB, 5개 백업)
- ✅ **구조화된 로깅**: Context 정보 자동 추가
- ✅ **간단한 API**: 기존 logging 코드와 호환

## 기본 사용법

### 1. 단일 프로세스 애플리케이션

```python
from config.logger import get_logger

# 로거 가져오기
logger = get_logger(__name__)

# 로그 작성
logger.info("Processing started")
logger.warning("Low memory")
logger.error("Failed to connect", extra={'host': 'localhost', 'port': 5432})
```

### 2. 멀티프로세싱 환경

**메인 프로세스:**
```python
from config.logger import setup_logging, get_logger

# 메인 프로세스에서 한 번만 호출
setup_logging(
    log_level='INFO',
    log_file='app.log',
    console_output=True,
    use_colors=True
)

logger = get_logger('main')
logger.info("Application started")
```

**워커 프로세스 (Ray, multiprocessing 등):**
```python
from config.logger import setup_logger_for_multiprocessing, get_logger

# 각 워커 프로세스 시작 시 호출
setup_logger_for_multiprocessing()

logger = get_logger(f'worker-{worker_id}')
logger.info("Worker started")
```

## 설정 옵션

### setup_logging() 파라미터

```python
setup_logging(
    log_level='INFO',              # DEBUG, INFO, WARNING, ERROR, CRITICAL
    log_file='log.txt',            # 로그 파일 이름
    log_dir='.',                   # 로그 파일 디렉토리
    console_output=True,           # 콘솔 출력 활성화
    file_output=True,              # 파일 출력 활성화
    use_colors=True,               # 컬러 출력 (콘솔)
    max_bytes=10 * 1024 * 1024,   # 10MB (파일 로테이션 크기)
    backup_count=5,                # 백업 파일 개수
    format_string=None             # 커스텀 포맷 (기본값 사용 권장)
)
```

## 로그 레벨

로그 레벨은 심각도 순으로 다음과 같습니다:

| 레벨 | 사용 목적 | 색상 (콘솔) |
|------|-----------|------------|
| `DEBUG` | 상세한 디버깅 정보 | 시안 |
| `INFO` | 일반 정보 메시지 | 녹색 |
| `WARNING` | 경고 메시지 | 노랑 |
| `ERROR` | 에러 메시지 | 빨강 |
| `CRITICAL` | 치명적 에러 | 자주 |

## 로그 포맷

기본 로그 포맷:
```
[2025-01-15 10:30:45][INFO    ][MainProcess] Message here (module_name:123)
```

- 시간: ISO 형식 타임스탬프
- 레벨: 로그 레벨 (8자리 고정 폭)
- 프로세스: 프로세스 이름
- 메시지: 로그 메시지
- 위치: 파일명:라인번호

## 실전 예제

### 예제 1: 기본 사용

```python
from config.logger import get_logger

class DataProcessor:
    def __init__(self):
        self.logger = get_logger(__name__)

    def process(self, data):
        self.logger.info(f"Processing {len(data)} items")

        try:
            result = self._do_process(data)
            self.logger.info("✅ Processing completed")
            return result
        except Exception as e:
            self.logger.error(f"❌ Processing failed: {e}", exc_info=True)
            raise
```

### 예제 2: Context 정보 추가

```python
from config.logger import get_logger

logger = get_logger('trading')

# 추가 컨텍스트와 함께 로그
logger.info(
    "Trade executed",
    extra={
        'symbol': 'AAPL',
        'quantity': 100,
        'price': 150.50
    }
)
```

### 예제 3: Ray 워커에서 사용

```python
import ray
from config.logger import setup_logger_for_multiprocessing, get_logger

@ray.remote
def process_data(worker_id, data):
    # 워커 시작 시 로거 설정
    setup_logger_for_multiprocessing()
    logger = get_logger(f'worker-{worker_id}')

    logger.info(f"Processing {len(data)} items")
    # ... 처리 로직
    logger.info("Processing completed")

    return result
```

## 마이그레이션 가이드

### 기존 코드에서 마이그레이션

**Before (기존 logging):**
```python
import logging

logging.info("Message")
logging.error("Error occurred")
```

**After (새 통합 로거):**
```python
from config.logger import get_logger

logger = get_logger(__name__)

logger.info("Message")
logger.error("Error occurred")
```

### 기존 로거 어댑터 마이그레이션

**Before:**
```python
logger = logging.getLogger(name)
adapter = logging.LoggerAdapter(logger, extra={'name': name})
```

**After:**
```python
from config.logger import get_logger

logger = get_logger(name)  # 자동으로 컨텍스트 포함
```

## 성능 고려사항

1. **메인 프로세스**: 로그는 QueueListener를 통해 비동기로 처리되어 성능 영향 최소화
2. **워커 프로세스**: QueueHandler를 통해 빠르게 로그를 큐에 전달
3. **파일 I/O**: 메인 프로세스의 리스너가 일괄 처리

## 문제 해결

### 로그가 출력되지 않음

```python
# 로거가 초기화되었는지 확인
from config.logger import setup_logging

setup_logging(log_level='DEBUG')  # 명시적 초기화
```

### 멀티프로세싱 환경에서 로그 누락

```python
# 각 워커에서 반드시 호출
from config.logger import setup_logger_for_multiprocessing

def worker_init():
    setup_logger_for_multiprocessing()  # 필수!
```

### 로그 파일이 너무 큼

```python
# 로테이션 설정 조정
setup_logging(
    max_bytes=5 * 1024 * 1024,  # 5MB로 축소
    backup_count=3               # 백업 3개만 유지
)
```

## 참고

- Python logging 문서: https://docs.python.org/3/library/logging.html
- QueueHandler: https://docs.python.org/3/library/logging.handlers.html#queuehandler
- 로깅 Best Practices: https://docs.python.org/3/howto/logging.html

## 기술 지원

로깅 관련 문제가 발생하면 다음 정보와 함께 이슈를 등록해주세요:

1. 로그 레벨 설정
2. 멀티프로세싱 사용 여부
3. 에러 메시지 및 스택 트레이스
4. 재현 가능한 최소 코드 예제
