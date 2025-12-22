# Troubleshooting Guide

## Python 3.13 호환성 문제

**문제**: `ModuleNotFoundError: No module named 'ray'`

**원인**: Python 3.13은 2024년 10월에 출시된 최신 버전으로, 일부 패키지가 아직 공식 지원하지 않을 수 있습니다.

### 해결 방법

#### 옵션 1: Ray 단독 설치 시도 (권장)
```bash
pip install ray --upgrade
```

실패 시:
```bash
# Pre-release 버전 설치
pip install ray --pre

# 또는 nightly build
pip install -U "ray[default] @ https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-3.0.0.dev0-cp313-cp313-win_amd64.whl"
```

#### 옵션 2: Python 3.11 또는 3.12 사용 (가장 안정적)
```bash
# Python 3.11 또는 3.12 가상환경 생성
python3.11 -m venv venv
# 또는
python3.12 -m venv venv

# 가상환경 활성화
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 패키지 재설치
pip install -r requirements.txt
```

#### 옵션 3: Ray 없이 실행 (임시 방편)
Ray는 데이터 수집 시 병렬 처리에 사용됩니다. 급한 경우 코드를 수정하여 Ray 없이 실행 가능:

1. `src/data_collector/fmp_fetch_worker.py` 수정:
   ```python
   # import ray 주석 처리
   try:
       import ray
       HAS_RAY = True
   except ImportError:
       HAS_RAY = False
       print("⚠️  Ray not available, using sequential processing")
   ```

2. FMP 클래스에서 Ray 사용 여부 체크 추가

### 권장 환경

**Production 환경**:
- Python: 3.11.x 또는 3.12.x (안정성)
- OS: Linux (Ubuntu 20.04+) 또는 Windows 10/11
- RAM: 16GB 이상
- CUDA: 11.8 또는 12.x (GPU 사용 시)

**개발 환경**:
- Python: 3.11.9 (가장 안정적)
- Virtual Environment 사용 필수

### 설치 확인

```bash
# 설치된 패키지 확인
pip list | findstr ray  # Windows
# pip list | grep ray  # Linux/Mac

# Ray 작동 테스트
python -c "import ray; ray.init(); print('Ray works!'); ray.shutdown()"
```

### 추가 문제 발생 시

1. **scipy 버전 충돌**:
   ```bash
   pip install "scipy>=1.11.0,<1.13.0"
   ```

2. **CuPy 설치 실패** (GPU 없는 경우):
   - requirements.txt에서 `cupy-cuda11x` 줄 주석 처리
   - 또는 `pip install --no-deps -r requirements.txt`

3. **Microsoft Visual C++ 필요** (Windows):
   - [Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe) 설치

4. **권한 오류**:
   ```bash
   pip install --user -r requirements.txt
   ```

## 기타 일반적인 문제

### 1. YAML 파일 없음
```
FileNotFoundError: config/conf.yaml
```

**해결**:
```bash
cp config/conf.yaml.template config/conf.yaml
# conf.yaml 파일을 열어 API_KEY 등 설정 입력
```

### 2. 데이터 파일 없음
```
FileNotFoundError: processed/ml_data/...
```

**해결**: 데이터 수집부터 순차적으로 실행
```bash
# 1. 데이터 수집
python main.py collect

# 2. Feature 생성
python main.py make-mldata

# 3. 학습
python main.py train

# 4. 백테스트
python main.py backtest
```

### 3. 메모리 부족
```
MemoryError or OOM
```

**해결**:
- 데이터 범위 축소 (TRAIN_START_YEAR 증가)
- Batch size 감소
- Swap 메모리 증가 (Linux)

### 4. GPU 감지 안 됨
```
CUDA not available
```

**해결**:
```bash
# CUDA 버전 확인
nvidia-smi

# 올바른 CuPy 버전 설치
pip uninstall cupy-cuda11x cupy-cuda12x
pip install cupy-cuda11x  # CUDA 11.x
# 또는
pip install cupy-cuda12x  # CUDA 12.x
```

---

**Last Updated**: 2025-12-21
