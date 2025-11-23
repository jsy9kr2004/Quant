# ROOT_PATH 폴더 구조 변경 가이드

프로젝트 구조 개선에 따라 ROOT_PATH (외장하드) 폴더도 재구성이 필요합니다.

## ⚠️ 중요사항

- **FMP API 재호출 불필요**: 기존 데이터를 이동만 하면 됩니다
- **데이터 재생성 불필요**: ML 학습 데이터도 이동만 하면 됩니다
- **작업 시간**: 데이터 크기에 따라 다르지만, 일반적으로 10-30분 소요

---

## 📋 이동 작업 순서

### 1. 백업 (선택사항이지만 강력 권장)

```bash
# 중요한 데이터는 백업 후 진행
cp -r ${ROOT_PATH} ${ROOT_PATH}_backup
```

### 2. 새 폴더 구조 생성

```bash
# ROOT_PATH로 이동
cd ${ROOT_PATH}

# 새 폴더 구조 생성
mkdir -p fmp_raw
mkdir -p processed/views
mkdir -p processed/ml_data/per_year
mkdir -p processed/intermediate/DATE_TABLE
mkdir -p processed/intermediate/parquet
mkdir -p models/production
mkdir -p models/walkforward
mkdir -p analysis/nan_analysis
mkdir -p analysis/nan_removal
mkdir -p debug
mkdir -p cache/samples
```

### 3. FMP 원본 데이터 이동

**현재 ROOT_PATH 바로 하위에 있는 모든 FMP 카테고리 폴더들을 fmp_raw/ 하위로 이동:**

```bash
cd ${ROOT_PATH}

# 모든 FMP 카테고리 폴더 이동
mv income_statement/ fmp_raw/
mv balance_sheet_statement/ fmp_raw/
mv cash_flow_statement/ fmp_raw/
mv key_metrics/ fmp_raw/
mv financial_ratios/ fmp_raw/
mv stock_list/ fmp_raw/
mv delisted_companies/ fmp_raw/
mv historical_daily_discounted_cash_flow/ fmp_raw/

# 기타 FMP 카테고리가 있다면 모두 이동
# (ROOT_PATH 하위에 있던 모든 데이터 폴더들)
mv {카테고리명}/ fmp_raw/
```

**또는 한 번에 이동 (파일은 제외, 폴더만):**

```bash
cd ${ROOT_PATH}
for dir in */; do
    # 특수 폴더 제외
    if [[ "$dir" != "fmp_raw/" ]] && [[ "$dir" != "processed/" ]] && \
       [[ "$dir" != "models/" ]] && [[ "$dir" != "analysis/" ]] && \
       [[ "$dir" != "debug/" ]] && [[ "$dir" != "cache/" ]] && \
       [[ "$dir" != "VIEW/" ]] && [[ "$dir" != "ml_per_year/" ]] && \
       [[ "$dir" != "MODELS_WALKFORWARD/" ]] && [[ "$dir" != "MODELS/" ]] && \
       [[ "$dir" != "NAN_ANALYSIS/" ]] && [[ "$dir" != "NAN_REMOVAL_DETAILS/" ]] && \
       [[ "$dir" != "DATE_TABLE/" ]] && [[ "$dir" != "parquet/" ]] && \
       [[ "$dir" != "samples/" ]]; then
        mv "$dir" fmp_raw/
        echo "Moved $dir to fmp_raw/"
    fi
done
```

### 4. 가공 데이터 이동

```bash
cd ${ROOT_PATH}

# VIEW 폴더 이동
mv VIEW/ processed/views/

# ML 학습 데이터 이동
mv ml_per_year/ processed/ml_data/per_year/

# 중간 데이터 이동
mv DATE_TABLE/ processed/intermediate/DATE_TABLE/ 2>/dev/null || true
mv parquet/ processed/intermediate/parquet/ 2>/dev/null || true
mv samples/ cache/samples/ 2>/dev/null || true
```

### 5. 모델 파일 이동

```bash
cd ${ROOT_PATH}

# Walk-Forward 백테스트 모델
mv MODELS_WALKFORWARD/ models/walkforward/

# 프로덕션 모델 (있다면)
mv MODELS/ models/production/ 2>/dev/null || true
```

### 6. 분석 결과 이동

```bash
cd ${ROOT_PATH}

# NaN 분석 결과
mv NAN_ANALYSIS/ analysis/nan_analysis/ 2>/dev/null || true

# NaN 제거 상세
mv NAN_REMOVAL_DETAILS/ analysis/nan_removal/ 2>/dev/null || true
```

### 7. 디버그 파일 이동

```bash
cd ${ROOT_PATH}

# fs_metric_wdate 파일들
mv fs_metric_wdate_*.parquet debug/ 2>/dev/null || true
mv fs_metric_wdate_*.csv debug/ 2>/dev/null || true
```

---

## 🔍 이동 후 확인

### 최종 폴더 구조 확인:

```bash
cd ${ROOT_PATH}
tree -L 2 -d
```

**예상 결과:**

```
.
├── fmp_raw
│   ├── income_statement
│   ├── balance_sheet_statement
│   ├── cash_flow_statement
│   ├── key_metrics
│   ├── financial_ratios
│   ├── stock_list
│   └── delisted_companies
│
├── processed
│   ├── views
│   ├── ml_data
│   └── intermediate
│
├── models
│   ├── production
│   └── walkforward
│
├── analysis
│   ├── nan_analysis
│   └── nan_removal
│
├── debug
└── cache
    └── samples
```

### 파일 개수 확인:

```bash
# FMP 원본 데이터 확인
find fmp_raw -type f -name "*.parquet" | wc -l

# VIEW 데이터 확인
find processed/views -type f -name "*.parquet" | wc -l

# ML 학습 데이터 확인
find processed/ml_data -type f -name "*.parquet" | wc -l

# 모델 파일 확인
find models -type f -name "*.pkl" | wc -l
```

---

## ✅ 정리

### 불필요한 빈 폴더 삭제:

```bash
cd ${ROOT_PATH}

# 빈 폴더만 삭제 (안전)
find . -type d -empty -delete
```

---

## 🚀 테스트

이동 완료 후, 코드가 제대로 동작하는지 확인:

```bash
cd /home/user/Quant/Quant-refactoring

# 간단한 테스트
python -c "from src.config.context_loader import load_config, MainContext; config = load_config('config/conf.yaml'); ctx = MainContext(config); print('✅ Config loaded successfully')"
```

**에러가 나지 않으면 성공!**

---

## ⚠️ 문제 발생 시

### 롤백 방법:

```bash
# 백업으로 복원
rm -rf ${ROOT_PATH}
mv ${ROOT_PATH}_backup ${ROOT_PATH}
```

### 일반적인 문제:

1. **경로를 못 찾는 오류**: 폴더 이동이 제대로 되지 않았을 가능성
   - `ls -la ${ROOT_PATH}` 로 구조 확인

2. **권한 오류**: 폴더 권한 문제
   - `chmod -R 755 ${ROOT_PATH}`

3. **파일이 없다는 오류**: 이동 명령어 실행 시 이미 없는 폴더였을 수 있음 (정상)

---

## 📝 참고사항

- **원본 데이터는 `fmp_raw/` 에만 존재**: 재수집 불필요
- **가공 데이터는 `processed/` 에만 존재**: 재생성 불필요
- **코드는 이미 수정됨**: 이동만 하면 바로 동작
- **외장하드 이동성 유지**: ROOT_PATH만 변경하면 어디서든 사용 가능

---

## 완료!

이제 깔끔하게 정리된 구조로 프로젝트를 사용할 수 있습니다! 🎉
