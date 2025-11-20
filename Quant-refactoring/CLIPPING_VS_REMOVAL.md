# Clipping vs Removal: Handling Too Large Values

## ✅ You Were Right!

Infinite와 Too Large는 **완전히 다른 문제**입니다!

---

## 🎯 핵심 차이

### Infinite (제거해야 함)
```python
# 수학적 오류 - 의미 없는 값
price_change = 100 / 0  # inf
pe_ratio = price / 0  # inf
log_negative = log(-10)  # -inf

→ 이런 값들은 제거 ✅
```

### Too Large (변환해서 사용해야 함)
```python
# 실제 극단값 - 의미 있는 정보!
market_cap = 10_000_000_000_000  # 삼성전자 시총 10조원
revenue_growth = 5000  # 스타트업 매출 5000% 증가

→ 이런 값들을 지우면 정보 손실! ❌
→ Clipping으로 사용 가능하게 변환 ✅
```

---

## 🛠️ 해결 방법 비교

### ❌ 방법 1: 제거 (Wrong - My mistake)
```python
# 극단값 있는 행 제거
mask = (abs(data) > threshold).any(axis=1)
data = data[~mask]

문제점:
- 정보 손실
- 데이터 줄어듦
- 극단값(이상치)도 중요한 정보!
```

### ✅ 방법 2: Clipping (Better - Quick fix)
```python
# 값을 범위 내로 제한
data_clipped = np.clip(data, -threshold, threshold)

장점:
- 정보 보존 (극단값의 방향은 유지)
- 데이터 개수 유지
- XGBoost가 처리 가능

단점:
- 여전히 정보 왜곡 (10조 → 1e10으로 잘림)
```

### ✅✅ 방법 3: 근본 해결 (Best - Long term)

**make_mldata.py에서 처리:**

#### Option A: Winsorization
```python
# 극단값을 percentile로 대체
from scipy.stats import mstats
data_winsorized = mstats.winsorize(data, limits=[0.01, 0.01])

# 예: 1% 극단값을 99 percentile 값으로 대체
# 10조 → 1000억 (99 percentile)
```

#### Option B: Log Transformation
```python
# 큰 값을 log 스케일로 변환
data_log = np.sign(data) * np.log1p(np.abs(data))

# 10조 → log(10조) = ~30
# 범위: 1~1e13 → 0~30
```

#### Option C: RobustScaler + Clip
```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
data_scaled = scaler.fit_transform(data)

# 스케일링 후 극단값 제한
data_scaled = np.clip(data_scaled, -5, 5)  # ±5 std
```

---

## 🔬 Why This Matters

### 예시: 삼성전자 vs 소형주

| 특성 | 소형주 | 삼성전자 |
|------|--------|----------|
| 시가총액 | 1000억 (1e11) | 400조 (4e14) |
| 거래량 | 10만주 | 1000만주 |

**제거하면:**
- 삼성전자 데이터 삭제 ❌
- 대형주 정보 손실 ❌

**Clipping하면:**
- 1e11 vs 1e10 (둘 다 cap) → 구분 못함
- 여전히 정보 손실 있지만 데이터는 보존

**Log transform하면:**
- log(1e11) = 25 vs log(4e14) = 33
- 상대적 크기 유지 ✅
- 범위도 적절 ✅

---

## 📊 Implementation Plan

### Phase 1: Quick Fix (Now - regressor.py)
```python
# Clipping으로 당장 에러 해결
data = np.clip(data, -1e10, 1e10)
```
✅ XGBoost 돌아감
⚠️ 정보 약간 손실

### Phase 2: Better Scaling (Next - make_mldata.py)
```python
# 스케일링 후 Winsorization
scaled_df = scaler.fit_transform(filtered_df)
scaled_df = mstats.winsorize(scaled_df, limits=[0.01, 0.01])
```
✅ 극단값 보존
✅ 범위 제어

### Phase 3: Feature Engineering (Future)
```python
# 큰 값은 log transform
for col in large_value_cols:
    df[col] = np.log1p(df[col])
```
✅ 근본 해결
✅ 모델 성능 향상 가능

---

## 🎯 Current Change (Phase 1)

### Before (Wrong):
```python
# 큰 값 있는 행 제거
if large_values_found:
    data = data[~large_mask]  # 정보 손실!
```

### After (Better):
```python
# 큰 값을 clipping
if large_values_found:
    data = np.clip(data, -1e10, 1e10)  # 정보 보존!
```

---

## 🚀 Next Steps

1. **현재 수정으로 테스트**
   - Clipping이 XGBoost 에러 해결하는지 확인

2. **로그 분석**
   - 얼마나 많은 값이 clipped 되는지
   - 어떤 컬럼에 극단값이 많은지

3. **make_mldata.py 개선** (다음 PR)
   - Winsorization 추가
   - 또는 log transformation
   - 스케일링 후 clip

4. **모델 성능 확인**
   - Clipping이 성능에 영향 주는지
   - 필요하면 더 나은 방법 적용

---

## 📝 Key Takeaway

**Infinite ≠ Too Large**

- **Infinite**: 에러 → 제거
- **Too Large**: 실제값 → 변환

**정보를 보존하면서 XGBoost가 처리 가능하게 만드는게 목표!**
