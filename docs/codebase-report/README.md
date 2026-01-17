# Codebase Analysis Report

> **작성자**: Claude AI
> **최종 업데이트**: 2026-01-17

이 폴더는 Quant 프로젝트의 **핵심 분석 레포트**를 담고 있습니다.

---

## 레포트 목록

| # | 파일 | 주제 | 설명 |
|---|------|------|------|
| 0 | [00_overview.md](./00_overview.md) | 전체 개요 | 프로젝트 목적, 철학, 종합 평가 |
| 1 | [01_architecture.md](./01_architecture.md) | 아키텍처 | 시스템 구조, 모듈 관계, 데이터 흐름 |
| 2 | [02_data_pipeline.md](./02_data_pipeline.md) | 데이터 파이프라인 | FMP API, 전처리, 저장 구조 |
| 3 | [03_ml_strategy.md](./03_ml_strategy.md) | ML 전략 | 2-Stage 모델, Classifier/Regressor |
| 4 | [04_backtesting.md](./04_backtesting.md) | 백테스팅 | Walk-Forward, 거래 비용, 벤치마크 |
| 5 | [05_code_quality.md](./05_code_quality.md) | 코드 품질 | 가독성, 유지보수성, 테스트 |
| 6 | [06_quant_perspective.md](./06_quant_perspective.md) | 퀀트 관점 | 시장 효율성, 리스크, 알파 |
| 7 | [07_recommendations.md](./07_recommendations.md) | 개선 권고 | 우선순위별 개선사항 |
| 8 | [08_recent_changes.md](./08_recent_changes.md) | 최신 변경사항 | 2026년 1월 업데이트 내역 |

---

## 읽는 순서

```
00_overview (전체 파악)
    ↓
01_architecture (구조 이해)
    ↓
02_data_pipeline → 03_ml_strategy → 04_backtesting (핵심 파이프라인)
    ↓
05_code_quality → 06_quant_perspective (평가)
    ↓
07_recommendations (개선 방향)
    ↓
08_recent_changes (최신 상태)
```

---

## 용도

1. **프로젝트 이해**: 새로운 기여자가 전체 구조를 빠르게 파악
2. **의사결정 참고**: 개선 우선순위, 기술적 트레이드오프 검토
3. **진행 상황 추적**: 08_recent_changes.md에서 최신 변경사항 확인
4. **TODO 관리**: 07_recommendations.md의 개선사항을 기준으로 작업 계획

---

## 업데이트 규칙

- 주요 기능 추가/변경 시 관련 문서 업데이트
- 08_recent_changes.md에 변경 내역 기록
- 헤더의 "최종 업데이트" 날짜 갱신
