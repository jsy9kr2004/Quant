"""
무한대 값을 포함하는 컬럼을 진단하는 스크립트

로그 파일을 분석하여 어떤 컬럼들이 무한대 값을 가지고 있는지 파악합니다.
"""

import re
from collections import Counter

def analyze_log():
    """로그 파일에서 무한대 값 관련 정보 추출"""

    print("=" * 80)
    print("🔍 로그 파일 분석: 무한대 값 컬럼 찾기")
    print("=" * 80)

    with open('Quant-refactoring/log_to_git_5', 'r') as f:
        content = f.read()

    # 1. 필터 통계 분석 (not_in_data, nan>=30%, has_infinite)
    print("\n" + "="*80)
    print("📊 필터 통계 분석")
    print("="*80)

    filter_patterns = re.findall(
        r'\[(\d{6})\].*?Filter breakdown:.*?not_in_data=(\d+), nan>=30%=(\d+), has_infinite=(\d+)',
        content,
        re.DOTALL
    )

    if filter_patterns:
        print(f"\n발견된 분기: {len(filter_patterns)}개\n")
        for year_period, not_in, nan_pct, inf in filter_patterns:
            year = int(year_period[:4])
            quarter = int(year_period[4:])
            status = "❌ FAILED" if int(inf) > 0 else "✅ OK"
            print(f"{status} [{year} Q{quarter}] 누락={not_in}, NaN≥30%={nan_pct}, 무한대={inf}")

    # 2. 누락된 컬럼 찾기
    print("\n" + "="*80)
    print("🔍 누락된 컬럼 (not_in_data)")
    print("="*80)

    missing_cols = re.findall(r"Sample missing columns.*?: \['(.+?)'\]", content)
    if missing_cols:
        all_missing = set()
        for cols_str in missing_cols:
            cols = [c.strip().strip("'") for c in cols_str.split(',')]
            all_missing.update(cols)
        print(f"\n누락된 컬럼 (총 {len(all_missing)}개):")
        for col in sorted(all_missing):
            print(f"  - {col}")
    else:
        print("\n누락된 컬럼 정보를 찾을 수 없습니다.")

    # 3. 무한대 값 컬럼 패턴 찾기
    print("\n" + "="*80)
    print("∞ 무한대 값을 포함하는 컬럼")
    print("="*80)

    # "Contains infinite" 메시지 전후 텍스트 분석
    infinite_sections = re.findall(
        r'\[(\d{6})\].*?Starting time series.*?NaN analysis.*?((?:.*?\n){0,30}).*?Contains infinite: (\d+)',
        content,
        re.DOTALL
    )

    if infinite_sections:
        print(f"\n무한대 값이 발견된 분기: {len(infinite_sections)}개")

        # 첫 번째 케이스 상세 출력
        if infinite_sections:
            year_period, context, inf_count = infinite_sections[0]
            year = int(year_period[:4])
            quarter = int(year_period[4:])
            print(f"\n첫 번째 케이스 [{year} Q{quarter}]:")
            print(f"무한대 컬럼 개수: {inf_count}")
            print(f"\n컨텍스트 (일부):")
            print(context[:500])

    # 4. 경고 메시지 패턴 찾기
    print("\n" + "="*80)
    print("⚠️  경고 메시지")
    print("="*80)

    warnings = re.findall(r'(WARNING:.*?(?:infinite|inf|division).*?)(?:\n|$)', content, re.IGNORECASE)
    if warnings:
        print(f"\n발견된 경고: {len(warnings)}개 (최근 5개):")
        for i, warning in enumerate(warnings[-5:], 1):
            print(f"{i}. {warning[:150]}")

    # 5. 요약 및 권장사항
    print("\n" + "="*80)
    print("💡 분석 결과 및 권장사항")
    print("="*80)

    if filter_patterns:
        # 2023 vs 2024+ 비교
        y2023_cases = [(yp, inf) for yp, _, _, inf in filter_patterns if yp.startswith('2023')]
        y2024_cases = [(yp, inf) for yp, _, _, inf in filter_patterns if yp.startswith('2024')]

        print(f"\n2023년 분기 수: {len(y2023_cases)}")
        print(f"2024년+ 분기 수: {len(y2024_cases)}")

        if y2023_cases:
            avg_2023 = sum(int(inf) for _, inf in y2023_cases) / len(y2023_cases)
            print(f"2023년 평균 무한대 컬럼: {avg_2023:.1f}")

        if y2024_cases:
            avg_2024 = sum(int(inf) for _, inf in y2024_cases) / len(y2024_cases)
            print(f"2024년+ 평균 무한대 컬럼: {avg_2024:.1f}")

    print("\n다음 조치:")
    print("1. VIEW 원본 데이터의 무한대 값을 NaN으로 변환")
    print("2. metrics_table과 fs_table 로드 직후 infinite 체크 추가")
    print("3. 각 비율 계산 전에 분모가 0이 아닌지 검증")
    print("4. replace([np.inf, -np.inf], np.nan) 적용")

    print("\n" + "="*80)


if __name__ == "__main__":
    try:
        analyze_log()
    except FileNotFoundError:
        print("❌ 로그 파일을 찾을 수 없습니다: Quant-refactoring/log_to_git_5")
        print("현재 디렉토리에서 실행해야 합니다.")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
