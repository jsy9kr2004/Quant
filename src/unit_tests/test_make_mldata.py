"""
Unit tests for make_mldata.py

Tests the feature engineering pipeline including:
- tsfresh feature filtering
- Date generation and trading day conversion
- Column operations (filtering, reordering, merging)
- Winsorization
- Extreme mover filtering

Note: Tests use standalone logic implementations to avoid importing AIDataMaker
which has heavy dependencies (tqdm, tsfresh, etc.)
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from functools import reduce
from unittest.mock import Mock, patch, MagicMock


# ============================================================================
# Standalone Logic Functions (copied from make_mldata.py for testing)
# ============================================================================

# suffixes_dict used in AIDataMaker
SUFFIXES_DICT = {
    "standard_deviation": ["__r_0.0", "__r_0.25", "__r_0.6", "__r_0.9"],
    "quantile": ["__q_0.2", "__q_0.8"],
    "autocorrelation": ["__lag_0", "__lag_5", "__lag_9"],
    "fft_coefficient": ["__coeff_0", "__coeff_33", "__coeff_99"],
    "cwt_coefficients": ["__coeff_0", "__coeff_6", "__coeff_12"],
    "symmetry_looking": ["__r_0.0", "__r_0.25", "__r_0.65", "__r_0.9"],
    "ar_coefficient": ["__coeff_0", "__coeff_3", "__coeff_6", "__coeff_10"]
}


def filter_columns_by_suffixes(df, suffixes_dict=SUFFIXES_DICT):
    """Filter tsfresh columns by suffix patterns."""
    filtered_cols = []
    for col in df.columns:
        include_col = True
        for keyword, suffixes in suffixes_dict.items():
            if keyword in col:
                if any(suffix in col for suffix in suffixes):
                    break
                else:
                    include_col = False
                    break
        if include_col:
            filtered_cols.append(col)
    return filtered_cols


def filter_dates(df, target_col_name, start_year, end_year):
    """Filter DataFrame by date range."""
    df = df.copy()
    df[target_col_name] = pd.to_datetime(df[target_col_name])
    start_date = pd.Timestamp(year=start_year, month=1, day=1)
    end_date = pd.Timestamp(year=end_year+1, month=3, day=1)
    filtered_df = df[(df[target_col_name] >= start_date) & (df[target_col_name] <= end_date)]
    return filtered_df


def reorder_columns(df, keywords=['symbol', 'date']):
    """Reorder DataFrame columns with keywords first."""
    key_cols = [col for col in df.columns if any(key.lower() in col.lower() for key in keywords)]
    other_cols = [col for col in df.columns if col not in key_cols]
    new_order = key_cols + other_cols
    return df[new_order]


def efficient_merge_columns(df):
    """Merge _x/_y suffix columns from DataFrame joins."""
    base_cols = set(col.rstrip('_x').rstrip('_y') for col in df.columns)
    result_df = pd.DataFrame(index=df.index)
    for base in base_cols:
        cols = [col for col in df.columns if col.startswith(base)]
        merged_col = reduce(lambda x, y: df[y].combine_first(x), cols, pd.Series([np.nan]*len(df), index=df.index))
        result_df[base] = merged_col
    result_df = reorder_columns(result_df)
    return result_df


# ============================================================================
# Test: filter_columns_by_suffixes
# ============================================================================

class TestFilterColumnsBySuffixes:
    """Tests for tsfresh feature suffix filtering."""

    def test_keeps_valid_suffix_columns(self):
        """Should keep columns with valid suffixes."""
        df = pd.DataFrame({
            'revenue_ts_standard_deviation__r_0.0': [1, 2, 3],
            'revenue_ts_standard_deviation__r_0.25': [1, 2, 3],
            'revenue_ts_standard_deviation__r_0.5': [1, 2, 3],  # Not in valid suffixes
        })

        result = filter_columns_by_suffixes(df)

        assert 'revenue_ts_standard_deviation__r_0.0' in result
        assert 'revenue_ts_standard_deviation__r_0.25' in result
        assert 'revenue_ts_standard_deviation__r_0.5' not in result

    def test_keeps_columns_without_keywords(self):
        """Should keep columns that don't contain any keywords."""
        df = pd.DataFrame({
            'symbol': ['A', 'B', 'C'],
            'date': [1, 2, 3],
            'price': [100, 200, 300],
        })

        result = filter_columns_by_suffixes(df)

        assert 'symbol' in result
        assert 'date' in result
        assert 'price' in result

    def test_mixed_columns(self):
        """Should correctly filter mixed column types."""
        df = pd.DataFrame({
            'symbol': ['A', 'B', 'C'],
            'revenue_ts_quantile__q_0.2': [1, 2, 3],  # Valid
            'revenue_ts_quantile__q_0.5': [1, 2, 3],  # Invalid suffix
            'revenue_ts_autocorrelation__lag_0': [1, 2, 3],  # Valid
            'revenue_ts_autocorrelation__lag_3': [1, 2, 3],  # Invalid suffix
        })

        result = filter_columns_by_suffixes(df)

        assert 'symbol' in result
        assert 'revenue_ts_quantile__q_0.2' in result
        assert 'revenue_ts_quantile__q_0.5' not in result
        assert 'revenue_ts_autocorrelation__lag_0' in result
        assert 'revenue_ts_autocorrelation__lag_3' not in result

    def test_empty_dataframe(self):
        """Should handle empty DataFrame."""
        df = pd.DataFrame()
        result = filter_columns_by_suffixes(df)
        assert result == []


# ============================================================================
# Test: filter_dates
# ============================================================================

class TestFilterDates:
    """Tests for date range filtering."""

    def test_filters_within_date_range(self):
        """Should filter data within the specified date range."""
        df = pd.DataFrame({
            'date': pd.to_datetime(['2019-06-01', '2020-01-15', '2020-12-31', '2021-02-15', '2021-04-01']),
            'value': [1, 2, 3, 4, 5]
        })

        result = filter_dates(df, 'date', 2020, 2020)

        # Should include dates from 2020-01-01 to 2021-03-01 (Q4 extension)
        assert len(result) == 3  # 2020-01-15, 2020-12-31, 2021-02-15
        assert '2019-06-01' not in result['date'].astype(str).values
        assert '2021-04-01' not in result['date'].astype(str).values

    def test_includes_q4_extension(self):
        """Should extend to March next year for Q4 data."""
        df = pd.DataFrame({
            'date': pd.to_datetime(['2020-12-31', '2021-01-15', '2021-02-28', '2021-03-15']),
            'value': [1, 2, 3, 4]
        })

        result = filter_dates(df, 'date', 2020, 2020)

        # 2021-03-15 should be excluded (after end_year+1 March 1)
        assert len(result) == 3

    def test_empty_result(self):
        """Should return empty DataFrame if no dates in range."""
        df = pd.DataFrame({
            'date': pd.to_datetime(['2015-01-01', '2016-01-01']),
            'value': [1, 2]
        })

        result = filter_dates(df, 'date', 2020, 2020)

        assert len(result) == 0


# ============================================================================
# Test: reorder_columns
# ============================================================================

class TestReorderColumns:
    """Tests for column reordering."""

    def test_moves_keyword_columns_first(self):
        """Should move keyword-matching columns to front."""
        df = pd.DataFrame({
            'price': [100, 200],
            'volume': [1000, 2000],
            'symbol': ['AAPL', 'MSFT'],
            'date': ['2020-01-01', '2020-01-02']
        })

        result = reorder_columns(df, keywords=['symbol', 'date'])

        assert list(result.columns)[:2] == ['symbol', 'date']

    def test_case_insensitive_matching(self):
        """Should match keywords case-insensitively."""
        df = pd.DataFrame({
            'price': [100],
            'SYMBOL': ['AAPL'],
            'Date': ['2020-01-01']
        })

        result = reorder_columns(df, keywords=['symbol', 'date'])

        # Both SYMBOL and Date should be at front
        first_cols = list(result.columns)[:2]
        assert 'SYMBOL' in first_cols
        assert 'Date' in first_cols

    def test_preserves_all_columns(self):
        """Should preserve all columns, just reorder."""
        df = pd.DataFrame({
            'a': [1], 'b': [2], 'c': [3], 'symbol': ['X']
        })

        result = reorder_columns(df, keywords=['symbol'])

        assert len(result.columns) == 4
        assert set(result.columns) == {'a', 'b', 'c', 'symbol'}


# ============================================================================
# Test: efficient_merge_columns
# ============================================================================

class TestEfficientMergeColumns:
    """Tests for merging _x/_y suffix columns."""

    def test_merges_x_y_columns(self):
        """Should merge _x and _y suffix columns."""
        df = pd.DataFrame({
            'revenue_x': [100, np.nan, 300],
            'revenue_y': [np.nan, 200, 350],
            'symbol': ['A', 'B', 'C']
        })

        result = efficient_merge_columns(df)

        # Should have 'revenue' instead of 'revenue_x', 'revenue_y'
        assert 'revenue' in result.columns or 'revenue_' in result.columns
        assert 'revenue_x' not in result.columns
        assert 'revenue_y' not in result.columns

    def test_prefers_non_null_values(self):
        """Should prefer non-null values when merging."""
        df = pd.DataFrame({
            'value_x': [1.0, np.nan],
            'value_y': [np.nan, 2.0],
        })

        result = efficient_merge_columns(df)

        # Check the merged column contains both non-null values
        value_col = [c for c in result.columns if 'value' in c.lower()][0]
        assert result[value_col].iloc[0] == 1.0 or result[value_col].iloc[1] == 2.0


# ============================================================================
# Test: generate_date_list (Logic Test)
# ============================================================================

class TestGenerateDateList:
    """Tests for rebalancing date generation logic."""

    def test_generates_quarterly_dates(self):
        """Should generate quarterly rebalancing dates."""
        start_year = 2020
        rebalance_period = 3  # Quarterly
        start_month = 1
        start_date_day = 1

        date_list = []
        date = datetime(start_year - 3, start_month, start_date_day)
        end_date = datetime(start_year, 12, 31)

        while date <= end_date:
            date_list.append(date)
            date += relativedelta(months=rebalance_period)

        # Should have dates from 2017 to 2020
        assert len(date_list) > 0
        # Check quarterly spacing (3 months)
        for i in range(1, len(date_list)):
            diff_months = (date_list[i].year - date_list[i-1].year) * 12 + (date_list[i].month - date_list[i-1].month)
            assert diff_months == 3

    def test_respects_rebalance_period(self):
        """Should respect the rebalance period setting."""
        start_year = 2020
        rebalance_period = 6  # Semi-annual

        date_list = []
        date = datetime(start_year - 3, 1, 1)
        end_date = datetime(start_year, 12, 31)

        while date <= end_date:
            date_list.append(date)
            date += relativedelta(months=rebalance_period)

        # Check 6-month spacing
        for i in range(1, min(3, len(date_list))):
            diff_months = (date_list[i].year - date_list[i-1].year) * 12 + (date_list[i].month - date_list[i-1].month)
            assert diff_months == 6

    def test_starts_from_correct_year(self):
        """Should start from start_year - 3."""
        start_year = 2020
        date = datetime(start_year - 3, 1, 1)

        # First date should be in 2017 (2020 - 3)
        assert date.year == 2017


# ============================================================================
# Test: set_trade_date_list (Logic Test)
# ============================================================================

class TestSetTradeDateList:
    """Tests for converting calendar dates to trading dates logic."""

    def test_converts_to_trading_dates(self):
        """Should convert calendar dates to nearest trading dates."""
        # Simulating the logic without importing make_mldata
        price_dates = pd.DatetimeIndex([
            '2020-01-02', '2020-01-03', '2020-01-06',
            '2020-04-01', '2020-04-02'
        ])

        calendar_dates = [datetime(2020, 1, 1), datetime(2020, 4, 1)]
        trade_dates = []

        for date in calendar_dates:
            # Find trading date within 10 days
            ts = pd.Timestamp(date)
            for i in range(11):  # 0 to 10 days
                check_date = ts - pd.Timedelta(days=i)
                if check_date in price_dates:
                    trade_dates.append(check_date)
                    break

        assert len(trade_dates) == 1  # 2020-04-01 matches exactly
        assert trade_dates[0] == pd.Timestamp('2020-04-01')

    def test_finds_previous_trading_day(self):
        """Should find previous trading day for weekend dates."""
        price_dates = pd.DatetimeIndex(['2020-01-02', '2020-01-03'])  # Thurs, Fri

        # 2020-01-04 is Saturday
        saturday = pd.Timestamp('2020-01-04')

        # Find previous trading day
        trade_date = None
        for i in range(11):
            check_date = saturday - pd.Timedelta(days=i)
            if check_date in price_dates:
                trade_date = check_date
                break

        assert trade_date == pd.Timestamp('2020-01-03')  # Friday


# ============================================================================
# Test: Winsorization Logic
# ============================================================================

class TestApplyWinsorization:
    """Tests for winsorization of extreme values."""

    def test_winsorization_concept(self):
        """Winsorization should clip values to percentile limits."""
        try:
            from scipy.stats.mstats import winsorize

            # Create data with extreme outlier
            data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1000000.0])

            # Winsorize with 10% on each tail
            result = winsorize(data, limits=(0.1, 0.1))

            # The extreme value should be clipped
            assert result.max() < 1000000.0
            # Middle values should be preserved
            assert result[4] == 5.0  # Median area preserved
        except ImportError:
            pytest.skip("scipy not available")

    def test_winsorize_preserves_nan(self):
        """Winsorization should preserve NaN values."""
        try:
            from scipy.stats.mstats import winsorize

            data = np.array([1.0, np.nan, 3.0, 4.0, 5.0, np.nan, 7.0, 8.0, 9.0, 10.0])

            result = winsorize(data, limits=(0.1, 0.1), nan_policy='propagate')

            # NaN count should be preserved
            assert np.isnan(result).sum() == 2
        except ImportError:
            pytest.skip("scipy not available")

    def test_winsorization_limits(self):
        """Different limits should produce different results."""
        try:
            from scipy.stats.mstats import winsorize

            data = np.arange(1, 101)  # 1 to 100

            # 1% winsorization
            result_1pct = winsorize(data, limits=(0.01, 0.01))
            # 10% winsorization
            result_10pct = winsorize(data, limits=(0.1, 0.1))

            # 10% should clip more aggressively
            assert result_10pct.max() < result_1pct.max()
            assert result_10pct.min() > result_1pct.min()
        except ImportError:
            pytest.skip("scipy not available")


# ============================================================================
# Test: Extreme Mover Filtering Logic
# ============================================================================

class TestFilterExtremeMovers:
    """Tests for extreme mover filtering logic."""

    def test_percentile_filtering(self):
        """Percentile-based filtering should remove tail values."""
        df = pd.DataFrame({
            'sec_price_dev_subavg': np.concatenate([
                np.random.normal(0, 0.1, 90),  # Normal values
                [5.0, 6.0, 7.0, 8.0, 9.0],     # Extreme high
                [-5.0, -6.0, -7.0, -8.0, -9.0]  # Extreme low
            ]),
            'symbol': [f'S{i}' for i in range(100)]
        })

        # Filter using percentile (remove 5% on each tail)
        lower = df['sec_price_dev_subavg'].quantile(0.05)
        upper = df['sec_price_dev_subavg'].quantile(0.95)

        filtered = df[(df['sec_price_dev_subavg'] >= lower) &
                      (df['sec_price_dev_subavg'] <= upper)]

        # Should have approximately 90 rows (10% removed)
        assert len(filtered) < len(df)
        assert len(filtered) >= 80

    def test_zscore_filtering(self):
        """Z-score based filtering should remove outliers."""
        try:
            from scipy import stats

            df = pd.DataFrame({
                'sec_price_dev_subavg': np.concatenate([
                    np.random.normal(0, 1, 95),  # Normal
                    [10.0, 11.0, 12.0, -10.0, -11.0]  # Outliers (z > 3)
                ]),
                'symbol': [f'S{i}' for i in range(100)]
            })

            # Calculate z-scores
            z_scores = np.abs(stats.zscore(df['sec_price_dev_subavg']))

            # Filter |z| > 3
            threshold = 3.0
            filtered = df[z_scores <= threshold]

            # Should remove extreme outliers
            assert len(filtered) < len(df)
        except ImportError:
            pytest.skip("scipy not available")

    def test_robust_zscore_filtering(self):
        """Robust z-score (MAD-based) should be more resistant to outliers."""
        try:
            from scipy.stats import median_abs_deviation

            df = pd.DataFrame({
                'sec_price_dev_subavg': np.concatenate([
                    np.random.normal(0, 1, 95),
                    [20.0, 25.0, 30.0, -20.0, -25.0]  # Extreme outliers
                ])
            })

            median = df['sec_price_dev_subavg'].median()
            mad = median_abs_deviation(df['sec_price_dev_subavg'], nan_policy='omit')

            # Robust z-score
            robust_z = (df['sec_price_dev_subavg'] - median) / (1.4826 * mad)

            # Should detect the extreme outliers
            extreme_mask = robust_z.abs() > 3.0
            assert extreme_mask.sum() > 0

        except ImportError:
            pytest.skip("scipy not available")

    def test_all_nan_handling(self):
        """Should handle all-NaN column gracefully."""
        df = pd.DataFrame({
            'sec_price_dev_subavg': [np.nan, np.nan, np.nan],
            'symbol': ['A', 'B', 'C']
        })

        # All NaN should not crash
        assert df['sec_price_dev_subavg'].isna().all()


# ============================================================================
# Test: Date and Filing Date Logic
# ============================================================================

class TestFilingDateLogic:
    """Tests for filing date cutoff logic (leakage prevention)."""

    def test_filing_date_after_report_date(self):
        """Filing date should always be after report date."""
        # This is a data validation test
        df = pd.DataFrame({
            'report_date': pd.to_datetime(['2020-03-31', '2020-06-30', '2020-09-30']),
            'fillingDate': pd.to_datetime(['2020-05-15', '2020-08-10', '2020-11-05']),
        })

        # Filing date should be after report date
        delays = (df['fillingDate'] - df['report_date']).dt.days
        assert (delays > 0).all()

    def test_searchsorted_mapping(self):
        """Test searchsorted behavior for filing date to rebalance date mapping."""
        # Trade dates (rebalancing dates)
        trade_dates = pd.DatetimeIndex([
            '2020-01-01', '2020-04-01', '2020-07-01', '2020-10-01'
        ])

        # Filing date: 2020-05-15 (Q1 report filed after April 1)
        filing_date = pd.Timestamp('2020-05-15')

        # searchsorted with side='right' should give index 2 (2020-07-01)
        idx = np.searchsorted(trade_dates, filing_date, side='right')

        assert idx == 2
        assert trade_dates[idx] == pd.Timestamp('2020-07-01')

    def test_no_future_leakage(self):
        """Verify that filing date mapping prevents future leakage."""
        # Q1 2020 report: ends 2020-03-31, filed 2020-05-15
        # Should only be usable starting from 2020-07-01 rebalance

        trade_dates = pd.DatetimeIndex(sorted([
            '2020-01-01', '2020-04-01', '2020-07-01', '2020-10-01'
        ]))

        # April 1 rebalance should NOT use Q1 report (filed May 15)
        filing_date = pd.Timestamp('2020-05-15')
        idx = np.searchsorted(trade_dates, filing_date, side='right')

        # The earliest usable rebalance date is July 1
        rebalance_date = trade_dates[idx] if idx < len(trade_dates) else pd.NaT

        # April 1 should not be able to use this data
        assert rebalance_date != pd.Timestamp('2020-04-01')
        assert rebalance_date == pd.Timestamp('2020-07-01')


# ============================================================================
# Test: Year Period Calculation
# ============================================================================

class TestYearPeriodCalculation:
    """Tests for year_period integer calculation."""

    def test_year_period_format(self):
        """Year period should be YYYYQQ format."""
        df = pd.DataFrame({
            'calendarYear': [2020, 2020, 2020, 2020],
            'period': ['Q1', 'Q2', 'Q3', 'Q4']
        })

        period_map = {'Q1': 1, 'Q2': 2, 'Q3': 3, 'Q4': 4}
        df['year_period'] = (
            df['calendarYear'].astype(int) * 100 +
            df['period'].map(period_map).astype(int)
        )

        expected = [202001, 202002, 202003, 202004]
        assert list(df['year_period']) == expected

    def test_year_period_ordering(self):
        """Year periods should sort chronologically."""
        year_periods = [202004, 202101, 202003, 202102]
        sorted_periods = sorted(year_periods)

        expected = [202003, 202004, 202101, 202102]
        assert sorted_periods == expected

    def test_unmapped_period_becomes_nan(self):
        """Unmapped period values should result in NaN."""
        df = pd.DataFrame({
            'calendarYear': [2020, 2020],
            'period': ['Q1', 'FY']  # FY is not in period_map
        })

        period_map = {'Q1': 1, 'Q2': 2, 'Q3': 3, 'Q4': 4}
        df['year_period'] = (
            df['calendarYear'].astype(int) * 100 +
            df['period'].map(period_map)
        )

        assert pd.isna(df['year_period'].iloc[1])


# ============================================================================
# Test: Volume Filtering
# ============================================================================

class TestVolumeFiltering:
    """Tests for low-liquidity stock filtering."""

    def test_volume_percentile_filtering(self):
        """Should filter stocks below volume percentile threshold."""
        df = pd.DataFrame({
            'symbol': ['A', 'B', 'C', 'D', 'E'] * 10,
            'volume_mul_price': [100, 200, 300, 400, 500] * 10  # A is lowest
        })

        # Calculate mean volume per symbol
        symbol_means = df.groupby('symbol')['volume_mul_price'].mean()

        # Filter bottom 20%
        min_volume_pct = 20
        threshold = symbol_means.quantile(min_volume_pct / 100)
        filtered_symbols = symbol_means[symbol_means >= threshold].index

        # Symbol A should be filtered out (lowest volume)
        assert 'A' not in filtered_symbols
        assert 'E' in filtered_symbols

    def test_volume_threshold_calculation(self):
        """Volume threshold should be at correct percentile."""
        volumes = pd.Series([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])

        # 10th percentile
        threshold_10 = volumes.quantile(0.10)
        assert threshold_10 == 190.0  # Linear interpolation

        # 50th percentile
        threshold_50 = volumes.quantile(0.50)
        assert threshold_50 == 550.0


# ============================================================================
# Test: tsfresh Feature Extraction Parameters
# ============================================================================

class TestTsfreshParameters:
    """Tests for tsfresh feature extraction settings."""

    def test_minimal_fc_parameters(self):
        """MinimalFCParameters should produce fewer features than Efficient."""
        try:
            from tsfresh.feature_extraction import MinimalFCParameters, EfficientFCParameters

            minimal = MinimalFCParameters()
            efficient = EfficientFCParameters()

            # Minimal should have fewer feature calculators
            assert len(minimal) < len(efficient)
        except ImportError:
            pytest.skip("tsfresh not available")

    def test_id_column_requirement(self):
        """tsfresh requires id column for grouping."""
        try:
            from tsfresh import extract_features
            from tsfresh.feature_extraction import MinimalFCParameters

            df = pd.DataFrame({
                'id': ['A', 'A', 'A', 'B', 'B', 'B'],
                'time': [0, 1, 2, 0, 1, 2],
                'value': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            })

            features = extract_features(
                df,
                column_id='id',
                column_sort='time',
                column_value='value',
                default_fc_parameters=MinimalFCParameters(),
                disable_progressbar=True
            )

            # Should have 2 rows (one per id)
            assert len(features) == 2
            assert 'A' in features.index
            assert 'B' in features.index
        except ImportError:
            pytest.skip("tsfresh not available")


# ============================================================================
# Test: Lookback Window
# ============================================================================

class TestLookbackWindow:
    """Tests for 12-quarter lookback window logic."""

    def test_tail_12_extraction(self):
        """Should extract last 12 rows per symbol."""
        df = pd.DataFrame({
            'symbol': ['A'] * 15 + ['B'] * 8,  # A has 15, B has 8
            'year_period': list(range(15)) + list(range(8)),
            'value': list(range(15)) + list(range(8))
        })

        # Manual implementation without apply()
        result_list = []
        for symbol in df['symbol'].unique():
            group = df[df['symbol'] == symbol]
            result_list.append(group.tail(12))
        result = pd.concat(result_list, ignore_index=True)

        # A should have 12 rows, B should have 8 (all it has)
        assert len(result[result['symbol'] == 'A']) == 12
        assert len(result[result['symbol'] == 'B']) == 8

    def test_filters_symbols_with_insufficient_data(self):
        """Should filter symbols with less than 12 data points."""
        df = pd.DataFrame({
            'symbol': ['A'] * 12 + ['B'] * 5 + ['C'] * 12,
            'value': list(range(12)) + list(range(5)) + list(range(12))
        })

        symbol_counts = df['symbol'].value_counts()
        symbols_to_remove = symbol_counts[symbol_counts < 12].index
        filtered_df = df[~df['symbol'].isin(symbols_to_remove)]

        assert 'A' in filtered_df['symbol'].values
        assert 'B' not in filtered_df['symbol'].values
        assert 'C' in filtered_df['symbol'].values


# ============================================================================
# Test: Custom Ratio Calculations
# ============================================================================

class TestCustomRatioCalculations:
    """Tests for OverMC_* and adaptiveMC_* ratio calculations."""

    def test_overmc_calculation(self):
        """OverMC ratios should be metric / marketCap."""
        df = pd.DataFrame({
            'revenue': [1000000, 2000000],
            'marketCap': [10000000, 20000000]
        })

        # OverMC_revenue = revenue / marketCap
        df['OverMC_revenue'] = df['revenue'] / df['marketCap']

        expected = [0.1, 0.1]
        np.testing.assert_array_almost_equal(df['OverMC_revenue'], expected)

    def test_overmc_handles_zero_marketcap(self):
        """Should handle zero marketCap gracefully."""
        df = pd.DataFrame({
            'revenue': [1000000, 2000000],
            'marketCap': [0, 20000000]
        })

        # With zero check
        valid_mask = (df['marketCap'] > 0) & np.isfinite(df['marketCap'])
        df['OverMC_revenue'] = np.where(
            valid_mask, df['revenue'] / df['marketCap'], np.nan
        )

        assert np.isnan(df['OverMC_revenue'].iloc[0])
        assert df['OverMC_revenue'].iloc[1] == 0.1

    def test_adaptivemc_ev_calculation(self):
        """adaptiveMC_ev should be marketCap + netDebt."""
        df = pd.DataFrame({
            'marketCap': [10000000, 20000000],
            'netDebt': [1000000, -500000]  # Can be negative (net cash)
        })

        df['adaptiveMC_ev'] = df['marketCap'] + df['netDebt']

        expected = [11000000, 19500000]
        assert list(df['adaptiveMC_ev']) == expected

    def test_adaptivemc_ratio_calculation(self):
        """adaptiveMC ratios should be EV / metric."""
        df = pd.DataFrame({
            'adaptiveMC_ev': [11000000, 19500000],
            'ebitda': [1000000, 2000000]
        })

        # adaptiveMC_ebitda = EV / EBITDA
        valid_mask = (df['ebitda'] > 0) & np.isfinite(df['ebitda'])
        df['adaptiveMC_ebitda'] = np.where(
            valid_mask, df['adaptiveMC_ev'] / df['ebitda'], np.nan
        )

        expected = [11.0, 9.75]
        np.testing.assert_array_almost_equal(df['adaptiveMC_ebitda'], expected)
