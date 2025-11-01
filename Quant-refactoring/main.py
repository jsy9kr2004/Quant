#!/usr/bin/env python
"""
Quant 트레이딩 시스템의 메인 진입점입니다.

이 모듈은 트레이딩 전략을 백테스팅하기 위한 커맨드 라인 인터페이스를 제공합니다.
다양한 데이터 소스(FMP, MarketDB)와 models 모듈에 정의된 트레이딩 전략을
지원합니다.

사용 예시:
    python main.py backtest --start-date 2020-01-01 --end-date 2020-12-31
    python main.py backtest --strategy-name momentum --capital 10000000
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import click
from dotenv import load_dotenv

from config.context_loader import MainContext
from models.base_model import BaseModel
from models.momentum import MomentumModel
from models.mean_reversion import MeanReversionModel
from data_sources.fmp_source import FMPDataSource
from data_sources.marketdb_source import MarketDBDataSource

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_strategy_class(strategy_name: str) -> type[BaseModel]:
    """
    전략 이름으로 전략 클래스를 가져옵니다.

    Args:
        strategy_name: 전략 이름 ('momentum', 'mean_reversion')

    Returns:
        Strategy 클래스

    Raises:
        ValueError: 전략 이름이 지원되지 않는 경우
        NotImplementedError: 전략 모듈이 아직 구현되지 않은 경우

    사용 예시:
        strategy_cls = get_strategy_class('momentum')
        isinstance(strategy_cls, type)
        True
    """
    # TODO: 전략 모듈 구현 후 활성화
    raise NotImplementedError(
        "전략 모듈이 아직 구현되지 않았습니다. "
        "models/ 디렉토리에 구체적인 전략을 구현해주세요."
    )

    # strategies = {
    #     'momentum': MomentumModel,
    #     'mean_reversion': MeanReversionModel,
    # }
    #
    # if strategy_name not in strategies:
    #     raise ValueError(
    #         f"Unknown strategy: {strategy_name}. "
    #         f"Available strategies: {', '.join(strategies.keys())}"
    #     )
    #
    # return strategies[strategy_name]


def create_context(
    data_source: str,
    start_date: str,
    end_date: str,
    initial_capital: float,
    config: Optional[Dict[str, Any]] = None
) -> MainContext:
    """
    지정된 데이터 소스와 파라미터로 MainCtx를 생성합니다.

    Args:
        data_source: 데이터 소스 이름 ('fmp' 또는 'marketdb')
        start_date: YYYY-MM-DD 형식의 시작 날짜
        end_date: YYYY-MM-DD 형식의 종료 날짜
        initial_capital: 트레이딩 초기 자본금
        config: 추가 설정 딕셔너리

    Returns:
        설정된 MainCtx 인스턴스

    Raises:
        ValueError: 데이터 소스가 지원되지 않는 경우

    사용 예시:
        ctx = create_context('fmp', '2020-01-01', '2020-12-31', 10000000)
        ctx.data_source is not None
        True
    """
    # Initialize data source
    if data_source.lower() == 'fmp':
        api_key = os.getenv('FMP_API_KEY')
        if not api_key:
            raise ValueError("FMP_API_KEY not found in environment variables")
        source = FMPDataSource(api_key=api_key)
    elif data_source.lower() == 'marketdb':
        source = MarketDBDataSource()
    else:
        raise ValueError(
            f"Unknown data source: {data_source}. "
            f"Available sources: fmp, marketdb"
        )

    # Create context
    ctx = MainCtx(
        data_source=source,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital
    )

    # Apply additional configuration if provided
    if config:
        for key, value in config.items():
            setattr(ctx, key, value)

    return ctx


def validate_dates(start_date: str, end_date: str) -> None:
    """
    날짜 형식과 범위를 검증합니다.

    Args:
        start_date: YYYY-MM-DD 형식의 시작 날짜
        end_date: YYYY-MM-DD 형식의 종료 날짜

    Raises:
        ValueError: 날짜 형식이 유효하지 않거나 end_date < start_date인 경우

    사용 예시:
        validate_dates('2020-01-01', '2020-12-31')  # 예외 없음
        validate_dates('2020-12-31', '2020-01-01')  # ValueError 발생
    """
    try:
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
    except ValueError as e:
        raise ValueError(f"Invalid date format. Use YYYY-MM-DD: {e}")

    if end < start:
        raise ValueError("end_date must be after start_date")


@click.group()
def cli():
    """
    Quant 트레이딩 시스템 CLI

    퀀트 트레이딩 전략을 백테스팅하고 실행하기 위한 커맨드 라인 인터페이스입니다.
    """
    pass


@cli.command()
@click.option(
    '--strategy-name',
    default='momentum',
    help='Strategy to use (momentum, mean_reversion)'
)
@click.option(
    '--data-source',
    default='fmp',
    help='Data source to use (fmp, marketdb)'
)
@click.option(
    '--start-date',
    required=True,
    help='Start date (YYYY-MM-DD)'
)
@click.option(
    '--end-date',
    required=True,
    help='End date (YYYY-MM-DD)'
)
@click.option(
    '--capital',
    default=10000000.0,
    type=float,
    help='Initial capital (default: 10,000,000 KRW)'
)
@click.option(
    '--top-n',
    default=20,
    type=int,
    help='Number of stocks to select (default: 20)'
)
@click.option(
    '--rebalance-period',
    default='monthly',
    help='Rebalance period (daily, weekly, monthly)'
)
@click.option(
    '--output-dir',
    default='./results',
    help='Directory to save results'
)
@click.option(
    '--verbose',
    is_flag=True,
    help='Enable verbose logging'
)
def backtest(
    strategy_name: str,
    data_source: str,
    start_date: str,
    end_date: str,
    capital: float,
    top_n: int,
    rebalance_period: str,
    output_dir: str,
    verbose: bool
):
    """
    트레이딩 전략에 대한 백테스트를 실행합니다.

    이 명령은 지정된 데이터 소스의 과거 데이터를 사용하여 백테스트를 실행합니다.
    전략의 성능을 평가하고 메트릭과 시각화가 포함된 상세 리포트를 생성합니다.

    사용 예시:
        python main.py backtest --start-date 2020-01-01 --end-date 2020-12-31
        python main.py backtest --strategy-name momentum --capital 10000000
        python main.py backtest --data-source marketdb --top-n 30
    """
    try:
        # Configure logging level
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.debug("Verbose logging enabled")

        # Validate inputs
        validate_dates(start_date, end_date)
        logger.info(f"Starting backtest: {strategy_name} strategy")
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info(f"Data source: {data_source}")
        logger.info(f"Initial capital: {capital:,.0f} KRW")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Results will be saved to: {output_path.absolute()}")

        # Get strategy class
        strategy_cls = get_strategy_class(strategy_name)

        # Create strategy configuration
        strategy_config = {
            'top_n': top_n,
            'rebalance_period': rebalance_period,
        }

        # Create context
        logger.info("Initializing context and data source...")
        ctx = create_context(
            data_source=data_source,
            start_date=start_date,
            end_date=end_date,
            initial_capital=capital
        )

        # Initialize strategy
        logger.info(f"Initializing {strategy_name} strategy...")
        strategy = strategy_cls(ctx=ctx, **strategy_config)

        # Run backtest
        logger.info("Running backtest...")
        results = strategy.backtest()

        # Display results
        logger.info("\n" + "="*50)
        logger.info("BACKTEST RESULTS")
        logger.info("="*50)

        if results:
            for key, value in results.items():
                if isinstance(value, float):
                    logger.info(f"{key}: {value:.2f}")
                else:
                    logger.info(f"{key}: {value}")

        # Save results
        logger.info(f"\nSaving results to {output_path}...")
        strategy.save_results(output_path)

        logger.info("\nBacktest completed successfully!")
        logger.info(f"Results saved to: {output_path.absolute()}")

    except Exception as e:
        logger.error(f"Backtest failed: {str(e)}", exc_info=verbose)
        sys.exit(1)


@cli.command()
@click.option(
    '--strategy-name',
    required=True,
    help='Strategy to use (momentum, mean_reversion)'
)
@click.option(
    '--data-source',
    default='fmp',
    help='Data source to use (fmp, marketdb)'
)
@click.option(
    '--capital',
    default=10000000.0,
    type=float,
    help='Initial capital (default: 10,000,000 KRW)'
)
@click.option(
    '--top-n',
    default=20,
    type=int,
    help='Number of stocks to select (default: 20)'
)
@click.option(
    '--config-file',
    type=click.Path(exists=True),
    help='Path to strategy configuration file (JSON)'
)
@click.option(
    '--dry-run',
    is_flag=True,
    help='Run in dry-run mode (no actual trades)'
)
def run(
    strategy_name: str,
    data_source: str,
    capital: float,
    top_n: int,
    config_file: Optional[str],
    dry_run: bool
):
    """
    라이브 모드(또는 드라이런 모드)로 전략을 실행합니다.

    이 명령은 실시간 데이터로 전략을 실행합니다. 드라이런 모드에서는
    실제 거래를 실행하지 않고 트레이딩을 시뮬레이션합니다.

    사용 예시:
        python main.py run --strategy-name momentum --dry-run
        python main.py run --strategy-name mean_reversion --config-file config.json
    """
    try:
        logger.info(f"Starting {strategy_name} strategy in {'dry-run' if dry_run else 'live'} mode")

        # Get current date for live trading
        today = datetime.now().strftime('%Y-%m-%d')

        # Load config if provided
        config = {}
        if config_file:
            import json
            with open(config_file, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {config_file}")

        # Get strategy class
        strategy_cls = get_strategy_class(strategy_name)

        # Create strategy configuration
        strategy_config = {
            'top_n': top_n,
            **config
        }

        # Create context for live mode (use today as both start and end)
        logger.info("Initializing context for live mode...")
        ctx = create_context(
            data_source=data_source,
            start_date=today,
            end_date=today,
            initial_capital=capital
        )

        # Initialize strategy
        logger.info(f"Initializing {strategy_name} strategy...")
        strategy = strategy_cls(ctx=ctx, **strategy_config)

        # Generate signals
        logger.info("Generating trading signals...")
        signals = strategy.generate_signals()

        # Display signals
        if signals:
            logger.info("\n" + "="*50)
            logger.info("TRADING SIGNALS")
            logger.info("="*50)
            logger.info(signals.to_string())
        else:
            logger.info("No trading signals generated")

        if dry_run:
            logger.info("\nDry-run mode: No trades executed")
        else:
            logger.info("\nLive mode: Trades would be executed here")
            # TODO: 실제 거래 실행 구현 필요

    except Exception as e:
        logger.error(f"Strategy execution failed: {str(e)}", exc_info=True)
        sys.exit(1)


@cli.command()
@click.option(
    '--strategy-name',
    required=True,
    help='Strategy to analyze'
)
@click.option(
    '--results-dir',
    default='./results',
    help='Directory containing backtest results'
)
def analyze(strategy_name: str, results_dir: str):
    """
    백테스트 결과를 분석하고 상세 리포트를 생성합니다.

    이 명령은 저장된 백테스트 결과를 로드하고 리스크 메트릭, 낙폭 분석,
    성능 기여도 분석을 포함한 추가 분석을 생성합니다.

    사용 예시:
        python main.py analyze --strategy-name momentum
        python main.py analyze --strategy-name momentum --results-dir ./custom_results
    """
    try:
        results_path = Path(results_dir)

        if not results_path.exists():
            raise ValueError(f"Results directory not found: {results_path}")

        logger.info(f"Analyzing results for {strategy_name} strategy")
        logger.info(f"Results directory: {results_path.absolute()}")

        # TODO: 결과 분석 기능 구현 필요
        logger.info("Analysis functionality coming soon...")

    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}", exc_info=True)
        sys.exit(1)


@cli.command()
def list_strategies():
    """
    사용 가능한 모든 트레이딩 전략을 나열합니다.

    각 전략의 이름, 설명, 필수 파라미터를 포함한 정보를 표시합니다.
    """
    strategies = {
        'momentum': {
            'name': 'Momentum Strategy',
            'description': 'Selects stocks with highest momentum (price returns)',
            'parameters': ['top_n', 'lookback_period', 'rebalance_period']
        },
        'mean_reversion': {
            'name': 'Mean Reversion Strategy',
            'description': 'Selects oversold stocks that are likely to revert to mean',
            'parameters': ['top_n', 'lookback_period', 'rebalance_period']
        }
    }

    logger.info("\n" + "="*50)
    logger.info("AVAILABLE STRATEGIES")
    logger.info("="*50 + "\n")

    for key, info in strategies.items():
        logger.info(f"Strategy: {key}")
        logger.info(f"  Name: {info['name']}")
        logger.info(f"  Description: {info['description']}")
        logger.info(f"  Parameters: {', '.join(info['parameters'])}")
        logger.info("")


@cli.command()
@click.option(
    '--data-source',
    default='fmp',
    help='Data source to validate (fmp, marketdb)'
)
def validate_data(data_source: str):
    """
    데이터 소스 연결과 데이터 가용성을 검증합니다.

    지정된 데이터 소스에 대한 연결을 테스트하고
    데이터를 성공적으로 가져올 수 있는지 확인합니다.

    사용 예시:
        python main.py validate-data --data-source fmp
        python main.py validate-data --data-source marketdb
    """
    try:
        logger.info(f"Validating {data_source} data source...")

        # Initialize data source
        if data_source.lower() == 'fmp':
            api_key = os.getenv('FMP_API_KEY')
            if not api_key:
                raise ValueError("FMP_API_KEY not found in environment variables")
            source = FMPDataSource(api_key=api_key)
        elif data_source.lower() == 'marketdb':
            source = MarketDBDataSource()
        else:
            raise ValueError(f"Unknown data source: {data_source}")

        # Test data fetching
        logger.info("Testing data fetch...")
        test_date = datetime.now().strftime('%Y-%m-%d')

        # Try to get available symbols
        symbols = source.get_available_symbols()
        logger.info(f"✓ Successfully fetched {len(symbols)} symbols")

        # Try to get data for first few symbols
        if symbols:
            test_symbols = symbols[:5]  # Test with first 5 symbols
            logger.info(f"Testing data fetch for symbols: {test_symbols}")

            data = source.fetch_data(
                symbols=test_symbols,
                start_date='2024-01-01',
                end_date=test_date
            )

            if not data.empty:
                logger.info(f"✓ Successfully fetched data: {len(data)} rows")
                logger.info(f"  Date range: {data.index.min()} to {data.index.max()}")
                logger.info(f"  Symbols: {data['symbol'].unique().tolist()}")
            else:
                logger.warning("⚠ No data returned (might be expected for recent dates)")

        logger.info(f"\n✓ {data_source} data source validation successful!")

    except Exception as e:
        logger.error(f"✗ Validation failed: {str(e)}", exc_info=True)
        sys.exit(1)


def main():
    """
    CLI의 메인 진입점입니다.

    Click CLI를 초기화하고 예외를 처리하지 못한 예외를 처리합니다.
    """
    try:
        cli()
    except KeyboardInterrupt:
        logger.info("\nOperation cancelled by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
