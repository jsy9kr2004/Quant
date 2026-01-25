"""
Windows 호환 파일명 처리 유틸리티

Windows에서는 특정 이름들이 디바이스 이름으로 예약되어 있어 파일명으로 사용할 수 없습니다.
이 모듈은 symbol을 안전한 파일명으로 변환하고, 다시 원래 symbol로 복원하는 함수를 제공합니다.

예약어:
- CON, PRN, AUX, NUL
- COM1, COM2, ..., COM9
- LPT1, LPT2, ..., LPT9

사용 예시:
    from src.infra.file_utils import safe_filename, restore_symbol_from_filename

    # 파일 저장 시
    symbol = "CON"
    filename = safe_filename(symbol)  # "RESERVED_CON"
    df.to_parquet(f"{path}/{filename}.parquet")

    # 파일 읽기 시
    symbol = "CON"
    filename = safe_filename(symbol)  # "RESERVED_CON"
    df = pd.read_parquet(f"{path}/{filename}.parquet")

    # 파일명에서 원래 심볼 복원
    original = restore_symbol_from_filename("RESERVED_CON")  # "CON"
"""

# Windows 예약어 목록 (대소문자 구분 없음)
WINDOWS_RESERVED_NAMES = {
    'CON', 'PRN', 'AUX', 'NUL',
    'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
    'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
}

# 예약어 변환 접두사
RESERVED_PREFIX = "RESERVED_"


def is_windows_reserved(name: str) -> bool:
    """
    주어진 이름이 Windows 예약어인지 확인합니다.

    Args:
        name: 확인할 이름 (symbol 등)

    Returns:
        bool: Windows 예약어이면 True, 아니면 False

    Examples:
        >>> is_windows_reserved("CON")
        True
        >>> is_windows_reserved("AAPL")
        False
        >>> is_windows_reserved("con")  # 대소문자 구분 없음
        True
    """
    return name.upper() in WINDOWS_RESERVED_NAMES


def safe_filename(symbol: str) -> str:
    """
    Symbol을 Windows에서 안전한 파일명으로 변환합니다.

    Windows 예약어인 경우 'RESERVED_' 접두사를 추가합니다.
    그렇지 않은 경우 원본 symbol을 그대로 반환합니다.

    Args:
        symbol: 변환할 symbol (예: "CON", "AAPL")

    Returns:
        str: 안전한 파일명 (예: "RESERVED_CON", "AAPL")

    Examples:
        >>> safe_filename("CON")
        'RESERVED_CON'
        >>> safe_filename("AAPL")
        'AAPL'
        >>> safe_filename("PRN")
        'RESERVED_PRN'
    """
    if is_windows_reserved(symbol):
        return f"{RESERVED_PREFIX}{symbol}"
    return symbol


def restore_symbol_from_filename(filename: str) -> str:
    """
    안전한 파일명에서 원래 symbol을 복원합니다.

    'RESERVED_' 접두사가 있으면 제거하여 원본 symbol을 반환합니다.
    그렇지 않은 경우 원본 파일명을 그대로 반환합니다.

    Args:
        filename: 복원할 파일명 (확장자 제외, 예: "RESERVED_CON", "AAPL")

    Returns:
        str: 원본 symbol (예: "CON", "AAPL")

    Examples:
        >>> restore_symbol_from_filename("RESERVED_CON")
        'CON'
        >>> restore_symbol_from_filename("AAPL")
        'AAPL'
        >>> restore_symbol_from_filename("RESERVED_PRN")
        'PRN'
    """
    if filename.startswith(RESERVED_PREFIX):
        return filename[len(RESERVED_PREFIX):]
    return filename
