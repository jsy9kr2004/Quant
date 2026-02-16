"""
Config 마스킹 유틸리티

민감한 정보(API_KEY 등)를 마스킹하여 Drive 업로드용 Config 생성
"""

import copy
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class ConfigMasker:
    """Config에서 민감한 정보를 마스킹하는 클래스"""

    MASK_VALUE = "***MASKED***"

    def __init__(self, mask_keys: List[str]):
        """
        초기화

        Args:
            mask_keys: 마스킹할 키 목록 (예: ["API_KEY", "DATABASE_PASSWORD"])
        """
        self.mask_keys = [key.upper() for key in mask_keys]  # 대소문자 무시
        logger.debug(f"ConfigMasker initialized with keys: {self.mask_keys}")

    def mask_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Config에서 민감한 정보 마스킹

        Args:
            config: 원본 config dict

        Returns:
            마스킹된 config dict (원본은 변경 안 됨)
        """
        # Deep copy로 원본 보호
        masked_config = copy.deepcopy(config)

        # 재귀적으로 마스킹
        self._recursive_mask(masked_config)

        logger.info(f"Config masked: {len(self.mask_keys)} sensitive keys processed")
        return masked_config

    def _recursive_mask(self, obj: Any, path: str = "") -> None:
        """
        재귀적으로 dict를 순회하며 민감한 키 마스킹

        Args:
            obj: 처리할 객체 (dict, list, 또는 primitive)
            path: 현재 경로 (디버깅용)
        """
        if isinstance(obj, dict):
            for key, value in obj.items():
                current_path = f"{path}.{key}" if path else key

                # 키 이름이 마스킹 대상인지 확인 (대소문자 무시)
                if key.upper() in self.mask_keys:
                    # 값이 이미 마스킹되어 있는지 확인
                    if value != self.MASK_VALUE:
                        logger.debug(f"Masking: {current_path}")
                        obj[key] = self.MASK_VALUE
                else:
                    # 재귀적으로 하위 탐색
                    self._recursive_mask(value, current_path)

        elif isinstance(obj, list):
            # 리스트의 각 요소 처리
            for i, item in enumerate(obj):
                current_path = f"{path}[{i}]"
                self._recursive_mask(item, current_path)

        # Primitive 타입 (str, int, float 등)은 처리 안 함

    def extract_changes(self, template: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, tuple]:
        """
        Template config와 현재 config를 비교하여 변경사항 추출

        Args:
            template: 기준 config (conf.yaml.template)
            current: 현재 config (conf.yaml)

        Returns:
            변경사항 dict: {key: (before, after), ...}
            예: {"BACKTEST.TOPK": (5, 10), "ML.LOSS_THRESHOLD": (-0.3, -0.4)}
        """
        changes = {}
        self._compare_recursive(template, current, changes, path="")

        logger.info(f"Config changes detected: {len(changes)} parameters")
        return changes

    def _compare_recursive(
        self,
        template: Any,
        current: Any,
        changes: Dict[str, tuple],
        path: str
    ) -> None:
        """
        재귀적으로 두 config를 비교하여 변경사항 추출

        Args:
            template: 기준값
            current: 현재값
            changes: 변경사항을 저장할 dict
            path: 현재 경로 (예: "BACKTEST.TOPK")
        """
        # 타입이 다르면 변경으로 간주
        if type(template) != type(current):
            changes[path] = (template, current)
            return

        if isinstance(template, dict) and isinstance(current, dict):
            # 모든 키 확인 (template + current)
            all_keys = set(template.keys()) | set(current.keys())

            for key in all_keys:
                current_path = f"{path}.{key}" if path else key

                # Template에만 있는 키 (current에서 삭제됨)
                if key not in current:
                    changes[current_path] = (template[key], None)

                # Current에만 있는 키 (새로 추가됨)
                elif key not in template:
                    changes[current_path] = (None, current[key])

                # 둘 다 있으면 재귀 비교
                else:
                    self._compare_recursive(
                        template[key],
                        current[key],
                        changes,
                        current_path
                    )

        elif isinstance(template, list) and isinstance(current, list):
            # 리스트는 길이나 내용이 다르면 통째로 변경으로 간주
            if template != current:
                changes[path] = (template, current)

        else:
            # Primitive 타입 비교
            if template != current:
                changes[path] = (template, current)

    def format_changes_for_display(self, changes: Dict[str, tuple]) -> str:
        """
        변경사항을 간결한 문자열로 포맷팅

        Args:
            changes: extract_changes()의 결과

        Returns:
            포맷팅된 문자열 (예: "TOPK=10, START_MONTH=3")
        """
        if not changes:
            return "(default)"

        # 주요 파라미터만 표시 (너무 많으면 가독성 저하)
        # 우선순위: BACKTEST > ML > DATA
        priority_keys = [
            "BACKTEST.TOPK",
            "BACKTEST.START_MONTH",
            "BACKTEST.START_DATE",
            "ML.CLASSIFIER_MODE",
            "ML.NEGATIVE_SCREEN.LOSS_THRESHOLD",
            "ML.USE_SECTOR_MODEL",
            "ML.USE_CLASSIFIER",
        ]

        # 우선순위 키부터 표시
        display_items = []
        for key in priority_keys:
            if key in changes:
                before, after = changes[key]
                # 경로에서 마지막 부분만 사용 (BACKTEST.TOPK → TOPK)
                short_key = key.split(".")[-1]
                # 민감 키 마스킹: 경로의 어떤 부분이든 mask_keys에 포함되면 마스킹
                key_parts = key.upper().split(".")
                if any(part in self.mask_keys for part in key_parts):
                    display_items.append(f"{short_key}={self.MASK_VALUE}")
                else:
                    display_items.append(f"{short_key}={after}")

        # 우선순위에 없는 나머지 변경사항
        for key, (before, after) in changes.items():
            if key not in priority_keys:
                short_key = key.split(".")[-1]
                # 민감 키 마스킹
                key_parts = key.upper().split(".")
                if any(part in self.mask_keys for part in key_parts):
                    display_items.append(f"{short_key}={self.MASK_VALUE}")
                else:
                    display_items.append(f"{short_key}={after}")

        return ", ".join(display_items)


# 테스트용 코드
if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(level=logging.DEBUG)

    # 테스트 데이터
    test_config = {
        "DATA": {
            "API_KEY": "secret123",
            "START_YEAR": 1996
        },
        "ML": {
            "USE_CLASSIFIER": True,
            "NEGATIVE_SCREEN": {
                "LOSS_THRESHOLD": -0.3
            }
        },
        "BACKTEST": {
            "TOPK": 5,
            "START_MONTH": 1
        }
    }

    template_config = {
        "DATA": {
            "API_KEY": "template_key",
            "START_YEAR": 1996
        },
        "ML": {
            "USE_CLASSIFIER": True,
            "NEGATIVE_SCREEN": {
                "LOSS_THRESHOLD": -0.3
            }
        },
        "BACKTEST": {
            "TOPK": 5,
            "START_MONTH": 1
        }
    }

    current_config = {
        "DATA": {
            "API_KEY": "real_secret_key",
            "START_YEAR": 1996
        },
        "ML": {
            "USE_CLASSIFIER": True,
            "NEGATIVE_SCREEN": {
                "LOSS_THRESHOLD": -0.4  # Changed!
            }
        },
        "BACKTEST": {
            "TOPK": 10,  # Changed!
            "START_MONTH": 3  # Changed!
        }
    }

    # 마스커 생성
    masker = ConfigMasker(mask_keys=["API_KEY", "DATABASE_PASSWORD"])

    # 테스트 1: 마스킹
    print("=== Test 1: Masking ===")
    masked = masker.mask_config(test_config)
    print(f"Original API_KEY: {test_config['DATA']['API_KEY']}")
    print(f"Masked API_KEY: {masked['DATA']['API_KEY']}")
    print()

    # 테스트 2: 변경사항 추출
    print("=== Test 2: Extract Changes ===")
    changes = masker.extract_changes(template_config, current_config)
    for key, (before, after) in changes.items():
        print(f"{key}: {before} → {after}")
    print()

    # 테스트 3: 변경사항 포맷팅
    print("=== Test 3: Format Changes ===")
    formatted = masker.format_changes_for_display(changes)
    print(f"Changes: {formatted}")
