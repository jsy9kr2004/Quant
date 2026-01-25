"""
Memory Profiler for Walk-Forward Training

실행 중 메모리 사용량을 추적하고, 과거 데이터를 기반으로 최적의 worker 수를 계산합니다.
"""

import json
import psutil
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List, Any
import logging
import numpy as np


class MemoryProfiler:
    """Walk-Forward 학습의 메모리 사용량 추적 및 최적화"""

    def __init__(self, profile_path: str = "config/memory_profile.json"):
        self.profile_path = Path(profile_path)
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)

        self.current_run = None
        self.period_stats = []
        self.logger = logging.getLogger('MemoryProfiler')

        # 실행 중 메모리 추적
        self.memory_samples = []
        self.start_time = None

    def load_profile(self) -> Dict:
        """기존 프로파일 로드"""
        if self.profile_path.exists():
            try:
                with open(self.profile_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"⚠️  Failed to load profile: {e}")
                return self._empty_profile()
        return self._empty_profile()

    def _empty_profile(self) -> Dict:
        """빈 프로파일 생성"""
        return {
            "metadata": {
                "profile_version": "1.0",
                "code_version": self._get_code_version(),
                "last_updated": None,
                "total_runs": 0,
                "successful_runs": 0,
                "failed_runs": 0
            },
            "runs": [],
            "recommendations": {}
        }

    def save_profile(self, profile: Dict):
        """프로파일 저장"""
        profile["metadata"]["last_updated"] = datetime.now().isoformat()
        profile["metadata"]["code_version"] = self._get_code_version()

        try:
            with open(self.profile_path, 'w') as f:
                json.dump(profile, f, indent=2)
        except Exception as e:
            self.logger.error(f"❌ Failed to save profile: {e}")

    def _get_code_version(self) -> str:
        """현재 코드 버전 반환 (Git commit hash)"""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', '--short', 'HEAD'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return f"git:{result.stdout.strip()}"
        except:
            pass
        return "unknown"

    def compute_config_hash(self, config: Dict) -> str:
        """
        설정의 고유 해시 생성 (유사 실행 비교용)
        메모리에 영향을 주는 주요 설정만 해싱
        """
        eval_config = config.get('EVALUATION', {})
        ml_config = config.get('ML', {})

        relevant_config = {
            'train_start_year': eval_config.get('TRAIN_START_YEAR'),
            'num_periods': len(eval_config.get('PERIODS', [])),
            'use_sector': ml_config.get('USE_SECTOR_MODEL') == 'Y',
            'use_classifier': ml_config.get('USE_CLASSIFIER') == 'Y',
            'use_optuna': ml_config.get('USE_OPTUNA') == 'Y',
            'rebalance_period': eval_config.get('REBALANCE_PERIOD', 3)
        }

        config_str = json.dumps(relevant_config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:12]

    def is_profile_outdated(self, profile: Dict) -> bool:
        """코드 버전이 크게 변경되었는지 확인"""
        current_version = self._get_code_version()
        profile_version = profile['metadata'].get('code_version', 'unknown')

        if current_version == 'unknown' or profile_version == 'unknown':
            return False  # 버전 확인 불가 시 무시

        if current_version != profile_version:
            self.logger.warning(f"⚠️  Code version changed:")
            self.logger.warning(f"   Profile: {profile_version}")
            self.logger.warning(f"   Current: {current_version}")
            self.logger.warning(f"   Using conservative settings")
            return True

        return False

    def check_system_compatibility(self, profile: Dict, config_hash: str) -> bool:
        """
        시스템 환경이 크게 변경되었는지 확인
        변경되었으면 프로파일 무효화
        """
        if config_hash not in profile['recommendations']:
            return True  # 새 설정이면 OK

        # 성공한 원본 실행 찾기
        original_runs = [
            r for r in profile['runs']
            if r['config_hash'] == config_hash and r['execution']['success']
        ]

        if not original_runs:
            return True

        original = original_runs[-1]  # 가장 최근 성공 실행
        current_memory = psutil.virtual_memory().total / (1024**3)
        original_memory = original['system']['total_memory_gb']

        # 메모리 10% 이상 감소 시
        if current_memory < original_memory * 0.9:
            self.logger.warning(f"⚠️  System memory decreased significantly:")
            self.logger.warning(f"   Original: {original_memory:.1f} GB")
            self.logger.warning(f"   Current:  {current_memory:.1f} GB")
            self.logger.warning(f"   Profile may be invalid")
            return False

        return True

    def get_optimal_workers(self, config: Dict) -> Dict:
        """
        과거 실행 데이터를 기반으로 최적 worker 수 계산

        Returns:
            {
                'recommended_workers': int,
                'confidence': str,  # 'high', 'medium', 'low', 'none', 'recalibrating'
                'reason': str,
                'estimated_memory_per_worker_gb': float
            }
        """
        profile = self.load_profile()
        config_hash = self.compute_config_hash(config)

        # 1단계: 코드 버전 확인
        if self.is_profile_outdated(profile):
            return {
                'recommended_workers': 1,
                'confidence': 'none',
                'reason': 'Code version changed, using conservative settings',
                'estimated_memory_per_worker_gb': psutil.virtual_memory().total / (1024**3) * 0.7
            }

        # 2단계: 시스템 환경 확인
        if not self.check_system_compatibility(profile, config_hash):
            return {
                'recommended_workers': 1,
                'confidence': 'none',
                'reason': 'System memory decreased, using conservative settings',
                'estimated_memory_per_worker_gb': psutil.virtual_memory().total / (1024**3) * 0.7
            }

        # 3단계: 재보정 상태 확인
        if config_hash in profile['recommendations']:
            rec = profile['recommendations'][config_hash]

            if rec.get('confidence') == 'recalibrating':
                self.logger.warning("🔄 Configuration in recalibration mode (previous OOM)")
                self.logger.warning("   Using 1 worker to safely rebuild profile")
                return {
                    'recommended_workers': 1,
                    'confidence': 'recalibrating',
                    'reason': rec.get('notes', 'Previous OOM, needs recalibration'),
                    'estimated_memory_per_worker_gb': rec.get('safe_memory_per_worker_gb', 0)
                }

        # 4단계: 정상 추천 사용
        if config_hash in profile['recommendations']:
            rec = profile['recommendations'][config_hash]
            self.logger.info(f"📊 Found historical data for this configuration")
            self.logger.info(f"   Based on {rec['based_on_runs']} previous runs")
            self.logger.info(f"   Confidence: {rec['confidence']}")

            return {
                'recommended_workers': rec['recommended_workers'],
                'confidence': rec['confidence'],
                'reason': f"Based on {rec['based_on_runs']} successful runs",
                'estimated_memory_per_worker_gb': rec['safe_memory_per_worker_gb']
            }

        # 5단계: 유사한 설정의 실행 찾기
        similar_runs = [
            run for run in profile['runs']
            if run['config_hash'] == config_hash and run['execution']['success']
        ]

        if similar_runs:
            # 성공한 실행들의 평균 메모리 사용량
            avg_peak = sum(r['memory_usage']['peak_per_worker_gb']
                          for r in similar_runs) / len(similar_runs)
            safe_memory = avg_peak * 1.2  # 20% 안전 마진

            system_memory = psutil.virtual_memory().total / (1024**3)
            recommended = max(1, int(system_memory * 0.8 / safe_memory))

            self.logger.info(f"📊 Found {len(similar_runs)} similar runs")
            self.logger.info(f"   Average peak memory: {avg_peak:.1f} GB/worker")

            return {
                'recommended_workers': recommended,
                'confidence': 'medium',
                'reason': f"Based on {len(similar_runs)} similar runs",
                'estimated_memory_per_worker_gb': safe_memory
            }

        # 6단계: 전체 실행 데이터에서 보수적으로 추정
        if profile['runs']:
            all_peaks = [
                r['memory_usage']['peak_per_worker_gb']
                for r in profile['runs']
                if r['execution']['success'] and 'memory_usage' in r
            ]

            if all_peaks:
                max_peak = max(all_peaks)
                safe_memory = max_peak * 1.5  # 50% 안전 마진 (보수적)

                system_memory = psutil.virtual_memory().total / (1024**3)
                recommended = max(1, int(system_memory * 0.7 / safe_memory))

                self.logger.warning(f"⚠️  No exact match, using conservative estimate")
                self.logger.warning(f"   Based on historical max: {max_peak:.1f} GB/worker")

                return {
                    'recommended_workers': recommended,
                    'confidence': 'low',
                    'reason': "Conservative estimate from all historical runs",
                    'estimated_memory_per_worker_gb': safe_memory
                }

        # 7단계: 첫 실행 - 매우 보수적
        system_memory = psutil.virtual_memory().total / (1024**3)

        self.logger.warning("⚠️  No historical data available")
        self.logger.warning("   Using conservative default: 1 worker")

        return {
            'recommended_workers': 1,
            'confidence': 'none',
            'reason': "First run - no historical data",
            'estimated_memory_per_worker_gb': system_memory * 0.7
        }

    def start_run(self, config: Dict, num_workers: int):
        """실행 시작 - 메모리 추적 시작"""
        eval_config = config.get('EVALUATION', {})
        ml_config = config.get('ML', {})

        self.current_run = {
            "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "timestamp": datetime.now().isoformat(),
            "config_signature": {
                "train_start_year": eval_config.get('TRAIN_START_YEAR'),
                "num_periods": len(eval_config.get('PERIODS', [])),
                "use_sector_model": ml_config.get('USE_SECTOR_MODEL') == 'Y',
                "use_classifier": ml_config.get('USE_CLASSIFIER') == 'Y'
            },
            "config_hash": self.compute_config_hash(config),
            "memory_usage": {},
            "system": {
                "total_memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                "available_at_start_gb": round(psutil.virtual_memory().available / (1024**3), 2),
                "cpu_cores": psutil.cpu_count(),
                "gpu_available": self._check_gpu()
            },
            "execution": {
                "num_workers": num_workers,
                "runtime_minutes": 0,
                "success": None,
                "error": None
            },
            "period_details": []
        }

        self.memory_samples = []
        self.period_stats = []
        self.start_time = datetime.now()

        self.logger.info(f"🔍 Memory profiling started (Run ID: {self.current_run['run_id']})")

    def record_period_memory(self, period_idx: int, train_samples: int):
        """각 period 실행 중 메모리 샘플링"""
        import os
        try:
            process = psutil.Process(os.getpid())
            memory_gb = process.memory_info().rss / (1024**3)

            self.memory_samples.append(memory_gb)

            period_stat = {
                "period": period_idx,
                "memory_peak_gb": round(memory_gb, 2),
                "train_samples": train_samples,
                "timestamp": datetime.now().isoformat()
            }
            self.period_stats.append(period_stat)
        except Exception as e:
            self.logger.debug(f"Memory sampling failed: {e}")

    def end_run(self, success: bool, error: Optional[str] = None):
        """실행 종료 - 결과 저장 및 프로파일 업데이트"""
        if not self.current_run:
            return

        runtime = (datetime.now() - self.start_time).total_seconds() / 60

        # 메모리 통계 계산
        if self.memory_samples:
            self.current_run["memory_usage"] = {
                "avg_per_worker_gb": round(float(np.mean(self.memory_samples)), 2),
                "peak_per_worker_gb": round(float(np.max(self.memory_samples)), 2),
                "min_per_worker_gb": round(float(np.min(self.memory_samples)), 2),
                "std_per_worker_gb": round(float(np.std(self.memory_samples)), 2)
            }

        self.current_run["execution"]["runtime_minutes"] = round(runtime, 2)
        self.current_run["execution"]["success"] = success
        self.current_run["execution"]["error"] = str(error) if error else None
        self.current_run["period_details"] = self.period_stats

        # 프로파일 업데이트
        profile = self.load_profile()

        # OOM 감지 및 무효화 처리
        is_oom = not success and error and ('memory' in error.lower() or 'oom' in error.lower())

        if is_oom:
            self._handle_oom(profile)

        # 실행 기록 추가
        profile["runs"].append(self.current_run)

        # 메타데이터 업데이트
        profile["metadata"]["total_runs"] += 1
        if success:
            profile["metadata"]["successful_runs"] += 1
            # 추천 데이터 업데이트
            self._update_recommendations(profile)
        else:
            profile["metadata"]["failed_runs"] += 1

        # 실패한 run 정리
        self._cleanup_failed_runs(profile)

        # 저장
        self.save_profile(profile)

        if success:
            self.logger.info(f"✅ Memory profile saved successfully")
            if self.memory_samples:
                self.logger.info(f"   Peak memory: {self.current_run['memory_usage']['peak_per_worker_gb']:.2f} GB/worker")
            self.logger.info(f"   Runtime: {runtime:.1f} minutes")
        else:
            self.logger.warning(f"⚠️  Failed run recorded for future reference")

    def _handle_oom(self, profile: Dict):
        """OOM 발생 시 프로파일 무효화"""
        config_hash = self.current_run['config_hash']

        if config_hash in profile['recommendations']:
            # 기존 추천이 있었는데 OOM 발생 = 프로파일 무효화

            self.logger.error("🚨 OOM detected despite using profile recommendation!")
            self.logger.error("   This indicates:")
            self.logger.error("   - Model/code has changed significantly")
            self.logger.error("   - System memory has decreased")
            self.logger.error("   - Profile is no longer valid")
            self.logger.error("")
            self.logger.error("🔄 INVALIDATING profile for this configuration")
            self.logger.error("   Next run will start conservatively with 1 worker")

            # 무효화: 재보정 상태로 변경
            profile['recommendations'][config_hash] = {
                "recommended_workers": 1,  # 강제로 1로 재설정
                "confidence": "recalibrating",  # 특수 상태
                "safe_memory_per_worker_gb": None,
                "based_on_runs": 0,
                "last_status": "invalidated_due_to_oom",
                "invalidated_at": datetime.now().isoformat(),
                "notes": f"OOM occurred with {self.current_run['execution']['num_workers']} workers. Needs recalibration."
            }

    def _update_recommendations(self, profile: Dict):
        """성공한 실행 기반으로 추천 데이터 갱신"""
        config_hash = self.current_run['config_hash']

        # 같은 설정의 성공한 실행들만 필터링
        successful_runs = [
            run for run in profile['runs']
            if run['config_hash'] == config_hash
            and run['execution']['success']
            and 'memory_usage' in run
        ]

        if not successful_runs:
            return

        # 평균 peak 메모리 계산
        avg_peak = sum(r['memory_usage']['peak_per_worker_gb']
                      for r in successful_runs) / len(successful_runs)

        safe_memory = avg_peak * 1.2  # 20% 안전 마진
        system_memory = psutil.virtual_memory().total / (1024**3)
        recommended_workers = max(1, int(system_memory * 0.8 / safe_memory))

        # Confidence 계산
        if len(successful_runs) >= 3:
            confidence = "high"
        elif len(successful_runs) >= 1:
            confidence = "medium"
        else:
            confidence = "low"

        # 실패한 실행 체크
        failed_runs = [
            run for run in profile['runs']
            if run['config_hash'] == config_hash
            and not run['execution']['success']
        ]

        notes = []
        if failed_runs:
            notes.append(f"{len(failed_runs)} failed run(s) recorded")
            # 실패한 worker 수보다 적게 추천
            failed_workers = [r['execution']['num_workers'] for r in failed_runs]
            max_failed_workers = max(failed_workers)
            if recommended_workers >= max_failed_workers:
                recommended_workers = max(1, max_failed_workers - 1)
                notes.append(f"Reduced due to OOM with {max_failed_workers} workers")

        profile['recommendations'][config_hash] = {
            "safe_memory_per_worker_gb": round(safe_memory, 2),
            "recommended_workers": recommended_workers,
            "confidence": confidence,
            "based_on_runs": len(successful_runs),
            "last_successful_run": successful_runs[-1]['run_id'],
            "notes": "; ".join(notes) if notes else None
        }

    def _cleanup_failed_runs(self, profile: Dict, keep_recent: int = 5):
        """
        실패한 실행 정리
        - 최근 N개만 유지
        - 성공한 실행은 모두 유지
        """
        successful = [r for r in profile['runs'] if r['execution']['success']]
        failed = [r for r in profile['runs'] if not r['execution']['success']]

        # 실패한 실행은 최근 N개만
        failed_sorted = sorted(failed, key=lambda x: x['timestamp'], reverse=True)
        failed_to_keep = failed_sorted[:keep_recent]

        profile['runs'] = successful + failed_to_keep

        if len(failed) > keep_recent:
            self.logger.info(f"🧹 Cleaned up {len(failed) - keep_recent} old failed runs")

    def _check_gpu(self) -> bool:
        """GPU 사용 가능 여부 확인"""
        try:
            import cupy
            return True
        except:
            return False
