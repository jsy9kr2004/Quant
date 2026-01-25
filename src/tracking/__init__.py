"""Experiment tracking module for ML experiment management.

This package provides tools for:
1. Tracking experiments via Google Sheets/Gist
2. Monitoring model performance
3. Generating integrated reports

Main Components:
    SheetsTracker: Google Sheets based experiment tracker
    ConfigMasker: Configuration masking for safe sharing
    PerformanceMonitor: Model performance monitoring
    IntegratedReportWriter: Comprehensive Excel report generation

Example:
    Tracking experiment results::

        from src.tracking import SheetsTracker, IntegratedReportWriter

        # Upload experiment results
        tracker = SheetsTracker(config)
        tracker.upload_result(experiment_data)

        # Generate integrated report
        writer = IntegratedReportWriter(output_path)
        writer.write_report(results)
"""

from .sheets_tracker import SheetsTracker, upload_experiment_result
from .config_masker import ConfigMasker
from .performance_monitor import PerformanceMonitor
from .integrated_report import IntegratedReportWriter

__all__ = [
    'SheetsTracker',
    'upload_experiment_result',
    'ConfigMasker',
    'PerformanceMonitor',
    'IntegratedReportWriter',
]

__version__ = '1.0.0'
__author__ = 'Quant Trading Team'
__description__ = 'Experiment tracking and reporting'
