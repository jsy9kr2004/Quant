"""
Reporting module for integrated ML backtest reports.

This module provides tools for generating comprehensive Excel reports
that combine regressor predictions with backtest performance results.
"""

from .integrated_report import IntegratedReportWriter

__all__ = ['IntegratedReportWriter']
