"""Business-side athlete report generation."""

from .performance import build_report_evidence, generate_performance_report

__all__ = ["build_report_evidence", "generate_performance_report"]
