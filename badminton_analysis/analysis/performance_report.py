"""Deprecated compatibility import for the business-owned report builder.

GPU analysis code must not generate reports. Business deployments should
import ``business_gateway.report.performance`` directly.
"""

from business_gateway.report.performance import *  # noqa: F401,F403
