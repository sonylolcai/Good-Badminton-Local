"""Deprecated compatibility import for business-owned movement metrics.

GPU analysis code must not call this module. It remains temporarily so older
integrations can import the public functions while deployments migrate to
``business_gateway.metrics.movement``.
"""

from business_gateway.metrics.movement import *  # noqa: F401,F403
