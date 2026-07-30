"""Application services: use cases that coordinate several components.

Distinct from ``identity/``, which owns one cohesive domain concern. A service
here answers a request that spans layers — "turn these uploaded frames into an
enrolled person" needs decoding, detection, cropping, and enrolment, and none of
that belongs in an HTTP handler.
"""

from __future__ import annotations

from mcmot.services.registration import RegistrationService

__all__ = ["RegistrationService"]
