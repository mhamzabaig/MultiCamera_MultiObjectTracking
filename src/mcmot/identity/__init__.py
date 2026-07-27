"""Identity resolution: enrolment, gallery search, thresholds, and voting.

The only module the tracking pipeline talks to about *who* someone is. Trackers never
reach the database or the vector index themselves — they hand this service a crop and
receive a verdict.
"""

from __future__ import annotations

from mcmot.identity.service import IdentityService
from mcmot.identity.voting import TrackVote, TrackVoter

__all__ = ["IdentityService", "TrackVote", "TrackVoter"]
