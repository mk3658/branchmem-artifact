"""BranchGuard: calibration-certified routing for two-branch memory conflicts.

Introduced in the revised manuscript. `core` provides the Hoeffding/
Clopper-Pearson certified router; `contextual` adds context-conditional
certification with bounded-drift thresholds and an anytime revocation monitor.
"""
from branchmem.branchguard.core import *  # noqa: F401,F403
