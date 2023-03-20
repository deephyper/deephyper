from deephyper.stopper._stopper import Stopper


class IdleStopper(Stopper):
    """Idle stopper which nevers stops the evaluation unless a failure is observed.

    .. list-table::
        :widths: 25 25 25
        :header-rows: 1

        * - Single-Objective
          - Multi-Objectives
          - Failures
        * - ✅
          - ❌
          - ✅
    """
