"""Per-lane measurement modules for research lanes 01-07.

Each module exports:
  - name() -> str   : the METRIC name for this lane
  - measure() -> float : the lane's primary metric value, or float('nan')
                          if the lane cannot be measured at this baseline
"""
