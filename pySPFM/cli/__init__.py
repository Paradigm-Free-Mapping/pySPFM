"""Command-line interface for pySPFM.

Entry Points
------------
main : Main pySPFM CLI with subcommands (sparse, lowrank, stability)
auc_to_estimates_cli : AUC to estimates workflow
"""

from pySPFM.cli.auc_to_estimates import auc_to_estimates as auc_to_estimates_cli
from pySPFM.cli.run import main

__all__ = ["main", "auc_to_estimates_cli"]
