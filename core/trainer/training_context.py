from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from core.trainer.train_state import TrainingState
from core.utils.distributed import DistContext
from core.trainer.optimization_helper import OptimizationHelper
from core.trainer.metric_helper import MetricHelper
from core.trainer.ema import EMAHelper
from core.act.act_container import ACTContainerManager
from core.interfaces import LossProtocol, HaltingController


@dataclass
class TrainingContext:

    train_state: TrainingState
    dist: DistContext

    optimizer: OptimizationHelper
    metric_helper: MetricHelper
    act_manager: ACTContainerManager
    criterion: LossProtocol
    halting_ctrl: HaltingController

    ema: Optional[EMAHelper] = None
    checkpoint_manager: Any = None