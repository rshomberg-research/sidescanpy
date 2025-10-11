from dataclasses import dataclass
from datetime import datetime
import numpy as np
from typing import List, Optional

@dataclass
class SidescanPing:
    time: datetime
    intensity: np.ndarray
    slant_range: float
    sample_interval: float
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude: Optional[float] = None
    pitch: Optional[float] = None
    roll: Optional[float] = None
    yaw: Optional[float] = None

@dataclass
class SidescanPingSeries:
    pings: List[SidescanPing]
