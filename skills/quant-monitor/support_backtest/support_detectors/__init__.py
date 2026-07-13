"""Support level detector plugins."""
from support_detectors.base import BaseSupportDetector, combine_detections
from support_detectors.ma_support import MASupport
from support_detectors.trend_line import TrendLineSupport
# TODO: uncomment as each detector is created
# from support_detectors.bollinger import BollingerSupport
# from support_detectors.prior_low import PriorLowSupport
# from support_detectors.box_range import BoxRangeSupport
# from support_detectors.volume_cluster import VolumeClusterSupport
# from support_detectors.round_number import RoundNumberSupport

__all__ = [
    "BaseSupportDetector", "combine_detections",
    "MASupport", "TrendLineSupport",
    # "BollingerSupport", "PriorLowSupport",
    # "BoxRangeSupport", "VolumeClusterSupport",
    # "RoundNumberSupport",
]
