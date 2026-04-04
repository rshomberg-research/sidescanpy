# TODO: document. this is in fact used

import logging

from .messages import JSFMessageBody
from .messages import *

def split_messages(messages: list[JSFMessageBody]) -> dict[str, list[JSFMessageBody]]:
    """Stub: classify messages by type."""
    groups = {
        "SonarDataMessages":                [], 
        "LegacySideScanDataMessages":       [], 
        "SystemInformationMessages":        [], 
        "FileTimeStampMessages":            [],
        "FilePaddingMessages":              [],
        "NMEAStringMessages":               [],
        "NMEADBTMessages":                  [],
        "NMEARMCMessages":                  [],
        "NMEADPTMessages":                  [],
        "PitchRollDataMessages":            [],
        "PressureSensorReadingMessage":     [],
        "DopplerVelocityLogDataMessages":   [],
        "SituationMessages":                [],
        "SituationComprehensiveMessages":   [],
        "CableCounterDataMessages":         [],
        "KilometerOfPipeDataMessages":      [],
        "ContainerTimeStampMessages":       [],
        }
    
    for m in messages:
        cls = str(type(m)).split('.')[-1][:-2]+'s'
        groups[cls].append(m)

    return groups
