'''
Main entry point for users

2026 rshomberg@uri.edu
'''

import logging 
from pathlib import Path 

from sidescanpy.core.survey import MultiFreqSideScanSurvey

def ingest_jsf(jsf_file: Path) -> MultiFreqSideScanSurvey:
    ...

def batch_ingest(directory: Path) -> MultiFreqSideScanSurvey:
    ...

    