from dataclasses import dataclass
from typing import List
import pandas as pd
import numpy as np

# класс детектированного ббокса
@dataclass
class DetectedBox:
    x: int
    y: int
    w: int
    h: int
    score: float
    label: str

# класс извлеченной таблицы
@dataclass
class ExtractedTable:
    source: str
    dataframe: pd.DataFrame

# класс результата извлечения
@dataclass
class ExtractionResult:
    method: str
    tables: List[ExtractedTable]
    debug_images: List[np.ndarray]
