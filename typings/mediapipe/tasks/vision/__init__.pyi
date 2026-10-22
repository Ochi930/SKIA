from enum import Enum
from typing import Callable, Optional

from mediapipe import Image
from mediapipe.tasks import BaseOptions


class RunningMode(Enum):
    IMAGE: RunningMode
    VIDEO: RunningMode
    LIVE_STREAM: RunningMode


class NormalizedLandmark:
    x: float
    y: float
    z: float


class HandLandmarkerResult:
    hand_landmarks: list[list[NormalizedLandmark]]


class HandLandmarkerOptions:
    base_options: BaseOptions
    running_mode: RunningMode
    num_hands: int
    min_hand_detection_confidence: float
    min_hand_presence_confidence: float
    min_tracking_confidence: float
    result_callback: Optional[
        Callable[[HandLandmarkerResult, Image, int], None]
    ]

    def __init__(
        self,
        base_options: BaseOptions,
        running_mode: RunningMode = ...,
        num_hands: int = ...,
        min_hand_detection_confidence: float = ...,
        min_hand_presence_confidence: float = ...,
        min_tracking_confidence: float = ...,
        result_callback: Optional[
            Callable[[HandLandmarkerResult, Image, int], None]
        ] = ...,
    ) -> None: ...


class HandLandmarker:
    @classmethod
    def create_from_options(
        cls, options: HandLandmarkerOptions
    ) -> HandLandmarker: ...

    def __enter__(self) -> HandLandmarker: ...

    def __exit__(
        self, exc_type: object, exc_value: object, traceback: object
    ) -> None: ...

    def detect_for_video(
        self,
        image: Image,
        timestamp_ms: int,
        image_processing_options: object = ...,
    ) -> HandLandmarkerResult: ...
