import os
from typing import Any, Callable, Dict, Optional

import pytorch_lightning as pl
from pytorchvideo.data.clip_sampling import RandomClipSampler, UniformClipSampler
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample,
)
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
    RandomCropVideo,
    RandomHorizontalFlipVideo,
)

from sign_language_dataset import sign_language_dataset


class SignLanguageDataModule(pl.LightningDataModule):
    """
    수어 데이터를 위한 PyTorch Lightning 데이터 모듈
    """
    
    def __init__(
        self,
        data_root: str,
        train_data_root: Optional[str] = None,
        val_data_root: Optional[str] = None,
        test_data_root: Optional[str] = None,
        batch_size: int = 8,
        num_workers: int = 4,
        clip_duration: float = 2.0,
        num_frames: int = 16,
        crop_size: int = 224,
        mean: tuple = (0.45, 0.45, 0.45),
        std: tuple = (0.225, 0.225, 0.225),
        pin_memory: bool = True,
    ):
        """
        Args:
            data_root: 기본 데이터 루트 경로
            train_data_root: 학습 데이터 경로 (None이면 data_root 사용)
            val_data_root: 검증 데이터 경로 (None이면 data_root 사용)
            test_data_root: 테스트 데이터 경로 (None이면 data_root 사용)
            batch_size: 배치 크기
            num_workers: 데이터 로더 워커 수
            clip_duration: 클립 지속 시간 (초)
            num_frames: 프레임 수
            crop_size: 크롭 크기
            mean: 정규화 평균값
            std: 정규화 표준편차
            pin_memory: 메모리 고정 여부
        """
        super().__init__()
        self.data_root = data_root
        self.train_data_root = train_data_root or data_root
        self.val_data_root = val_data_root or data_root
        self.test_data_root = test_data_root or data_root
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.clip_duration = clip_duration
        self.num_frames = num_frames
        self.crop_size = crop_size
        self.mean = mean
        self.std = std
        self.pin_memory = pin_memory
        
        # 데이터셋 인스턴스들
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # 클래스 정보
        self.num_classes = None
        self.class_to_idx = None
        self.idx_to_class = None
    
    def _get_train_transforms(self) -> Callable:
        """학습용 데이터 변환 함수"""
        return Compose([
            ApplyTransformToKey(
                key="video",
                transform=Compose([
                    UniformTemporalSubsample(self.num_frames),
                    Lambda(lambda x: x / 255.0),
                    NormalizeVideo(self.mean, self.std),
                    RandomShortSideScale(min_size=256, max_size=320),
                    RandomCropVideo(self.crop_size),
                    RandomHorizontalFlipVideo(p=0.5),
                ]),
            ),
            # Label is already an integer, no transform needed
            RemoveKey("audio"),
        ])
    
    def _get_val_transforms(self) -> Callable:
        """검증/테스트용 데이터 변환 함수"""
        return Compose([
            ApplyTransformToKey(
                key="video",
                transform=Compose([
                    UniformTemporalSubsample(self.num_frames),
                    Lambda(lambda x: x / 255.0),
                    NormalizeVideo(self.mean, self.std),
                    ShortSideScale(size=256),
                    CenterCropVideo(self.crop_size),
                ]),
            ),
            # Label is already an integer, no transform needed
            RemoveKey("audio"),
        ])
    
    def setup(self, stage: Optional[str] = None):
        """데이터셋 설정"""
        if stage == "fit" or stage is None:
            # 학습 데이터셋
            self.train_dataset = sign_language_dataset(
                data_root=self.train_data_root,
                clip_sampler=RandomClipSampler(clip_duration=self.clip_duration),
                transform=self._get_train_transforms(),
                decode_audio=False,
            )
            
            # 검증 데이터셋
            self.val_dataset = sign_language_dataset(
                data_root=self.val_data_root,
                clip_sampler=UniformClipSampler(clip_duration=self.clip_duration),
                transform=self._get_val_transforms(),
                decode_audio=False,
            )
            
            # 클래스 정보 설정 (학습 데이터셋 기준)
            self.num_classes = self.train_dataset.num_classes
            self.class_to_idx = self.train_dataset.class_to_idx
            self.idx_to_class = self.train_dataset.idx_to_class
            
            print(f"Training dataset: {len(self.train_dataset)} videos")
            print(f"Validation dataset: {len(self.val_dataset)} videos")
            print(f"Number of classes: {self.num_classes}")
            print(f"Classes: {list(self.class_to_idx.keys())}")
        
        if stage == "test" or stage is None:
            # 테스트 데이터셋
            self.test_dataset = sign_language_dataset(
                data_root=self.test_data_root,
                clip_sampler=UniformClipSampler(clip_duration=self.clip_duration),
                transform=self._get_val_transforms(),
                decode_audio=False,
            )
            
            if self.num_classes is None:
                self.num_classes = self.test_dataset.num_classes
                self.class_to_idx = self.test_dataset.class_to_idx
                self.idx_to_class = self.test_dataset.idx_to_class
            
            print(f"Test dataset: {len(self.test_dataset)} videos")
    
    def train_dataloader(self):
        """학습 데이터 로더"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )
    
    def val_dataloader(self):
        """검증 데이터 로더"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )
    
    def test_dataloader(self):
        """테스트 데이터 로더"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        ) 