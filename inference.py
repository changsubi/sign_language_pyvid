#!/usr/bin/env python3
"""
수어 분류 모델 추론 스크립트

사용 예시:
python inference.py --checkpoint ./outputs/sign_language_classification/checkpoints/best.ckpt --video_path ./test_video.mp4
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from pytorchvideo.data.clip_sampling import UniformClipSampler
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
)
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import CenterCropVideo, NormalizeVideo

from sign_language_model import SignLanguageClassifier


class SignLanguageInference:
    """수어 분류 추론 클래스"""
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "auto",
        num_frames: int = 16,
        crop_size: int = 224,
        mean: tuple = (0.45, 0.45, 0.45),
        std: tuple = (0.225, 0.225, 0.225),
    ):
        """
        Args:
            checkpoint_path: 체크포인트 파일 경로
            device: 디바이스 ('auto', 'cpu', 'cuda')
            num_frames: 샘플링할 프레임 수
            crop_size: 이미지 크롭 크기
            mean: 정규화 평균값
            std: 정규화 표준편차
        """
        self.checkpoint_path = checkpoint_path
        self.num_frames = num_frames
        self.crop_size = crop_size
        self.mean = mean
        self.std = std
        
        # 디바이스 설정
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # 모델 로드
        self.model = self._load_model()
        self.model.eval()
        
        # Efficient 모델인 경우 배포 모드로 변환
        if hasattr(self.model, 'is_efficient_model') and self.model.is_efficient_model:
            print("🔄 Efficient 모델을 배포 모드로 변환 중...")
            self.model.convert_to_deployment_mode()
        
        # 변환 함수 설정
        self.transform = self._get_transforms()
        
        print(f"✅ 모델 로드 완료")
        print(f"🔧 디바이스: {self.device}")
        print(f"📊 클래스 개수: {self.model.num_classes}")
    
    def _load_model(self) -> SignLanguageClassifier:
        """체크포인트에서 모델 로드"""
        print(f"📥 체크포인트 로드 중: {self.checkpoint_path}")
        
        model = SignLanguageClassifier.load_from_checkpoint(
            self.checkpoint_path,
            map_location=self.device
        )
        model.to(self.device)
        
        return model
    
    def _get_transforms(self):
        """추론용 데이터 변환 함수"""
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
        ])
    
    def predict_video(
        self,
        video_path: str,
        clip_duration: float = 2.0,
        start_time: float = 0.0,
        return_probabilities: bool = False,
    ) -> dict:
        """
        비디오 파일에 대한 수어 분류 예측
        
        Args:
            video_path: 비디오 파일 경로
            clip_duration: 클립 지속 시간 (초)
            start_time: 시작 시간 (초)
            return_probabilities: 확률값 반환 여부
            
        Returns:
            예측 결과 딕셔너리
        """
        print(f"🎬 비디오 처리 중: {video_path}")
        
        # 비디오 로드
        video = EncodedVideo.from_path(video_path)
        
        # 클립 샘플링
        end_time = start_time + clip_duration
        if end_time > video.duration:
            end_time = video.duration
            start_time = max(0, end_time - clip_duration)
        
        # 비디오 클립 추출
        video_data = video.get_clip(start_sec=start_time, end_sec=end_time)
        
        # 변환 적용
        video_data = self.transform(video_data)
        
        # 배치 차원 추가
        video_tensor = video_data["video"].unsqueeze(0).to(self.device)
        
        # 추론
        with torch.no_grad():
            logits = self.model(video_tensor)
            probabilities = F.softmax(logits, dim=1)
            predicted_idx = torch.argmax(logits, dim=1).item()
            confidence = probabilities[0, predicted_idx].item()
        
        # 결과 생성
        result = {
            "video_path": video_path,
            "start_time": start_time,
            "end_time": end_time,
            "predicted_class_idx": predicted_idx,
            "confidence": confidence,
            "clip_duration": end_time - start_time,
        }
        
        # 클래스 이름 추가 (가능한 경우)
        if hasattr(self.model, 'hparams') and hasattr(self.model.hparams, 'class_names'):
            class_names = self.model.hparams.class_names
            if predicted_idx < len(class_names):
                result["predicted_class_name"] = class_names[predicted_idx]
        
        # 확률값 추가 (요청된 경우)
        if return_probabilities:
            result["probabilities"] = probabilities[0].cpu().numpy().tolist()
        
        return result
    
    def predict_multiple_clips(
        self,
        video_path: str,
        clip_duration: float = 2.0,
        stride: float = 1.0,
        return_probabilities: bool = False,
    ) -> list:
        """
        비디오에서 여러 클립에 대한 예측
        
        Args:
            video_path: 비디오 파일 경로
            clip_duration: 클립 지속 시간 (초)
            stride: 클립 간격 (초)
            return_probabilities: 확률값 반환 여부
            
        Returns:
            예측 결과 리스트
        """
        print(f"🎬 비디오 다중 클립 처리 중: {video_path}")
        
        # 비디오 로드
        video = EncodedVideo.from_path(video_path)
        
        results = []
        start_time = 0.0
        
        while start_time + clip_duration <= video.duration:
            result = self.predict_video(
                video_path,
                clip_duration=clip_duration,
                start_time=start_time,
                return_probabilities=return_probabilities,
            )
            results.append(result)
            start_time += stride
        
        return results


def parse_args():
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(description="수어 분류 모델 추론")
    
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="모델 체크포인트 파일 경로")
    parser.add_argument("--video_path", type=str, required=True,
                       help="추론할 비디오 파일 경로")
    parser.add_argument("--output_path", type=str, default=None,
                       help="결과 저장 경로 (JSON 파일)")
    
    # 추론 설정
    parser.add_argument("--clip_duration", type=float, default=2.0,
                       help="클립 지속 시간 (초)")
    parser.add_argument("--start_time", type=float, default=0.0,
                       help="시작 시간 (초)")
    parser.add_argument("--multiple_clips", action="store_true",
                       help="여러 클립에 대한 예측 수행")
    parser.add_argument("--stride", type=float, default=1.0,
                       help="클립 간격 (초, multiple_clips 모드)")
    parser.add_argument("--return_probabilities", action="store_true",
                       help="확률값 포함하여 반환")
    
    # 모델 설정
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cpu", "cuda"],
                       help="디바이스")
    parser.add_argument("--num_frames", type=int, default=16,
                       help="샘플링할 프레임 수")
    parser.add_argument("--crop_size", type=int, default=224,
                       help="이미지 크롭 크기")
    
    return parser.parse_args()


def main():
    """메인 함수"""
    args = parse_args()
    
    print("🚀 수어 분류 추론 시작")
    print(f"📁 체크포인트: {args.checkpoint}")
    print(f"🎬 비디오: {args.video_path}")
    
    # 추론기 초기화
    inference = SignLanguageInference(
        checkpoint_path=args.checkpoint,
        device=args.device,
        num_frames=args.num_frames,
        crop_size=args.crop_size,
    )
    
    # 추론 수행
    if args.multiple_clips:
        print(f"🔄 다중 클립 모드 (stride: {args.stride}초)")
        results = inference.predict_multiple_clips(
            video_path=args.video_path,
            clip_duration=args.clip_duration,
            stride=args.stride,
            return_probabilities=args.return_probabilities,
        )
        
        print(f"\n📊 총 {len(results)}개 클립 처리 완료")
        for i, result in enumerate(results):
            print(f"클립 {i+1}: {result['start_time']:.1f}s-{result['end_time']:.1f}s")
            print(f"   예측: 클래스 {result['predicted_class_idx']} (신뢰도: {result['confidence']:.4f})")
            if 'predicted_class_name' in result:
                print(f"   클래스명: {result['predicted_class_name']}")
    else:
        print(f"🎯 단일 클립 모드")
        result = inference.predict_video(
            video_path=args.video_path,
            clip_duration=args.clip_duration,
            start_time=args.start_time,
            return_probabilities=args.return_probabilities,
        )
        
        print(f"\n📊 예측 결과:")
        print(f"   클립 구간: {result['start_time']:.1f}s - {result['end_time']:.1f}s")
        print(f"   예측 클래스: {result['predicted_class_idx']}")
        print(f"   신뢰도: {result['confidence']:.4f}")
        if 'predicted_class_name' in result:
            print(f"   클래스명: {result['predicted_class_name']}")
        
        results = result
    
    # 결과 저장
    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"💾 결과 저장: {output_path}")
    
    print("✅ 추론 완료!")


if __name__ == "__main__":
    main() 