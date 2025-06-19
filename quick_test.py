#!/usr/bin/env python3
"""
수어 분류 시스템 빠른 테스트 스크립트

데이터셋 로딩, 모델 생성, 단일 배치 순전파를 테스트합니다.
실제 학습 전에 코드가 정상 동작하는지 확인하는 용도입니다.

사용 예시:
python quick_test.py --data_root ./datasest
"""

import argparse
import warnings
from pathlib import Path

import torch
import pytorch_lightning as pl

from sign_language_datamodule import SignLanguageDataModule
from sign_language_model import SignLanguageClassifier

warnings.filterwarnings("ignore", category=UserWarning)


def test_datamodule(data_root: str, batch_size: int = 2):
    """데이터 모듈 테스트"""
    print("🔍 데이터 모듈 테스트 중...")
    
    try:
        # 데이터 모듈 생성
        datamodule = SignLanguageDataModule(
            data_root=data_root,
            batch_size=batch_size,
            num_workers=0,  # 테스트에서는 멀티프로세싱 비활성화
            clip_duration=2.0,
            num_frames=8,  # 테스트에서는 프레임 수 줄임
            crop_size=224,
        )
        
        # 데이터 설정
        datamodule.setup("fit")
        
        print(f"   ✅ 클래스 개수: {datamodule.num_classes}")
        print(f"   ✅ 클래스 목록: {list(datamodule.class_to_idx.keys())}")
        print(f"   ✅ 학습 데이터: {len(datamodule.train_dataset)} videos")
        print(f"   ✅ 검증 데이터: {len(datamodule.val_dataset)} videos")
        
        # 데이터 로더 테스트
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        
        print(f"   ✅ 학습 배치 수: {len(train_loader)}")
        print(f"   ✅ 검증 배치 수: {len(val_loader)}")
        
        # 첫 번째 배치 로딩 테스트
        print("   🔄 첫 번째 배치 로딩 중...")
        batch = next(iter(train_loader))
        
        video = batch["video"]
        label = batch["label"]
        
        print(f"   ✅ 비디오 텐서 크기: {video.shape}")
        print(f"   ✅ 라벨 개수: {len(label)}")
        print(f"   ✅ 라벨 예시: {label}")
        
        return datamodule, batch
        
    except Exception as e:
        print(f"   ❌ 데이터 모듈 테스트 실패: {e}")
        raise


def test_model(num_classes: int, batch: dict, model_name: str = "x3d_s"):
    """모델 테스트"""
    print(f"\n🤖 모델 테스트 중... (모델: {model_name})")
    
    try:
        # 모델 생성
        model = SignLanguageClassifier(
            num_classes=num_classes,
            model_name=model_name,
            learning_rate=1e-3,
            pretrained=False,  # 테스트에서는 사전훈련 가중치 비활성화 (빠른 로딩)
        )
        
        print(f"   ✅ 모델 생성 완료")
        print(f"   ✅ 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
        
        # 모델을 평가 모드로 설정
        model.eval()
        
        # 순전파 테스트
        print("   🔄 순전파 테스트 중...")
        video = batch["video"]
        
        with torch.no_grad():
            logits = model(video)
        
        print(f"   ✅ 출력 크기: {logits.shape}")
        print(f"   ✅ 예상 출력 크기: ({video.shape[0]}, {num_classes})")
        
        # 예측 테스트
        predictions = torch.argmax(logits, dim=1)
        probabilities = torch.softmax(logits, dim=1)
        
        print(f"   ✅ 예측 인덱스: {predictions.tolist()}")
        print(f"   ✅ 최대 확률: {probabilities.max(dim=1)[0].tolist()}")
        
        return model
        
    except Exception as e:
        print(f"   ❌ 모델 테스트 실패: {e}")
        raise


def test_training_step(model, batch, datamodule):
    """학습 스텝 테스트"""
    print("\n🏃 학습 스텝 테스트 중...")
    
    try:
        model.train()
        
        # 더미 트레이너 설정 (datamodule 접근을 위해)
        class DummyTrainer:
            def __init__(self, datamodule):
                self.datamodule = datamodule
        
        model.trainer = DummyTrainer(datamodule)
        
        # 학습 스텝 실행
        loss = model.training_step(batch, 0)
        
        print(f"   ✅ 손실값: {loss.item():.4f}")
        print(f"   ✅ 손실 텐서 크기: {loss.shape}")
        
        # 검증 스텝 실행
        model.eval()
        with torch.no_grad():
            val_loss = model.validation_step(batch, 0)
        
        print(f"   ✅ 검증 손실값: {val_loss.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 학습 스텝 테스트 실패: {e}")
        raise


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="수어 분류 시스템 빠른 테스트")
    parser.add_argument("--data_root", type=str, required=True,
                       help="데이터 루트 폴더 경로")
    parser.add_argument("--model_name", type=str, default="x3d_s",
                       choices=["slow_r50", "x3d_s", "x3d_m", "mvit_base_16x4", "efficient_x3d_xs", "efficient_x3d_s"],
                       help="테스트할 모델")
    parser.add_argument("--batch_size", type=int, default=2,
                       help="테스트 배치 크기")
    
    args = parser.parse_args()
    
    print("🚀 수어 분류 시스템 빠른 테스트 시작")
    print(f"📁 데이터 경로: {args.data_root}")
    print(f"🎯 모델: {args.model_name}")
    print(f"📊 배치 크기: {args.batch_size}")
    
    # 데이터 존재 확인
    data_path = Path(args.data_root)
    if not data_path.exists():
        print(f"❌ 데이터 경로가 존재하지 않습니다: {args.data_root}")
        return
    
    video_dir = data_path / "video"
    label_dir = data_path / "label"
    
    if not video_dir.exists():
        print(f"❌ 비디오 폴더가 존재하지 않습니다: {video_dir}")
        return
        
    if not label_dir.exists():
        print(f"❌ 라벨 폴더가 존재하지 않습니다: {label_dir}")
        return
    
    print(f"✅ 데이터 경로 확인 완료")
    
    try:
        # 1. 데이터 모듈 테스트
        datamodule, batch = test_datamodule(args.data_root, args.batch_size)
        
        # 2. 모델 테스트
        model = test_model(datamodule.num_classes, batch, args.model_name)
        
        # 3. 학습 스텝 테스트
        test_training_step(model, batch, datamodule)
        
        print("\n🎉 모든 테스트 통과!")
        print("\n📋 요약:")
        print(f"   • 데이터셋: {len(datamodule.train_dataset)} 학습 + {len(datamodule.val_dataset)} 검증")
        print(f"   • 클래스 수: {datamodule.num_classes}")
        print(f"   • 모델: {args.model_name}")
        print(f"   • 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
        
        print("\n🚀 이제 본격적인 학습을 시작할 수 있습니다!")
        print("다음 명령어로 학습을 시작하세요:")
        print(f"python train_sign_language.py --data_root {args.data_root} --model_name {args.model_name} --batch_size {args.batch_size} --epochs 10 --fast_dev_run")
        
    except Exception as e:
        print(f"\n💥 테스트 실패: {e}")
        print("\n🔧 문제 해결 방법:")
        print("1. 데이터 형식이 올바른지 확인하세요")
        print("2. 필요한 패키지가 모두 설치되어 있는지 확인하세요 (pip install -r requirements.txt)")
        print("3. GPU 메모리가 부족하다면 --batch_size를 줄여보세요")
        print("4. 비디오 파일이 손상되지 않았는지 확인하세요")
        
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 