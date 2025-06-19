#!/usr/bin/env python3
"""
수어 분류 모델 학습 스크립트

사용 예시:
python train_sign_language.py --data_root ./datasest --model_name slow_r50 --batch_size 4 --epochs 50
"""

import argparse
import os
import warnings
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from sign_language_datamodule import SignLanguageDataModule
from sign_language_model import SignLanguageClassifier

# PyTorchVideo 관련 경고 억제
warnings.filterwarnings("ignore", category=UserWarning)


def parse_args():
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(description="수어 분류 모델 학습")
    
    # 데이터 관련 인수
    parser.add_argument("--data_root", type=str, required=True,
                       help="데이터 루트 폴더 경로 (video와 label 폴더 포함)")
    parser.add_argument("--train_data_root", type=str, default=None,
                       help="학습 데이터 경로 (기본값: data_root)")
    parser.add_argument("--val_data_root", type=str, default=None,
                       help="검증 데이터 경로 (기본값: data_root)")
    parser.add_argument("--test_data_root", type=str, default=None,
                       help="테스트 데이터 경로 (기본값: data_root)")
    
    # 모델 관련 인수
    parser.add_argument("--model_name", type=str, default="slow_r50",
                       choices=["slow_r50", "x3d_s", "x3d_m", "mvit_base_16x4", "efficient_x3d_xs", "efficient_x3d_s"],
                       help="사용할 모델 이름")
    parser.add_argument("--pretrained", action="store_true", default=True,
                       help="사전 훈련된 가중치 사용")
    parser.add_argument("--freeze_backbone", action="store_true",
                       help="백본 네트워크 고정")
    parser.add_argument("--dropout_rate", type=float, default=0.5,
                       help="드롭아웃 비율")
    
    # 학습 관련 인수
    parser.add_argument("--batch_size", type=int, default=8,
                       help="배치 크기")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                       help="학습률")
    parser.add_argument("--epochs", type=int, default=50,
                       help="학습 에포크 수")
    parser.add_argument("--optimizer", type=str, default="adam",
                       choices=["adam", "sgd"],
                       help="옵티마이저")
    parser.add_argument("--scheduler", type=str, default="cosine",
                       choices=["cosine", "step", "none"],
                       help="학습률 스케줄러")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                       help="가중치 감쇠")
    parser.add_argument("--label_smoothing", type=float, default=0.0,
                       help="라벨 스무딩")
    
    # 데이터 로더 관련 인수
    parser.add_argument("--num_workers", type=int, default=4,
                       help="데이터 로더 워커 수")
    parser.add_argument("--clip_duration", type=float, default=2.0,
                       help="비디오 클립 지속 시간 (초)")
    parser.add_argument("--num_frames", type=int, default=16,
                       help="샘플링할 프레임 수")
    parser.add_argument("--crop_size", type=int, default=224,
                       help="이미지 크롭 크기")
    
    # 학습 환경 관련 인수
    parser.add_argument("--gpus", type=int, default=1,
                       help="사용할 GPU 수")
    parser.add_argument("--accelerator", type=str, default="auto",
                       help="가속기 타입")
    parser.add_argument("--precision", type=str, default="16-mixed",
                       choices=["16-mixed", "32", "bf16-mixed"],
                       help="연산 정밀도")
    parser.add_argument("--seed", type=int, default=42,
                       help="랜덤 시드")
    
    # 체크포인트 및 로깅 관련 인수
    parser.add_argument("--output_dir", type=str, default="./outputs",
                       help="출력 폴더 경로")
    parser.add_argument("--experiment_name", type=str, default="sign_language_classification",
                       help="실험 이름")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                       help="체크포인트에서 재개")
    
    # 기타 옵션
    parser.add_argument("--test_only", action="store_true",
                       help="테스트만 수행")
    parser.add_argument("--fast_dev_run", action="store_true",
                       help="빠른 개발 실행 (디버깅용)")
    
    # Efficient 모델 관련 옵션
    parser.add_argument("--enable_efficient_deployment", action="store_true",
                       help="효율적 배포 모드 활성화 (efficient 모델 전용)")
    parser.add_argument("--export_mobile_model", action="store_true",
                       help="모바일 배포용 모델 내보내기")
    parser.add_argument("--quantize_model", action="store_true",
                       help="INT8 양자화 적용 (모바일 내보내기 시)")
    parser.add_argument("--mobile_model_path", type=str, default="./mobile_model.pt",
                       help="모바일 모델 저장 경로")
    
    return parser.parse_args()


def setup_callbacks(output_dir: str, monitor_metric: str = "val/accuracy"):
    """콜백 설정"""
    callbacks = [
        # 모델 체크포인트
        ModelCheckpoint(
            dirpath=os.path.join(output_dir, "checkpoints"),
            filename="best-{epoch:02d}-{val_accuracy:.4f}",
            monitor=monitor_metric,
            mode="max",
            save_top_k=3,
            save_last=True,
            verbose=True,
        ),
        
        # 조기 종료
        EarlyStopping(
            monitor=monitor_metric,
            mode="max",
            patience=10,
            verbose=True,
        ),
        
        # 학습률 모니터링
        LearningRateMonitor(logging_interval="epoch"),
        
        # 진행률 표시
        RichProgressBar(),
    ]
    
    return callbacks


def main():
    """메인 함수"""
    args = parse_args()
    
    # 시드 설정
    pl.seed_everything(args.seed, workers=True)
    
    # 출력 디렉토리 생성
    output_dir = Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 수어 분류 모델 학습 시작")
    print(f"📁 데이터 경로: {args.data_root}")
    print(f"🎯 모델: {args.model_name}")
    print(f"📊 배치 크기: {args.batch_size}")
    print(f"🔄 에포크: {args.epochs}")
    print(f"💾 출력 경로: {output_dir}")
    
    # 데이터 모듈 설정
    print("\n📋 데이터 모듈 설정 중...")
    datamodule = SignLanguageDataModule(
        data_root=args.data_root,
        train_data_root=args.train_data_root,
        val_data_root=args.val_data_root,
        test_data_root=args.test_data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        clip_duration=args.clip_duration,
        num_frames=args.num_frames,
        crop_size=args.crop_size,
    )
    
    # 데이터 정보 확인을 위해 setup 호출
    datamodule.setup("fit")
    num_classes = datamodule.num_classes
    class_names = list(datamodule.class_to_idx.keys())
    
    print(f"✅ 클래스 개수: {num_classes}")
    print(f"📝 클래스 목록: {class_names}")
    
    # 모델 설정
    print(f"\n🤖 모델 설정 중... ({args.model_name})")
    
    # Efficient 모델을 위한 입력 크기 계산
    efficient_input_size = (args.batch_size, 3, args.num_frames, args.crop_size, args.crop_size)
    if args.model_name.startswith("efficient_x3d_xs"):
        efficient_input_size = (1, 3, 4, 160, 160)  # X3D-XS 권장 크기
    elif args.model_name.startswith("efficient_x3d_s"):
        efficient_input_size = (1, 3, 13, 160, 160)  # X3D-S 권장 크기
    
    model = SignLanguageClassifier(
        num_classes=num_classes,
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
        dropout_rate=args.dropout_rate,
        label_smoothing=args.label_smoothing,
        weight_decay=args.weight_decay,
        enable_efficient_deployment=args.enable_efficient_deployment,
        efficient_input_size=efficient_input_size,
    )
    
    # 로거 설정
    logger = TensorBoardLogger(
        save_dir=output_dir,
        name="tensorboard_logs",
        version=None,
    )
    
    # 콜백 설정
    callbacks = setup_callbacks(output_dir)
    
    # 트레이너 설정
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.gpus,
        precision=args.precision,
        logger=logger,
        callbacks=callbacks,
        fast_dev_run=args.fast_dev_run,
        deterministic=True,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
    )
    
    # 모델 요약 출력
    print(f"\n📈 모델 정보:")
    print(f"   - 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   - 학습 가능한 파라미터 수: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    if args.test_only:
        # 테스트만 수행
        print("\n🧪 테스트 모드")
        if args.resume_from_checkpoint:
            print(f"📥 체크포인트 로드: {args.resume_from_checkpoint}")
            model = SignLanguageClassifier.load_from_checkpoint(args.resume_from_checkpoint)
        trainer.test(model, datamodule=datamodule)
    else:
        # 학습 수행
        print("\n🏃 학습 시작!")
        trainer.fit(
            model,
            datamodule=datamodule,
            ckpt_path=args.resume_from_checkpoint,
        )
        
        # 최고 성능 모델로 테스트
        print("\n🏆 최고 성능 모델로 테스트 수행")
        trainer.test(datamodule=datamodule, ckpt_path="best")
        
        # 모바일 모델 내보내기 (efficient 모델인 경우)
        if args.export_mobile_model and args.model_name.startswith("efficient_"):
            print("\n📱 모바일 배포용 모델 내보내기 중...")
            try:
                # 최고 성능 체크포인트 로드
                best_checkpoint = None
                checkpoint_dir = output_dir / "checkpoints"
                if checkpoint_dir.exists():
                    checkpoints = list(checkpoint_dir.glob("best-*.ckpt"))
                    if checkpoints:
                        best_checkpoint = str(checkpoints[0])
                
                if best_checkpoint:
                    # 체크포인트에서 모델 로드
                    mobile_model = SignLanguageClassifier.load_from_checkpoint(
                        best_checkpoint,
                        enable_efficient_deployment=True,
                        efficient_input_size=efficient_input_size,
                    )
                    
                    # 모바일 모델 내보내기
                    mobile_model.export_to_mobile(
                        args.mobile_model_path, 
                        quantize=args.quantize_model
                    )
                    
                    print(f"✅ 모바일 모델 저장 완료: {args.mobile_model_path}")
                else:
                    print("⚠️ 최고 성능 체크포인트를 찾을 수 없어 모바일 모델 내보내기를 건너뜁니다.")
                    
            except Exception as e:
                print(f"❌ 모바일 모델 내보내기 실패: {e}")
    
    print(f"\n✅ 완료! 결과는 {output_dir}에 저장되었습니다.")
    print(f"📊 TensorBoard 로그: tensorboard --logdir {output_dir}/tensorboard_logs")
    
    # Efficient 모델 사용 팁 제공
    if args.model_name.startswith("efficient_"):
        print(f"\n💡 Efficient 모델 사용 팁:")
        print(f"   - 추론 시 --enable_efficient_deployment 플래그 사용으로 성능 향상")
        print(f"   - 모바일 배포: --export_mobile_model --quantize_model 플래그 사용")
        print(f"   - 권장 입력 크기: {efficient_input_size}")
        if args.model_name == "efficient_x3d_xs":
            print(f"   - 예상 성능: ~68.5% top-1 정확도 (Kinetics-400 기준)")
            print(f"   - 모바일 지연시간: ~233ms (fp32), ~165ms (int8) on Samsung S8")
        elif args.model_name == "efficient_x3d_s":
            print(f"   - 예상 성능: ~73.0% top-1 정확도 (Kinetics-400 기준)")
            print(f"   - 모바일 지연시간: ~764ms (fp32) on Samsung S8")


if __name__ == "__main__":
    main() 