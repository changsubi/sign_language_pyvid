#!/usr/bin/env python3
"""
PyTorchVideo Accelerator EfficientX3D 모델 학습 스크립트

X3D_XS (fp32), X3D_XS (int8), X3D_S (fp32) 모델 지원

사용 예시:
# X3D-XS 학습 (모바일 최적화)
python train_efficient_x3d.py --data_root ./datasest --model_variant XS --batch_size 4 --epochs 50

# X3D-S 학습 (균형잡힌 성능)
python train_efficient_x3d.py --data_root ./datasest --model_variant S --batch_size 2 --epochs 50 --export_mobile_model --quantize_model
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

warnings.filterwarnings("ignore", category=UserWarning)


def get_model_config(variant: str):
    """모델 변형에 따른 설정 반환"""
    configs = {
        "XS": {
            "model_name": "efficient_x3d_xs",
            "input_size": (1, 3, 4, 160, 160),
            "clip_duration": 1.0,  # 4 frames at 4 FPS
            "num_frames": 4,
            "crop_size": 160,
            "batch_size": 8,
            "learning_rate": 1e-3,
            "description": "모바일 최적화 - 빠른 추론, 낮은 메모리 사용량"
        },
        "S": {
            "model_name": "efficient_x3d_s", 
            "input_size": (1, 3, 13, 160, 160),
            "clip_duration": 2.17,  # 13 frames at 6 FPS
            "num_frames": 13,
            "crop_size": 160,
            "batch_size": 4,
            "learning_rate": 8e-4,
            "description": "균형잡힌 성능 - 정확도와 효율성의 밸런스"
        }
    }
    return configs.get(variant, configs["XS"])


def parse_args():
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(description="EfficientX3D 수어 분류 모델 학습")
    
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
    parser.add_argument("--model_variant", type=str, default="XS",
                       choices=["XS", "S"],
                       help="EfficientX3D 모델 변형 (XS: 모바일 최적화, S: 균형)")
    parser.add_argument("--pretrained", action="store_true", default=True,
                       help="사전 훈련된 가중치 사용")
    parser.add_argument("--freeze_backbone", action="store_true",
                       help="백본 네트워크 고정")
    parser.add_argument("--dropout_rate", type=float, default=0.5,
                       help="드롭아웃 비율")
    
    # 학습 관련 인수 (모델 설정으로 오버라이드 가능)
    parser.add_argument("--batch_size", type=int, default=None,
                       help="배치 크기 (기본값: 모델 설정값)")
    parser.add_argument("--learning_rate", type=float, default=None,
                       help="학습률 (기본값: 모델 설정값)")
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
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                       help="라벨 스무딩")
    
    # 데이터 로더 관련 인수
    parser.add_argument("--num_workers", type=int, default=4,
                       help="데이터 로더 워커 수")
    
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
    parser.add_argument("--experiment_name", type=str, default=None,
                       help="실험 이름 (기본값: efficient_x3d_{variant})")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                       help="체크포인트에서 재개")
    
    # EfficientX3D 전용 옵션
    parser.add_argument("--enable_efficient_deployment", action="store_true", default=True,
                       help="효율적 배포 모드 활성화")
    parser.add_argument("--export_mobile_model", action="store_true",
                       help="모바일 배포용 모델 내보내기")
    parser.add_argument("--quantize_model", action="store_true",
                       help="INT8 양자화 적용")
    parser.add_argument("--mobile_model_path", type=str, default=None,
                       help="모바일 모델 저장 경로 (기본값: 자동 생성)")
    
    # 기타 옵션
    parser.add_argument("--test_only", action="store_true",
                       help="테스트만 수행")
    parser.add_argument("--fast_dev_run", action="store_true",
                       help="빠른 개발 실행 (디버깅용)")
    
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
            patience=15,  # EfficientX3D는 더 긴 patience 사용
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
    
    # 모델 설정 가져오기
    model_config = get_model_config(args.model_variant)
    
    # 설정값 오버라이드
    batch_size = args.batch_size or model_config["batch_size"]
    learning_rate = args.learning_rate or model_config["learning_rate"]
    clip_duration = model_config["clip_duration"]
    num_frames = model_config["num_frames"]
    crop_size = model_config["crop_size"]
    efficient_input_size = model_config["input_size"]
    
    # 실험 이름 설정
    experiment_name = args.experiment_name or f"efficient_x3d_{args.model_variant.lower()}"
    
    # 모바일 모델 경로 설정
    mobile_model_path = args.mobile_model_path or f"./mobile_x3d_{args.model_variant.lower()}.pt"
    
    # 시드 설정
    pl.seed_everything(args.seed, workers=True)
    
    # 출력 디렉토리 생성
    output_dir = Path(args.output_dir) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 EfficientX3D-{args.model_variant} 수어 분류 모델 학습 시작")
    print(f"📝 모델 설명: {model_config['description']}")
    print(f"📁 데이터 경로: {args.data_root}")
    print(f"🎯 모델: {model_config['model_name']}")
    print(f"📊 배치 크기: {batch_size}")
    print(f"🔄 에포크: {args.epochs}")
    print(f"📐 입력 크기: {efficient_input_size}")
    print(f"🎬 클립 길이: {clip_duration}초 ({num_frames} 프레임)")
    print(f"💾 출력 경로: {output_dir}")
    
    # 데이터 모듈 설정
    print("\n📋 데이터 모듈 설정 중...")
    datamodule = SignLanguageDataModule(
        data_root=args.data_root,
        train_data_root=args.train_data_root,
        val_data_root=args.val_data_root,
        test_data_root=args.test_data_root,
        batch_size=batch_size,
        num_workers=args.num_workers,
        clip_duration=clip_duration,
        num_frames=num_frames,
        crop_size=crop_size,
    )
    
    # 데이터 정보 확인
    datamodule.setup("fit")
    num_classes = datamodule.num_classes
    class_names = list(datamodule.class_to_idx.keys())
    
    print(f"✅ 클래스 개수: {num_classes}")
    print(f"📝 클래스 목록: {class_names}")
    
    # 모델 설정
    print(f"\n🤖 EfficientX3D-{args.model_variant} 모델 설정 중...")
    model = SignLanguageClassifier(
        num_classes=num_classes,
        model_name=model_config["model_name"],
        learning_rate=learning_rate,
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
    print(f"   - 모델 크기: ~3.8M 파라미터")
    
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
        
        # 모바일 모델 내보내기
        if args.export_mobile_model:
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
                        mobile_model_path, 
                        quantize=args.quantize_model
                    )
                    
                    print(f"✅ 모바일 모델 저장 완료: {mobile_model_path}")
                else:
                    print("⚠️ 최고 성능 체크포인트를 찾을 수 없어 모바일 모델 내보내기를 건너뜁니다.")
                    
            except Exception as e:
                print(f"❌ 모바일 모델 내보내기 실패: {e}")
    
    print(f"\n✅ 완료! 결과는 {output_dir}에 저장되었습니다.")
    print(f"📊 TensorBoard 로그: tensorboard --logdir {output_dir}/tensorboard_logs")
    
    # 성능 및 사용 가이드
    print(f"\n🎯 EfficientX3D-{args.model_variant} 성능 정보:")
    if args.model_variant == "XS":
        print(f"   📊 Kinetics-400 정확도: ~68.5% (top-1), ~88.0% (top-5)")
        print(f"   ⚡ 모바일 지연시간: ~233ms (fp32), ~165ms (int8) on Samsung S8")
        print(f"   💾 모델 크기: ~3.8M 파라미터, ~15MB")
        print(f"   🎯 용도: 실시간 모바일 애플리케이션, IoT 디바이스")
    elif args.model_variant == "S":
        print(f"   📊 Kinetics-400 정확도: ~73.0% (top-1), ~90.6% (top-5)")
        print(f"   ⚡ 모바일 지연시간: ~764ms (fp32) on Samsung S8")
        print(f"   💾 모델 크기: ~3.8M 파라미터, ~15MB")
        print(f"   🎯 용도: 균형잡힌 모바일 애플리케이션")
    
    print(f"\n💡 사용 가이드:")
    print(f"   🔧 추론: inference.py --checkpoint {output_dir}/checkpoints/best-*.ckpt")
    print(f"   📱 모바일: {mobile_model_path} 파일을 모바일 앱에 통합")
    print(f"   ⚡ 최적화: 추론 시 배포 모드 자동 활성화")
    
    if args.quantize_model and args.export_mobile_model:
        print(f"   🗜️ 양자화: INT8 모델로 ~75% 크기 감소, ~30% 속도 향상")


if __name__ == "__main__":
    main() 