#!/usr/bin/env python3
"""
EfficientX3D 모델 기능 데모 스크립트

이 스크립트는 EfficientX3D 모델의 다양한 기능을 시연합니다:
1. 모델 생성 및 정보 출력
2. 추론 성능 벤치마킹
3. 모바일 배포 모드 변환
4. INT8 양자화
5. TorchScript 내보내기

사용 예시:
python demo_efficient_x3d.py --variant XS
python demo_efficient_x3d.py --variant S --benchmark --export_mobile
"""

import argparse
import time
import torch
import torch.nn as nn
from pathlib import Path
import psutil
import os

from pytorchvideo.models.accelerator.mobile_cpu.efficient_x3d import create_x3d
from pytorchvideo.accelerator.deployment.mobile_cpu.utils.model_conversion import (
    convert_to_deployable_form,
)


def get_model_config(variant: str):
    """모델 변형별 설정"""
    configs = {
        "XS": {
            "input_size": (1, 3, 4, 160, 160),
            "expansion": "XS",
            "expected_accuracy": {"top1": 68.5, "top5": 88.0},
            "mobile_latency": {"fp32": 233, "int8": 165},
            "description": "모바일 최적화 - 실시간 애플리케이션용"
        },
        "S": {
            "input_size": (1, 3, 13, 160, 160),
            "expansion": "S", 
            "expected_accuracy": {"top1": 73.0, "top5": 90.6},
            "mobile_latency": {"fp32": 764, "int8": None},
            "description": "균형잡힌 성능 - 정확도와 효율성의 밸런스"
        }
    }
    return configs.get(variant, configs["XS"])


def create_demo_model(variant: str, num_classes: int = 400):
    """데모용 EfficientX3D 모델 생성"""
    config = get_model_config(variant)
    
    model = create_x3d(
        num_classes=num_classes,
        dropout=0.5,
        expansion=config["expansion"],
        head_act="relu",
        enable_head=True,
    )
    
    return model, config


def benchmark_model(model, input_tensor, num_runs: int = 100, warmup_runs: int = 10):
    """모델 추론 성능 벤치마킹"""
    model.eval()
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    # Warmup
    print(f"⏳ Warmup 중... ({warmup_runs} runs)")
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_tensor)
    
    # 메모리 사용량 측정
    if device.type == 'cuda':
        torch.cuda.synchronize()
        memory_before = torch.cuda.memory_allocated() / 1024**2  # MB
    else:
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024**2  # MB
    
    # 벤치마킹
    print(f"🔥 벤치마킹 중... ({num_runs} runs)")
    times = []
    
    with torch.no_grad():
        for i in range(num_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.time()
            output = model(input_tensor)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # ms
            
            if (i + 1) % 20 == 0:
                print(f"  진행률: {i+1}/{num_runs}")
    
    # 메모리 사용량 측정
    if device.type == 'cuda':
        memory_after = torch.cuda.memory_allocated() / 1024**2  # MB
        memory_used = memory_after - memory_before
    else:
        process = psutil.Process(os.getpid())
        memory_after = process.memory_info().rss / 1024**2  # MB
        memory_used = memory_after - memory_before
    
    return {
        'mean_time': sum(times) / len(times),
        'min_time': min(times),
        'max_time': max(times),
        'std_time': (sum([(t - sum(times) / len(times))**2 for t in times]) / len(times))**0.5,
        'memory_used': memory_used,
        'output_shape': output.shape
    }


def quantize_model(model, input_tensor):
    """모델 INT8 양자화"""
    print("🔄 INT8 양자화 진행 중...")
    
    try:
        from torch.quantization import (
            convert, prepare, get_default_qconfig, QuantStub, DeQuantStub
        )
        
        # 양자화를 위한 래퍼 클래스
        class QuantWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.quant = QuantStub()
                self.model = model
                self.dequant = DeQuantStub()
            
            def forward(self, x):
                x = self.quant(x)
                x = self.model(x)
                x = self.dequant(x)
                return x
        
        # 양자화 설정
        torch.backends.quantized.engine = "qnnpack"
        wrapped_model = QuantWrapper(model.cpu())
        wrapped_model.qconfig = get_default_qconfig("qnnpack")
        
        # 양자화 준비 및 실행
        prepared_model = prepare(wrapped_model)
        
        # 캘리브레이션
        with torch.no_grad():
            prepared_model(input_tensor.cpu())
        
        quantized_model = convert(prepared_model)
        
        print("✅ 양자화 완료!")
        return quantized_model
        
    except Exception as e:
        print(f"❌ 양자화 실패: {e}")
        return None


def export_mobile_models(model, input_tensor, variant: str, output_dir: str = "./mobile_models"):
    """모바일 배포용 모델 내보내기"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("\n📱 모바일 모델 내보내기")
    
    # 1. FP32 배포 모델
    try:
        print("🔄 FP32 배포 모델 변환 중...")
        deployed_model = convert_to_deployable_form(model, input_tensor)
        
        from torch.utils.mobile_optimizer import optimize_for_mobile
        traced_model = torch.jit.trace(deployed_model, input_tensor, strict=False)
        optimized_model = optimize_for_mobile(traced_model)
        
        fp32_path = output_dir / f"efficient_x3d_{variant.lower()}_fp32.pt"
        optimized_model.save(str(fp32_path))
        print(f"✅ FP32 모델 저장: {fp32_path}")
        
        # 2. INT8 양자화 모델
        print("🔄 INT8 양자화 모델 생성 중...")
        quantized_model = quantize_model(deployed_model, input_tensor)
        
        if quantized_model is not None:
            traced_quant = torch.jit.trace(quantized_model, input_tensor.cpu(), strict=False)
            optimized_quant = optimize_for_mobile(traced_quant)
            
            int8_path = output_dir / f"efficient_x3d_{variant.lower()}_int8.pt"
            optimized_quant.save(str(int8_path))
            print(f"✅ INT8 모델 저장: {int8_path}")
            
            return fp32_path, int8_path
        else:
            return fp32_path, None
            
    except Exception as e:
        print(f"❌ 모바일 모델 내보내기 실패: {e}")
        return None, None


def main():
    parser = argparse.ArgumentParser(description="EfficientX3D 모델 기능 데모")
    parser.add_argument("--variant", type=str, default="XS", choices=["XS", "S"],
                       help="EfficientX3D 모델 변형")
    parser.add_argument("--num_classes", type=int, default=400,
                       help="분류 클래스 수")
    parser.add_argument("--benchmark", action="store_true",
                       help="추론 성능 벤치마킹 수행")
    parser.add_argument("--num_runs", type=int, default=100,
                       help="벤치마킹 실행 횟수")
    parser.add_argument("--export_mobile", action="store_true",
                       help="모바일 모델 내보내기")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cpu", "cuda"],
                       help="실행 디바이스")
    
    args = parser.parse_args()
    
    # 디바이스 설정
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"🚀 EfficientX3D-{args.variant} 모델 데모")
    print(f"💻 디바이스: {device}")
    print(f"🎯 클래스 수: {args.num_classes}")
    
    # 모델 생성
    print(f"\n🤖 EfficientX3D-{args.variant} 모델 생성 중...")
    model, config = create_demo_model(args.variant, args.num_classes)
    model = model.to(device)
    
    # 모델 정보 출력
    print(f"\n📊 모델 정보:")
    print(f"   - 설명: {config['description']}")
    print(f"   - 입력 크기: {config['input_size']}")
    print(f"   - 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   - 학습 가능한 파라미터 수: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    expected_acc = config['expected_accuracy']
    print(f"   - 예상 성능 (Kinetics-400): Top-1 {expected_acc['top1']}%, Top-5 {expected_acc['top5']}%")
    
    mobile_lat = config['mobile_latency']
    print(f"   - 모바일 지연시간 (Samsung S8): FP32 {mobile_lat['fp32']}ms", end="")
    if mobile_lat['int8']:
        print(f", INT8 {mobile_lat['int8']}ms")
    else:
        print()
    
    # 입력 텐서 생성
    input_tensor = torch.randn(config['input_size']).to(device)
    print(f"\n🎬 입력 텐서 크기: {input_tensor.shape}")
    
    # 기본 추론 테스트
    print(f"\n🔍 기본 추론 테스트...")
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        output = model(input_tensor)
        end_time = time.time()
    
    print(f"   - 출력 크기: {output.shape}")
    print(f"   - 추론 시간: {(end_time - start_time) * 1000:.2f}ms")
    print(f"   - 출력 범위: [{output.min().item():.4f}, {output.max().item():.4f}]")
    
    # 성능 벤치마킹
    if args.benchmark:
        print(f"\n⏱️ 성능 벤치마킹 시작...")
        benchmark_results = benchmark_model(model, input_tensor, args.num_runs)
        
        print(f"\n📈 벤치마킹 결과:")
        print(f"   - 평균 추론 시간: {benchmark_results['mean_time']:.2f} ± {benchmark_results['std_time']:.2f}ms")
        print(f"   - 최소 추론 시간: {benchmark_results['min_time']:.2f}ms")
        print(f"   - 최대 추론 시간: {benchmark_results['max_time']:.2f}ms")
        print(f"   - 메모리 사용량: {benchmark_results['memory_used']:.2f}MB")
        print(f"   - 처리량: {1000 / benchmark_results['mean_time']:.1f} FPS")
    
    # 배포 모드 변환 데모
    print(f"\n🔧 배포 모드 변환 데모...")
    try:
        deployed_model = convert_to_deployable_form(model, input_tensor)
        print("✅ 배포 모드 변환 성공!")
        
        # 배포 모델 추론 테스트
        with torch.no_grad():
            start_time = time.time()
            deploy_output = deployed_model(input_tensor)
            end_time = time.time()
        
        print(f"   - 배포 모델 추론 시간: {(end_time - start_time) * 1000:.2f}ms")
        
        # 결과 비교
        max_diff = torch.max(torch.abs(output - deploy_output)).item()
        print(f"   - 출력 차이 (최대): {max_diff:.6f}")
        
        if args.benchmark:
            print(f"\n⏱️ 배포 모델 벤치마킹...")
            deploy_benchmark = benchmark_model(deployed_model, input_tensor, args.num_runs // 2)
            print(f"   - 배포 모델 평균 시간: {deploy_benchmark['mean_time']:.2f}ms")
            improvement = (benchmark_results['mean_time'] - deploy_benchmark['mean_time']) / benchmark_results['mean_time'] * 100
            print(f"   - 성능 향상: {improvement:.1f}%")
        
    except Exception as e:
        print(f"❌ 배포 모드 변환 실패: {e}")
        deployed_model = model
    
    # 모바일 모델 내보내기
    if args.export_mobile:
        fp32_path, int8_path = export_mobile_models(deployed_model, input_tensor, args.variant)
        
        if fp32_path:
            # 내보낸 모델 테스트
            print(f"\n🧪 내보낸 모델 테스트...")
            try:
                mobile_model = torch.jit.load(str(fp32_path))
                mobile_model.eval()
                
                with torch.no_grad():
                    mobile_output = mobile_model(input_tensor.cpu())
                
                print(f"✅ FP32 모바일 모델 로드 및 추론 성공!")
                
                if int8_path:
                    int8_model = torch.jit.load(str(int8_path))
                    int8_model.eval()
                    
                    with torch.no_grad():
                        int8_output = int8_model(input_tensor.cpu())
                    
                    print(f"✅ INT8 모바일 모델 로드 및 추론 성공!")
                    
                    # 양자화 정확도 비교
                    accuracy_loss = torch.mean(torch.abs(mobile_output - int8_output)).item()
                    print(f"   - 양자화로 인한 정확도 손실: {accuracy_loss:.6f}")
                
            except Exception as e:
                print(f"❌ 내보낸 모델 테스트 실패: {e}")
    
    print(f"\n🎉 EfficientX3D-{args.variant} 데모 완료!")
    
    # 사용 가이드
    print(f"\n💡 사용 가이드:")
    print(f"   🎯 용도: {config['description']}")
    print(f"   📱 모바일 통합: TorchScript 모델 파일을 앱에 포함")
    print(f"   ⚡ 최적화: 추론 시 배포 모드 자동 사용")
    print(f"   🗜️ 압축: INT8 양자화로 ~75% 크기 감소")
    
    if args.variant == "XS":
        print(f"   🚀 실시간 애플리케이션에 적합 (< 250ms 지연시간)")
    else:
        print(f"   ⚖️ 정확도와 효율성의 균형 (~700ms 지연시간)")


if __name__ == "__main__":
    main() 