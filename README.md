# 수어 분류 (Sign Language Classification) 프로젝트

PyTorchVideo를 기반으로 한 수어 비디오 분류 시스템입니다. 수어 동작을 인식하고 분류할 수 있는 딥러닝 모델을 학습하고 추론할 수 있습니다.

## 📁 프로젝트 구조

```
.
├── sign_language_dataset.py      # 수어 데이터셋 클래스
├── sign_language_datamodule.py   # PyTorch Lightning 데이터 모듈
├── sign_language_model.py        # 수어 분류 모델
├── train_sign_language.py        # 학습 스크립트
├── inference.py                  # 추론 스크립트
├── requirements.txt              # 필요 패키지 목록
└── README_sign_language.md       # 프로젝트 가이드
```

## 📊 데이터 형식

### 디렉토리 구조
```
your_data_root/
├── video/
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
└── label/
    ├── video1_morpheme.json
    ├── video2_morpheme.json
    └── ...
```

### 라벨 JSON 형식
```json
{
    "metaData": {
        "name": "video_name.mp4",
        "duration": 3.884,
        "url": "...",
        "exportedOn": "2020/12/10"
    },
    "data": [
        {
            "start": 1.879,
            "end": 3.236,
            "attributes": [
                {
                    "name": "수어_단어명"
                }
            ]
        }
    ]
}
```

## 🚀 설치 및 설정

### 1. 환경 설정
```bash
# 가상환경 생성 (선택사항)
python -m venv sign_language_env
source sign_language_env/bin/activate  # Linux/Mac
# sign_language_env\Scripts\activate  # Windows

# 패키지 설치
pip install -r requirements.txt
```

### 2. GPU 설정 (선택사항)
CUDA가 설치된 환경에서 GPU 가속을 사용하려면:
```bash
# PyTorch GPU 버전 설치 (CUDA 버전에 맞게)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 🎯 사용 방법

### 1. 모델 학습

#### 기본 학습
```bash
python train_sign_language.py \
    --data_root ./datasest \
    --model_name slow_r50 \
    --batch_size 4 \
    --epochs 50 \
    --learning_rate 1e-3
```

#### 고급 설정으로 학습
```bash
python train_sign_language.py \
    --data_root ./your_data_root \
    --train_data_root ./train_data \
    --val_data_root ./val_data \
    --model_name x3d_m \
    --batch_size 8 \
    --epochs 100 \
    --learning_rate 5e-4 \
    --optimizer adam \
    --scheduler cosine \
    --dropout_rate 0.3 \
    --weight_decay 1e-4 \
    --gpus 1 \
    --precision 16-mixed \
    --output_dir ./experiments \
    --experiment_name my_sign_language_model
```

#### 사용 가능한 모델들
- `slow_r50`: SlowFast ResNet-50 (기본값)
- `x3d_s`: X3D-S (효율적, 빠름)
- `x3d_m`: X3D-M (균형)
- `mvit_base_16x4`: MViT-Base (트랜스포머 기반)
- `efficient_x3d_xs`: EfficientX3D-XS (모바일 최적화, 3.8M 파라미터)
- `efficient_x3d_s`: EfficientX3D-S (균형잡힌 모바일 모델)

### 2. 모델 추론

#### 단일 비디오 추론
```bash
python inference.py \
    --checkpoint ./outputs/sign_language_classification/checkpoints/best.ckpt \
    --video_path ./test_video.mp4 \
    --clip_duration 2.0 \
    --output_path ./result.json
```

#### 다중 클립 추론
```bash
python inference.py \
    --checkpoint ./outputs/sign_language_classification/checkpoints/best.ckpt \
    --video_path ./test_video.mp4 \
    --multiple_clips \
    --stride 1.0 \
    --return_probabilities \
    --output_path ./results.json
```

### 3. 학습 모니터링

TensorBoard를 사용하여 학습 과정을 모니터링할 수 있습니다:
```bash
tensorboard --logdir ./outputs/sign_language_classification/tensorboard_logs
```

## ⚙️ 주요 매개변수

### 학습 매개변수
- `--data_root`: 데이터 루트 경로
- `--model_name`: 사용할 모델 (slow_r50, x3d_s, x3d_m, mvit_base_16x4)
- `--batch_size`: 배치 크기 (GPU 메모리에 따라 조정)
- `--learning_rate`: 학습률
- `--epochs`: 학습 에포크 수
- `--num_frames`: 비디오에서 샘플링할 프레임 수
- `--clip_duration`: 비디오 클립 길이 (초)

### 최적화 매개변수
- `--optimizer`: adam 또는 sgd
- `--scheduler`: cosine, step, 또는 none
- `--weight_decay`: 가중치 감쇠
- `--dropout_rate`: 드롭아웃 비율
- `--label_smoothing`: 라벨 스무딩

### 데이터 처리 매개변수
- `--num_workers`: 데이터 로더 워커 수
- `--crop_size`: 이미지 크롭 크기
- `--pin_memory`: GPU 메모리 고정

## 🔧 모델 커스터마이징

### 새로운 모델 추가
`sign_language_model.py`의 `_create_model` 메서드에 새로운 모델을 추가할 수 있습니다:

```python
elif model_name == "custom_model":
    model = your_custom_model(pretrained=pretrained)
    # 분류 헤드 수정
    model.head = nn.Linear(model.head.in_features, num_classes)
```

### 데이터 증강 커스터마이징
`sign_language_datamodule.py`의 변환 함수를 수정하여 데이터 증강을 조정할 수 있습니다.

## 📈 성능 최적화 팁

### 1. 배치 크기 조정
- GPU 메모리에 따라 배치 크기를 조정하세요
- 일반적으로 4-16 사이의 값이 적합합니다

### 2. 혼합 정밀도 사용
```bash
--precision 16-mixed
```

### 3. 데이터 로더 최적화
```bash
--num_workers 4  # CPU 코어 수에 따라 조정
--pin_memory    # GPU 사용 시 활성화
```

### 4. 모델 선택

## 📱 모바일 배포 (EfficientX3D)

### EfficientX3D 모델 특징
- **X3D-XS**: 모바일 최적화 모델 (3.8M 파라미터, ~15MB)
  - Kinetics-400 정확도: 68.5% (top-1), 88.0% (top-5)
  - 모바일 지연시간: 233ms (fp32), 165ms (int8) on Samsung S8
  - 용도: 실시간 모바일 애플리케이션, IoT 디바이스

- **X3D-S**: 균형잡힌 모바일 모델 (3.8M 파라미터, ~15MB)
  - Kinetics-400 정확도: 73.0% (top-1), 90.6% (top-5)
  - 모바일 지연시간: 764ms (fp32) on Samsung S8
  - 용도: 균형잡힌 모바일 애플리케이션

### EfficientX3D 전용 학습 스크립트

```bash
# X3D-XS 학습 (모바일 최적화)
python train_efficient_x3d.py \
    --data_root ./datasest \
    --model_variant XS \
    --batch_size 8 \
    --epochs 50 \
    --export_mobile_model

# X3D-S 학습 (균형잡힌 성능) + INT8 양자화
python train_efficient_x3d.py \
    --data_root ./datasest \
    --model_variant S \
    --batch_size 4 \
    --epochs 50 \
    --export_mobile_model \
    --quantize_model
```

### 모바일 모델 내보내기

기존 학습 스크립트에서도 EfficientX3D 모델 사용 가능:
```bash
python train_sign_language.py \
    --data_root ./datasest \
    --model_name efficient_x3d_xs \
    --batch_size 8 \
    --epochs 50 \
    --enable_efficient_deployment \
    --export_mobile_model \
    --quantize_model \
    --mobile_model_path ./mobile_sign_language.pt
```

### 모바일 모델 사용
```python
import torch

# 모바일 모델 로드
mobile_model = torch.jit.load("./mobile_sign_language.pt")
mobile_model.eval()

# 추론
with torch.no_grad():
    output = mobile_model(input_tensor)
```

### 4. 모델 선택
- **빠른 프로토타이핑**: `x3d_s`
- **균형잡힌 성능**: `slow_r50` 또는 `x3d_m`
- **최고 정확도**: `mvit_base_16x4`

## 🚨 문제 해결

### 1. GPU 메모리 부족
```bash
# 배치 크기 줄이기
--batch_size 2

# 혼합 정밀도 사용
--precision 16-mixed

# 프레임 수 줄이기
--num_frames 8
```

### 2. 데이터 로딩 느림
```bash
# 워커 수 늘리기
--num_workers 8

# 메모리 고정 활성화
--pin_memory
```

### 3. 학습이 수렴하지 않음
```bash
# 학습률 조정
--learning_rate 1e-4

# 라벨 스무딩 추가
--label_smoothing 0.1

# 가중치 감쇠 조정
--weight_decay 1e-3
```

## 📊 예상 결과

### 학습 로그 예시
```
🚀 수어 분류 모델 학습 시작
📁 데이터 경로: ./datasest
🎯 모델: slow_r50
📊 배치 크기: 4
🔄 에포크: 50

📋 데이터 모듈 설정 중...
Training dataset: 20 videos
Validation dataset: 20 videos
✅ 클래스 개수: 5
📝 클래스 목록: ['왼쪽', '오른쪽', '위', '아래', '중앙']

🤖 모델 설정 중... (slow_r50)
📈 모델 정보:
   - 파라미터 수: 31,234,567
   - 학습 가능한 파라미터 수: 2,048

🏃 학습 시작!
Epoch 1/50: 100%|██████| 5/5 [00:45<00:00,  9.12s/it, loss=1.23, v_num=0, train/accuracy=0.400]
...
```

### 추론 결과 예시
```json
{
  "video_path": "./test_video.mp4",
  "start_time": 0.0,
  "end_time": 2.0,
  "predicted_class_idx": 0,
  "predicted_class_name": "왼쪽",
  "confidence": 0.8765,
  "clip_duration": 2.0
}
```

## 📚 참고 자료

- [PyTorchVideo 공식 문서](https://pytorchvideo.readthedocs.io/)
- [PyTorch Lightning 문서](https://lightning.ai/docs/pytorch/stable/)
- [비디오 분류 베스트 프랙티스](https://paperswithcode.com/task/video-classification)

## 🤝 기여하기

1. 이슈 리포트
2. 기능 제안
3. 코드 개선
4. 문서 업데이트

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 