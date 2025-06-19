import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torchmetrics import Accuracy, F1Score, ConfusionMatrix
import pytorchvideo.models as pv_models
from pytorchvideo.models.accelerator.mobile_cpu.efficient_x3d import create_x3d
from pytorchvideo.accelerator.deployment.mobile_cpu.utils.model_conversion import (
    convert_to_deployable_form,
)
from typing import Dict, Any, Optional


class SignLanguageClassifier(pl.LightningModule):
    """
    수어 분류를 위한 PyTorch Lightning 모델
    """
    
    def __init__(
        self,
        num_classes: int,
        model_name: str = "slow_r50",
        learning_rate: float = 1e-3,
        optimizer: str = "adam",
        scheduler: str = "cosine",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout_rate: float = 0.5,
        label_smoothing: float = 0.0,
        weight_decay: float = 1e-4,
        enable_efficient_deployment: bool = False,
        efficient_input_size: tuple = (1, 3, 4, 160, 160),
        **kwargs
    ):
        """
        Args:
            num_classes: 클래스 개수
            model_name: 모델 이름 (slow_r50, x3d_m, mvit_base_16x4, efficient_x3d_xs, efficient_x3d_s 등)
            learning_rate: 학습률
            optimizer: 옵티마이저 (adam, sgd)
            scheduler: 스케줄러 (cosine, step, none)
            pretrained: 사전 훈련된 가중치 사용 여부
            freeze_backbone: 백본 네트워크 고정 여부
            dropout_rate: 드롭아웃 비율
            label_smoothing: 라벨 스무딩
            weight_decay: 가중치 감쇠
            enable_efficient_deployment: 효율적 배포 모드 활성화 (추론 시 자동 변환)
            efficient_input_size: 효율적 모델 변환을 위한 입력 크기
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer
        self.scheduler_name = scheduler
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.enable_efficient_deployment = enable_efficient_deployment
        self.efficient_input_size = efficient_input_size
        self.is_efficient_model = model_name.startswith("efficient_")
        self.deployed_model = None
        
        # 모델 생성
        self.model = self._create_model(
            model_name, num_classes, pretrained, freeze_backbone, dropout_rate
        )
        
        # 손실 함수
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
        # 메트릭
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.test_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        
        # 혼동 행렬 (테스트 시에만 사용)
        self.test_confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=num_classes)
    
    def _create_model(self, model_name: str, num_classes: int, pretrained: bool, 
                     freeze_backbone: bool, dropout_rate: float) -> nn.Module:
        """모델 생성"""
        
        # PyTorchVideo 모델 생성
        if model_name == "slow_r50":
            model = pv_models.slow_r50(pretrained=pretrained)
            if hasattr(model, 'blocks') and hasattr(model.blocks[-1], 'proj'):
                in_features = model.blocks[-1].proj.in_features
                model.blocks[-1].proj = nn.Sequential(
                    nn.Dropout(dropout_rate),
                    nn.Linear(in_features, num_classes)
                )
        elif model_name == "x3d_m":
            model = pv_models.x3d_m(pretrained=pretrained)
            if hasattr(model, 'blocks') and hasattr(model.blocks[-1], 'proj'):
                in_features = model.blocks[-1].proj.in_features
                model.blocks[-1].proj = nn.Sequential(
                    nn.Dropout(dropout_rate),
                    nn.Linear(in_features, num_classes)
                )
        elif model_name == "x3d_s":
            model = pv_models.x3d_s(pretrained=pretrained)
            if hasattr(model, 'blocks') and hasattr(model.blocks[-1], 'proj'):
                in_features = model.blocks[-1].proj.in_features
                model.blocks[-1].proj = nn.Sequential(
                    nn.Dropout(dropout_rate),
                    nn.Linear(in_features, num_classes)
                )
        elif model_name == "mvit_base_16x4":
            model = pv_models.mvit_base_16x4(pretrained=pretrained)
            if hasattr(model, 'head') and hasattr(model.head, 'proj'):
                in_features = model.head.proj.in_features
                model.head.proj = nn.Sequential(
                    nn.Dropout(dropout_rate),
                    nn.Linear(in_features, num_classes)
                )
        elif model_name == "efficient_x3d_xs":
            model = create_x3d(
                num_classes=num_classes,
                dropout=dropout_rate,
                expansion="XS",
                head_act="relu",
                enable_head=True,
            )
            # 사전 훈련된 가중치 로드 (선택사항)
            if pretrained:
                try:
                    from torch.hub import load_state_dict_from_url
                    checkpoint_url = "https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/efficient_x3d_xs_original_form.pyth"
                    state_dict = load_state_dict_from_url(checkpoint_url, map_location="cpu")
                    # 마지막 분류 레이어는 클래스 수가 다를 수 있으므로 제외
                    filtered_state_dict = {k: v for k, v in state_dict.items() 
                                         if not k.startswith('projection')}
                    model.load_state_dict(filtered_state_dict, strict=False)
                except:
                    print("Warning: Could not load pretrained weights for efficient_x3d_xs")
        elif model_name == "efficient_x3d_s":
            model = create_x3d(
                num_classes=num_classes,
                dropout=dropout_rate,
                expansion="S",
                head_act="relu",
                enable_head=True,
            )
            # 사전 훈련된 가중치 로드 (선택사항)
            if pretrained:
                try:
                    from torch.hub import load_state_dict_from_url
                    checkpoint_url = "https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/efficient_x3d_s_original_form.pyth"
                    state_dict = load_state_dict_from_url(checkpoint_url, map_location="cpu")
                    # 마지막 분류 레이어는 클래스 수가 다를 수 있으므로 제외
                    filtered_state_dict = {k: v for k, v in state_dict.items() 
                                         if not k.startswith('projection')}
                    model.load_state_dict(filtered_state_dict, strict=False)
                except:
                    print("Warning: Could not load pretrained weights for efficient_x3d_s")
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # 백본 고정
        if freeze_backbone:
            for name, param in model.named_parameters():
                if "proj" not in name and "head" not in name:
                    param.requires_grad = False
        
        return model
    
    def forward(self, x):
        # 효율적 배포 모드가 활성화되고 추론 모드일 때
        if (self.enable_efficient_deployment and 
            not self.training and 
            self.is_efficient_model and 
            self.deployed_model is not None):
            return self.deployed_model(x)
        return self.model(x)
    
    def convert_to_deployment_mode(self):
        """
        효율적 모델을 배포 모드로 변환
        이 메서드는 추론 전에 호출되어야 합니다.
        """
        if self.is_efficient_model and self.enable_efficient_deployment:
            try:
                # 예시 입력 텐서 생성
                input_tensor = torch.randn(self.efficient_input_size)
                if next(self.parameters()).is_cuda:
                    input_tensor = input_tensor.cuda()
                
                # 배포 모드로 변환
                self.deployed_model = convert_to_deployable_form(self.model, input_tensor)
                print("✅ 모델이 효율적 배포 모드로 변환되었습니다.")
                return True
            except Exception as e:
                print(f"⚠️ 배포 모드 변환 실패: {e}")
                return False
        return False
    
    def export_to_mobile(self, output_path: str, quantize: bool = False):
        """
        모바일 배포를 위한 모델 내보내기
        
        Args:
            output_path: 출력 파일 경로 (.pt 확장자)
            quantize: INT8 양자화 여부
        """
        if not self.is_efficient_model:
            raise ValueError("모바일 내보내기는 efficient 모델에서만 지원됩니다.")
        
        self.eval()
        
        # 배포 모드로 변환
        if self.deployed_model is None:
            if not self.convert_to_deployment_mode():
                raise RuntimeError("배포 모드 변환에 실패했습니다.")
        
        input_tensor = torch.randn(self.efficient_input_size)
        if next(self.parameters()).is_cuda:
            input_tensor = input_tensor.cuda()
        
        if quantize:
            # INT8 양자화
            print("🔄 INT8 양자화 진행 중...")
            try:
                from torch.quantization import (
                    convert, prepare, get_default_qconfig, QuantStub, DeQuantStub
                )
                from torch.utils.mobile_optimizer import optimize_for_mobile
                
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
                wrapped_model = QuantWrapper(self.deployed_model.cpu())
                wrapped_model.qconfig = get_default_qconfig("qnnpack")
                
                # 양자화 준비 및 실행
                prepared_model = prepare(wrapped_model)
                # 캘리브레이션 (실제로는 몇 개의 샘플 데이터 필요)
                with torch.no_grad():
                    prepared_model(input_tensor.cpu())
                
                quantized_model = convert(prepared_model)
                
                # TorchScript로 내보내기
                traced_model = torch.jit.trace(quantized_model, input_tensor.cpu(), strict=False)
                optimized_model = optimize_for_mobile(traced_model)
                optimized_model.save(output_path)
                
                print(f"✅ 양자화된 모델이 저장되었습니다: {output_path}")
                
            except Exception as e:
                print(f"❌ 양자화 실패: {e}")
                print("🔄 FP32 모델로 대신 저장합니다...")
                quantize = False
        
        if not quantize:
            # FP32 모델 내보내기
            try:
                from torch.utils.mobile_optimizer import optimize_for_mobile
                
                traced_model = torch.jit.trace(self.deployed_model, input_tensor, strict=False)
                optimized_model = optimize_for_mobile(traced_model)
                optimized_model.save(output_path)
                
                print(f"✅ FP32 모델이 저장되었습니다: {output_path}")
                
            except Exception as e:
                print(f"❌ 모델 내보내기 실패: {e}")
                raise
    
    def training_step(self, batch, batch_idx):
        """학습 스텝"""
        videos, labels = batch["video"], batch["label"]
        
        # 라벨을 인덱스로 변환 (문자열 라벨인 경우)
        if isinstance(labels[0], str):
            # 클래스 이름을 인덱스로 변환
            label_indices = []
            for label in labels:
                if hasattr(self.trainer.datamodule, 'class_to_idx'):
                    label_indices.append(self.trainer.datamodule.class_to_idx[label])
                else:
                    # fallback: 문자열을 해시하여 인덱스 생성 (임시)
                    label_indices.append(hash(label) % self.num_classes)
            labels = torch.tensor(label_indices, device=self.device)
        
        # 순전파
        logits = self(videos)
        loss = self.criterion(logits, labels)
        
        # 메트릭 계산
        preds = torch.argmax(logits, dim=1)
        self.train_accuracy(preds, labels)
        
        # 로깅
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/accuracy", self.train_accuracy, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """검증 스텝"""
        videos, labels = batch["video"], batch["label"]
        
        # 라벨을 인덱스로 변환
        if isinstance(labels[0], str):
            label_indices = []
            for label in labels:
                if hasattr(self.trainer.datamodule, 'class_to_idx'):
                    label_indices.append(self.trainer.datamodule.class_to_idx[label])
                else:
                    label_indices.append(hash(label) % self.num_classes)
            labels = torch.tensor(label_indices, device=self.device)
        
        # 순전파
        logits = self(videos)
        loss = self.criterion(logits, labels)
        
        # 메트릭 계산
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy(preds, labels)
        self.val_f1(preds, labels)
        
        # 로깅
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log("val/accuracy", self.val_accuracy, on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_f1, on_epoch=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """테스트 스텝"""
        videos, labels = batch["video"], batch["label"]
        
        # 라벨을 인덱스로 변환
        if isinstance(labels[0], str):
            label_indices = []
            for label in labels:
                if hasattr(self.trainer.datamodule, 'class_to_idx'):
                    label_indices.append(self.trainer.datamodule.class_to_idx[label])
                else:
                    label_indices.append(hash(label) % self.num_classes)
            labels = torch.tensor(label_indices, device=self.device)
        
        # 순전파
        logits = self(videos)
        loss = self.criterion(logits, labels)
        
        # 메트릭 계산
        preds = torch.argmax(logits, dim=1)
        self.test_accuracy(preds, labels)
        self.test_f1(preds, labels)
        self.test_confusion_matrix(preds, labels)
        
        # 로깅
        self.log("test/loss", loss, on_epoch=True)
        self.log("test/accuracy", self.test_accuracy, on_epoch=True)
        self.log("test/f1", self.test_f1, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self):
        """옵티마이저 및 스케줄러 설정"""
        
        # 옵티마이저 선택
        if self.optimizer_name.lower() == "adam":
            optimizer = Adam(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_name.lower() == "sgd":
            optimizer = SGD(
                self.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")
        
        # 스케줄러 선택
        if self.scheduler_name.lower() == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                }
            }
        elif self.scheduler_name.lower() == "step":
            scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                }
            }
        else:
            return optimizer
    
    def on_test_epoch_end(self):
        """테스트 에포크 종료 시 혼동 행렬 출력"""
        confusion_matrix = self.test_confusion_matrix.compute()
        
        print("\nConfusion Matrix:")
        print(confusion_matrix)
        
        # 클래스별 정확도 계산
        if hasattr(self.trainer.datamodule, 'idx_to_class'):
            class_names = [self.trainer.datamodule.idx_to_class[i] for i in range(self.num_classes)]
            
            print("\nPer-class Accuracy:")
            for i, class_name in enumerate(class_names):
                if confusion_matrix[i].sum() > 0:
                    class_acc = confusion_matrix[i, i] / confusion_matrix[i].sum()
                    print(f"{class_name}: {class_acc:.4f}") 