import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import torch
from pytorchvideo.data.clip_sampling import ClipSampler
from pytorchvideo.data.video import VideoPathHandler


class SignLanguageDataset(torch.utils.data.Dataset):
    """
    수어 데이터셋을 위한 커스텀 데이터셋 클래스
    
    데이터 구조:
    - video 폴더: .mp4 비디오 파일들
    - label 폴더: _morpheme.json 라벨 파일들
    
    라벨 JSON 형식:
    {
        "metaData": {
            "name": "video_name.mp4",
            "duration": 3.884,
            ...
        },
        "data": [
            {
                "start": 1.879,
                "end": 3.236,
                "attributes": [
                    {"name": "왼쪽"}
                ]
            }
        ]
    }
    """
    
    def __init__(
        self,
        data_root: str,
        clip_sampler: ClipSampler,
        video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        decode_audio: bool = True,
        decoder: str = "pyav",
    ):
        """
        Args:
            data_root: 데이터 루트 폴더 (video와 label 폴더를 포함)
            clip_sampler: 클립 샘플링 전략
            video_sampler: 비디오 샘플러 (사용하지 않음, 호환성을 위해 유지)
            transform: 데이터 변환 함수
            decode_audio: 오디오 디코딩 여부
            decoder: 비디오 디코더 타입
        """
        self.data_root = data_root
        self.video_dir = os.path.join(data_root, "video")
        self.label_dir = os.path.join(data_root, "label")
        
        # 파라미터 저장
        self.clip_sampler = clip_sampler
        self.transform = transform
        self.decode_audio = decode_audio
        self.decoder = decoder
        
        # 비디오 경로 핸들러
        self.video_path_handler = VideoPathHandler()
        
        # 라벨 파일들로부터 데이터 경로와 라벨 정보 추출
        self.video_paths_and_labels = self._load_data_paths_and_labels()
        
        # 클래스 라벨 매핑 생성
        self.class_to_idx = self._create_class_mapping()
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
    
    def _load_data_paths_and_labels(self) -> List[Tuple[str, Dict[str, Any]]]:
        """라벨 파일들로부터 비디오 경로와 라벨 정보를 추출"""
        video_paths_and_labels = []
        
        # 먼저 모든 클래스 라벨을 수집
        all_class_labels = set()
        for label_filename in os.listdir(self.label_dir):
            if not label_filename.endswith('_morpheme.json'):
                continue
                
            label_path = os.path.join(self.label_dir, label_filename)
            with open(label_path, 'r', encoding='utf-8') as f:
                label_data = json.load(f)
                
            if label_data['data']:
                class_label = label_data['data'][0]['attributes'][0]['name']
                all_class_labels.add(class_label)
            else:
                all_class_labels.add('unknown')
        
        # 클래스 라벨을 정렬하고 인덱스 매핑 생성
        sorted_classes = sorted(list(all_class_labels))
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(sorted_classes)}
        
        # 라벨 폴더의 모든 JSON 파일 처리
        for label_filename in os.listdir(self.label_dir):
            if not label_filename.endswith('_morpheme.json'):
                continue
                
            label_path = os.path.join(self.label_dir, label_filename)
            
            with open(label_path, 'r', encoding='utf-8') as f:
                label_data = json.load(f)
            
            # 비디오 파일 이름 추출
            video_name = label_data['metaData']['name']
            video_path = os.path.join(self.video_dir, video_name)
            
            # 비디오 파일이 존재하는지 확인
            if not os.path.exists(video_path):
                print(f"Warning: Video file not found: {video_path}")
                continue
            
            # 라벨 정보 추출 (첫 번째 수어 동작의 라벨 사용)
            if label_data['data']:
                # 첫 번째 attributes의 name을 클래스 라벨로 사용
                class_label = label_data['data'][0]['attributes'][0]['name']
                
                # 시간 정보도 포함
                start_time = label_data['data'][0]['start']
                end_time = label_data['data'][0]['end']
                
                label_info = {
                    'label': class_to_idx[class_label],  # LabeledVideoDataset이 기대하는 형식
                    'class_label': class_label,
                    'start_time': start_time,
                    'end_time': end_time,
                    'video_duration': label_data['metaData']['duration']
                }
            else:
                # 라벨 데이터가 없는 경우 기본값 설정
                label_info = {
                    'label': class_to_idx['unknown'],  # LabeledVideoDataset이 기대하는 형식
                    'class_label': 'unknown',
                    'start_time': 0.0,
                    'end_time': label_data['metaData']['duration'],
                    'video_duration': label_data['metaData']['duration']
                }
            
            video_paths_and_labels.append((video_path, label_info))
        
        # 클래스 매핑도 저장
        self._class_to_idx_temp = class_to_idx
        return video_paths_and_labels
    
    def _create_class_mapping(self) -> Dict[str, int]:
        """클래스 라벨을 인덱스로 매핑하는 딕셔너리 생성"""
        # _load_data_paths_and_labels에서 이미 생성된 매핑 사용
        return self._class_to_idx_temp
    

    
    def __len__(self):
        return len(self.video_paths_and_labels)
    
    def __getitem__(self, idx):
        """인덱스로 데이터 아이템 반환"""
        video_path, label_info = self.video_paths_and_labels[idx]
        
        try:
            # 비디오 로드
            video = self.video_path_handler.video_from_path(
                video_path,
                decode_audio=self.decode_audio,
                decode_video=True,
                decoder=self.decoder,
            )
            
            # 클립 샘플링
            clip_start, clip_end, clip_index, aug_index, is_last_clip = self.clip_sampler(
                last_clip_end_time=None, 
                video_duration=video.duration, 
                annotation=label_info
            )
            
            # 클립 추출
            clip_dict = video.get_clip(clip_start, clip_end)
            
            # 비디오 닫기 (메모리 절약)
            video.close()
            
            if clip_dict is None or clip_dict["video"] is None:
                raise RuntimeError(f"Failed to load clip from {video_path}")
            
            # 출력 딕셔너리 구성
            sample = {
                "video": clip_dict["video"],
                "audio": clip_dict["audio"] if self.decode_audio else None,
                "label": label_info["label"],
                "video_label": label_info["label"],  # 호환성을 위해
                "video_index": idx,
                "clip_index": clip_index,
                "aug_index": aug_index,
                "video_name": os.path.basename(video_path),
            }
            
            # 변환 적용
            if self.transform is not None:
                sample = self.transform(sample)
            
            return sample
            
        except Exception as e:
            # 에러 발생 시 로그 출력하고 다시 발생
            print(f"Error loading video {video_path}: {e}")
            raise
    
    @property
    def num_classes(self):
        """클래스 개수 반환"""
        return len(self.class_to_idx)
    



def sign_language_dataset(
    data_root: str,
    clip_sampler: ClipSampler,
    video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
    transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    decode_audio: bool = True,
    decoder: str = "pyav",
) -> SignLanguageDataset:
    """
    수어 데이터셋 생성 함수
    
    Args:
        data_root: 데이터 루트 폴더 경로
        clip_sampler: 클립 샘플링 전략
        video_sampler: 비디오 샘플러
        transform: 데이터 변환 함수
        decode_audio: 오디오 디코딩 여부
        decoder: 비디오 디코더 타입
        
    Returns:
        SignLanguageDataset 인스턴스
    """
    return SignLanguageDataset(
        data_root=data_root,
        clip_sampler=clip_sampler,
        video_sampler=video_sampler,
        transform=transform,
        decode_audio=decode_audio,
        decoder=decoder,
    ) 