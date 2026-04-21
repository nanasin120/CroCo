# CroCo
본 프로젝트는 [CroCo: Self-Supervised Pre-training for 3D Vision Tasks by Cross-View Completion]논문을 바탕으로 했음을 알립니다.

# 1. 프로젝트 소개
본 프로젝트는 CroCo아키텍처를 Pytorch를 통해 재현한 프로젝트입니다.
<img width="672" height="224" alt="result_0" src="https://github.com/user-attachments/assets/a1b81b47-e1bf-4eed-bb52-f4e8d8a8fbf9" />
<img width="672" height="224" alt="result_50" src="https://github.com/user-attachments/assets/c8f2d321-cbd4-453d-84f8-04bb2ef37cf7" />
<img width="672" height="224" alt="result_100" src="https://github.com/user-attachments/assets/c4a8b2f4-37c7-4cd8-a5ca-22ce50c0d19b" />
<img width="672" height="224" alt="result_150" src="https://github.com/user-attachments/assets/becac727-e3b9-44df-bc45-f2534c225742" />

위 사진은 0번, 50번, 100번, 150번 학습한 결과입니다.

## Tech Stack
* Environment: Real World (플라스틱 컵)
* Deep Learning : Pytorch (CroCo 모델 설계 및 학습)
* Visualization : PIL (이미지 저장)

# 2. 모델 구현
| 이름 | 기능 |
|---|---|
|  |  | 
|  |  |
|  |  |

<img width="2404" height="8484" alt="image" src="https://github.com/user-attachments/assets/338e02f7-196a-48c2-aae3-ceeb6a4bae3d" />


# 4. 학습
설정한 하이퍼파라미터들입니다.
| 항목 | 설정값 | 비고 |
| --- | --- | --- |
| Epoch | 150 | 최종 학습 횟수 |
| Batch Size | 8 | 4, 8, 16 모두 해본 결과 가장 효율 좋은 배치사이즈입니다. |
| Learning rate | 5e-5 | 초기 학습률 |
| Optimizer | AdamW | 논문의 내용을 따랐습니다. |
| Image Size | 224 x 224 | 논문의 내용을 따랐습니다. 224 % 16 == 0|
