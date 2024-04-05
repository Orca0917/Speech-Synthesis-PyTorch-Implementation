### 데이터셋 다운로드

FastSpeech2 모델의 구현에 있어서 사용될 데이터셋은 LJSpeech 데이터셋이다. 단일 화자가 약 48시간 가량 녹음된 음성이 제공되며 실제 논문에서도 학습 및 추론 과정에서 해당 데이터셋을 사용하였다. 최대한 동일한 환경에서 실험을 하고자 LJSpeech 데이터셋을 사용한다. 데이터셋은 [이곳](https://keithito.com/LJ-Speech-Dataset/)에서 다운로드 받을 수 있다.

### TextGrid 다운로드

직접 Montreal Forced Aligner (MFA)를 사용하여 데이터를 준비할 수 있지만, 시간 절약과 환경설정의 불편함을 해소하고자 이미 만들어진 TextGrid를 사용한다. [이곳](https://drive.google.com/drive/folders/1DBRkALpPd6FL9gjHMmMEdHODmkgNIIK4)에서 LJSpeech 데이터셋에 대한 TextGrid를 다운받을 수 있다.

이 TextGrid는 사용하려고 하는 데이터셋의 가장 기본이 되기 때문에 정확하게 준비해주는 것이 중요하다. 모든 전처리 과정이 이 TextGrid를 기반으로 진행이 된다!!




### 1. 데이터 전처리

```py
python preprocess.py \
    --path='../LJSpeech-1.1'    # path 경로를 입력
```

### 2. 학습

```py
python train.py
```