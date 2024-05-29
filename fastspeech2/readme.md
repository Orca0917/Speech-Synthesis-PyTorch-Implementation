## 1. 데이터셋 다운로드

FastSpeech2 모델의 구현에 있어서 사용될 데이터셋은 LJSpeech 데이터셋이다. 단일 화자가 약 48시간 가량 녹음된 음성이 제공되며 실제 논문에서도 학습 및 추론 과정에서 해당 데이터셋을 사용하였다. 최대한 동일한 환경에서 실험을 하고자 LJSpeech 데이터셋을 사용한다. 데이터셋은 [이곳](https://keithito.com/LJ-Speech-Dataset/)에서 다운로드 받을 수 있다.

다운받은 데이터셋은 다음 단계를 위해서 전처리를 진행해야 한다. Montreal-Forced-Aligner 공식 홈페이지에 따르면 전처리를 위해 다음과 같은 데이터 형식을 가져야 한다고 한다. 따라서 먼저 모든 wav파일들을 `~/corpus/speaker1` 내에 위치 시켜줘야 한다.

```text
corpus
|- speaker1
    |- LJ001-0001.wav
    |- LJ001-0001.lab   # 전처리 과정에서 생성!
    |- LJ001-0002.wav
    |- LJ001-0002.lab   # 전처리 과정에서 생성!
    |- ...
```

## 2. 데이터셋 전처리하기

논문에서 정확한 발음을 위해 grapheme으로 작성된 transcript들을 모두 phoneme으로 변경하였다고 했다. 음소로 변환할 때는 논문에서 소개한 것과 같이 박규병님의 `g2p` 패키지를 활용하였다.

```zsh
python mfa_preprocess.py \
    --corpus_path ./LJSpeech-corpus/speaker1 \
    --metadata_path ../LJSpeech-1.1/metadata.csv
```

## 3. Montreal Forced Alinger 사용하기

FastSpeech2 모델은 knowledge distillation 없이 원본 데이터로부터 정답 duration, energy, pitch 정보를 추출해 사용하였다. 논문에서 밝히기를 montreal forced alinger를 사용해 해당 정보들을 추출했다고 한다.

```zsh
# Anaconda 가상환경을 기준으로 합니다.
conda create -n mfa
conda activate mfa

mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa

# 1. pretrained g2p 모델 다운로드
mfa model download g2p english_us_arpa

# 2. LJSpeech 데이터셋의 dictionary 생성
mfa g2p ./LJSpeech-corpus english_us_arpa ./new_dictionary.txt

# 다운로드 완료된 것 검사
# mfa model inspect acoustic english_us_arpa
```

위에서 생성하지 못했던 .lab 확장자를 갖는 텍스트 파일을 생성해야 한다. lab파일은 해당 음성의 transcript 가 적힌 파일로 아래 python 파일의 실행으로 생성해줄 수 있다.

```zsh
python mfa_preprocess.py
```

```zsh
# 주어진 transcript에 대한 발음 생성하기
mfa g2p ./LJSpeech-corpus/ englis_us_arpa ./new_dictionary.txt

# 발음에 대한 textgrid 생성하기 (alignment 생성!)
mfa align ./LJSpeech-corpus/ ./new_dictionary.txt english_us_arpa ./textgrid

# mfa validate ./LJSpeech-corpus english_us_arpa english_us_arpa

# oov 텍스트 재학습
# mfa g2p ./LJSpeech-corpus english_us_arpa ~/MFA/LJSpeech-corpus/oovs_found_english_us_arpa.txt --dictionary_path english_us_arpa

# mfa model add_words english_us_arpa ~/MFA/LJSpeech-corpus/oovs_found_english_us_arpa.txt

# mfa align ./LJSpeech-corpus english_us_arpa english_us_arpa ./out
```



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