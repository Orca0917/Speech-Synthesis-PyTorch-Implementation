from textgrid import TextGrid
import numpy as np
import tgt

from preprocess import to_phoneme

# TextGrid 파일 로드
textgrid = tgt.io.read_textgrid('./preprocessed/textgrid/LJ001-0002.TextGrid')
textgrid_object = textgrid.get_tier_by_name("phones")._objects


silent_phonemes = ['sil', 'sp', 'spn']
phonemes = []
duration = []

for phoneme_information in textgrid_object:
    start_time  = phoneme_information.start_time
    end_time    = phoneme_information.end_time
    phoneme     = phoneme_information.text
    
    # -- 초기 시작지점 찾기 (앞에 무성 제외)
    if len(phonemes) == 0:
        if phoneme in silent_phonemes:
            continue
        else:
            wav_start_time = start_time
        
    phonemes += [phoneme]

    # -- 음성 종료지점 찾기
    if phoneme not in silent_phonemes:
        wav_end_time = end_time
        last_idx = len(phonemes)

    duration += [
        int(
            round(end_time * 22050 / 256) -
            round(start_time * 22050 / 256)
        )
    ]

phonemes = phonemes[:last_idx]
duration = duration[:last_idx]

print(phonemes)

