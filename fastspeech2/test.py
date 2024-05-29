import tgt
from g2p_en import G2p

g2p = G2p()
file_name = 'LJ001-0010'

with open(f'./LJSpeech-corpus/speaker1/{file_name}.lab', 'r', encoding='utf-8') as file:
    transcript = file.readline()
    print(' '.join([word for word in g2p(transcript) if word.strip()]))

textgrid = tgt.io.read_textgrid(f'./out/speaker1/{file_name}.TextGrid')

phonemes, duration = [], []
textgrid_object = textgrid.get_tier_by_name("phones")._objects

phonemes = []
for phoneme_information in textgrid_object:
    start_time  = phoneme_information.start_time
    end_time    = phoneme_information.end_time
    phoneme     = phoneme_information.text
    phonemes += [phoneme]

print(' '.join(phonemes))