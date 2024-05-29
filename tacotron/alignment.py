from glob import glob
import subprocess
from tqdm import tqdm

wav_files = sorted(glob('./prosodylab_corpus_directory/speaker1/*.wav'))
textgrids = set(sorted(glob('./prosodylab_corpus_directory_aligned/speaker1/*.TextGrid')))
textgrid_name = './prosodylab_corpus_directory_aligned/speaker1/{}.TextGrid'

for wav_file in tqdm(wav_files):
    file_name = wav_file.split('/')[-1].split('.')[0]
    tg_name = textgrid_name.format(file_name)

    if tg_name in textgrids:
        continue

    command = ['mfa', 'align_one', wav_file, f'./prosodylab_corpus_directory/speaker1/{file_name}.lab', './new_dictionary.txt', 'english_us_arpa', './prosodylab_corpus_directory_aligned']
    subprocess.run(command)