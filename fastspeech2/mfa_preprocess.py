import os
import argparse
from tqdm import tqdm


def convert_graphemes_to_phonemes(metadata_path, corpus_path):
    lines = open(metadata_path, 'r').readlines()

    for line in tqdm(lines):
        fn, _, transcript = line.strip().split('|')
        open(os.path.join(corpus_path, f'{fn}.lab'), 'w').write(transcript)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert graphemes to phonemes and prepare corpus")
    parser.add_argument('--corpus_path', help="Path to save the .lab files", default='./LJSpeech-corpus/speaker1', type=str)
    parser.add_argument('--metadata_path', help="Path to the metadata CSV file", default='../LJSpeech-1.1/metadata.csv', type=str)
    args = parser.parse_args()

    convert_graphemes_to_phonemes(args.metadata_path, args.corpus_path)
