class Config:
    # default
    base_path = '../LJSpeech-1.1'
    preprocess_base_path = base_path + '/preprocessed'

    # preprocess
    wav_paths = preprocess_base_path + '/paths'
    mel_paths = preprocess_base_path + '/mels'
    spec_paths = preprocess_base_path + '/specs'
    transcript_paths = preprocess_base_path + '/transcripts'
    phoneme_paths = preprocess_base_path + '/phonemes'

    # metadata
    data_path = base_path + '/wavs'
    metadata_path = base_path + '/metadata.csv'

    # train
    batch_size = 16
    num_phonemes = 70
    num_mels = 80
    embedding_dim = 512