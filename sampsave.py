import argparse
import torch

from melotron.data_utils import TextMelLoader
from utils import load_filepaths_and_text
from multiprocessing import Pool
from pathlib import Path

def parse_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('-d', '--dataset-path', type=str,
                        default='./', help='Path to dataset')
    parser.add_argument('--wav-files', required=True,
                        type=str, help='Path to filelist with audio paths and text')
    parser.add_argument('--mel-files', required=True,
                        type=str, help='Path to filelist with mel paths and text')
    parser.add_argument('--text-cleaners', nargs='*',
                        default=['english_cleaners'], type=str,
                        help='Type of text cleaners for input text')
    parser.add_argument('--max-wav-value', default=32768.0, type=float,
                        help='Maximum audiowave value')
    parser.add_argument('--sampling-rate', default=22050, type=int,
                        help='Sampling rate')
    parser.add_argument('--filter-length', default=1024, type=int,
                        help='Filter length')
    parser.add_argument('--hop-length', default=256, type=int,
                        help='Hop (stride) length')
    parser.add_argument('--win-length', default=1024, type=int,
                        help='Window length')
    parser.add_argument('--mel-fmin', default=0.0, type=float,
                        help='Minimum mel frequency')
    parser.add_argument('--mel-fmax', default=8000.0, type=float,
                        help='Maximum mel frequency')
    parser.add_argument('--n-mel-channels', default=80, type=int,
                        help='Number of bins in mel-spectrograms')

    return parser
path = 'taco_train/samp/'
def samsave(i):
    global data_loader
    global cnt
    global audiopath_and_text
    audiopath, text, speaker = audiopath_and_text[i]
    text = data_loader.get_text(text)
    mel, f0 = data_loader.get_mel_and_f0(audiopath)
    speaker_id = data_loader.get_speaker_id(speaker)
    torch.save((text, mel, speaker_id, f0), path+Path(audiopath).stem + '.pt')
    if i%1000 == 0:
        print("done", i, "/", cnt)

def sampletrain(dataset_path, audiopaths_and_text, args):

    #melpaths_and_text_list = load_filepaths_and_text(dataset_path, melpaths_and_text)
    audiopaths_and_text_list = load_filepaths_and_text(dataset_path, audiopaths_and_text)

    data_loader = TextMelLoader(dataset_path, audiopaths_and_text, args)

    for i in range(len(audiopaths_and_text_list)):
        if i%100 == 0:
            print("done", i, "/", len(audiopaths_and_text_list))

        #mel = data_loader.get_mel(audiopaths_and_text_list[i][0])
        audiopath, text, speaker = audiopath_and_text[i]
        text = data_loader.get_text(text)
        mel, f0 = data_loader.get_mel_and_f0(audiopath)
        speaker_id = data_loader.get_speaker_id(speaker)
        torch.save(mel, melpaths_and_text_list[i][0])

def main():

    parser = argparse.ArgumentParser(description='PyTorch Tacotron 2 Training')
    parser = parse_args(parser)
    args = parser.parse_args()
    args.load_mel_from_disk = False

    audiopaths_and_text_list = load_filepaths_and_text(args.dataset_path, args.wav_files)

    data_loader = TextMelLoader(args.dataset_path, args.wav_files, args)

    index = range(len(audiopaths_and_text_list))
    cnt = len(audiopaths_and_text_list)


    pool = Pool(processes=-1)
    pool.map(samsave, index)
    #audio2mel(args.dataset_path, args.wav_files, args)

if __name__ == '__main__':
    main()