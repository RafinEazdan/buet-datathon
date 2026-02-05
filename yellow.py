# %% [markdown]
# # Inference code for YellowKing's model from  DL Sprint 2022
# https://www.kaggle.com/code/sameen53/yellowking-dlsprint-inference

# %%
!cp -r ../input/python-packages2 ./

# %%
!tar xvfz ./python-packages2/jiwer.tgz
!pip install ./jiwer/jiwer-2.3.0-py3-none-any.whl -f ./ --no-index
!tar xvfz ./python-packages2/normalizer.tgz
!pip install ./normalizer/bnunicodenormalizer-0.0.24.tar.gz -f ./ --no-index
!tar xvfz ./python-packages2/pyctcdecode.tgz
!pip install ./pyctcdecode/attrs-22.1.0-py2.py3-none-any.whl -f ./ --no-index --no-deps
!pip install ./pyctcdecode/exceptiongroup-1.0.0rc9-py3-none-any.whl -f ./ --no-index --no-deps
!pip install ./pyctcdecode/hypothesis-6.54.4-py3-none-any.whl -f ./ --no-index --no-deps
!pip install ./pyctcdecode/numpy-1.21.6-cp37-cp37m-manylinux_2_12_x86_64.manylinux2010_x86_64.whl -f ./ --no-index --no-deps
!pip install ./pyctcdecode/pygtrie-2.5.0.tar.gz -f ./ --no-index --no-deps
!pip install ./pyctcdecode/sortedcontainers-2.4.0-py2.py3-none-any.whl -f ./ --no-index --no-deps
!pip install ./pyctcdecode/pyctcdecode-0.4.0-py2.py3-none-any.whl -f ./ --no-index --no-deps

!tar xvfz ./python-packages2/pypikenlm.tgz
!pip install ./pypikenlm/pypi-kenlm-0.1.20220713.tar.gz -f ./ --no-index --no-deps



# %%
import os
import numpy as np
from tqdm.auto import tqdm
from glob import glob
from transformers import AutoFeatureExtractor, pipeline
import pandas as pd
import librosa
import IPython
from datasets import load_metric
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
import torch
import gc
import wave
from scipy.io import wavfile
import scipy.signal as sps
import pyctcdecode

tqdm.pandas()
import warnings
warnings.filterwarnings("ignore")



# %%
# CHANGE ACCORDINGLY - Optimized for memory
BATCH_SIZE = 1  # Reduced to 1 for long-form audio to avoid OOM
CHUNK_LENGTH_S = 30  # Reduced from 112 to 30 seconds
BASE_INPUT_DIR = "/kaggle/input/dl-sprint-4-0-bengali-long-form-speech-recognition/transcription/transcription"
TEST_DIRECTORY = os.path.join(BASE_INPUT_DIR, "test", "audio")
paths = sorted(glob(os.path.join(TEST_DIRECTORY,'*.wav')))

print(f"Found {len(paths)} audio files")

print(f"Batch size: {BATCH_SIZE}, Chunk length: {CHUNK_LENGTH_S}s")print(f"Sample files: {paths[:2]}")

# %%

class CFG:
    my_model_name = '../input/yellowking-dlsprint-model/YellowKing_model'
    processor_name = '../input/yellowking-dlsprint-model/YellowKing_processor'

# %%
from transformers import Wav2Vec2ProcessorWithLM

processor = Wav2Vec2ProcessorWithLM.from_pretrained(CFG.processor_name)


# %%
my_asrLM = pipeline("automatic-speech-recognition", model=CFG.my_model_name ,feature_extractor =processor.feature_extractor, tokenizer= processor.tokenizer,decoder=processor.decoder ,device=0)


# %%
# Test with first audio file
if len(paths) > 0:
    speech, sr = librosa.load(paths[0], sr=processor.feature_extractor.sampling_rate)
    print(f"Loaded audio: {paths[0]}")
    print(f"Shape: {speech.shape}, Sample rate: {sr}")

# %%
# Test inference with reduced chunk size
my_asrLM([speech], chunk_length_s=CHUNK_LENGTH_S, stride_length_s=None)

# %%
my_asrLM

# %% [markdown]
# **Following Sample Submission:**

# %%
class AudioDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths
    def __len__(self):
        return len(self.paths)
    def __getitem__(self,idx):
        speech, sr = librosa.load(self.paths[idx], sr=processor.feature_extractor.sampling_rate) 
#         print(speech.shape)
        return speech

# %%
dataset = AudioDataset(paths)
dataset[0]

# %%
device = 'cuda:0'

# %%
def collate_fn_padd(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    ## get sequence lengths
    lengths = torch.tensor([ t.shape[0] for t in batch ])
    ## padd
    batch = [ torch.Tensor(t) for t in batch ]
    batch = torch.nn.utils.rnn.pad_sequence(batch)
    ## compute mask
    mask = (batch != 0)
    return batch, lengths, mask


# %%
# Reduced batch size and workers to prevent OOM
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_fn_padd)

print(f"DataLoader created with batch_size={BATCH_SIZE}, num_workers=0")

# %%
preds_all = []
total_batches = len(dataloader)

for idx, (batch, lengths, mask) in enumerate(dataloader):
    # Process batch
    preds = my_asrLM(list(batch.numpy().transpose()), chunk_length_s=CHUNK_LENGTH_S, stride_length_s=None)
    preds_all.extend(preds)
    
    # Clear GPU memory periodically
    if (idx + 1) % 10 == 0:
        torch.cuda.empty_cache()
        gc.collect()
    
    # Progress update
    if (idx + 1) % 5 == 0 or (idx + 1) == total_batches:
        print(f"\rProcessed: {idx + 1}/{total_batches} batches", end="")

print(f"\nCompleted processing {len(preds_all)} files")

# Final cleanup
torch.cuda.empty_cache()
gc.collect()

# %%
from bnunicodenormalizer import Normalizer 


bnorm = Normalizer()
def normalize(sen):
    _words = [bnorm(word)['normalized']  for word in sen.split()]
    return " ".join([word for word in _words if word is not None])

def dari(sentence):
    try:
        if sentence[-1]!="।":
            sentence+="।"
    except:
        print(sentence)
    return sentence

# %%
df= pd.DataFrame(
    {
        "filename":[p.split(os.sep)[-1].replace('.wav','') for p in paths],
        "transcript":[p['text']for p in preds_all]
    }
)
df.transcript= df.transcript.apply(lambda x:normalize(x))
df.transcript= df.transcript.apply(lambda x:dari(x))

# %%
df

# %%
df.to_csv("submission.csv", index=False)


# %%



