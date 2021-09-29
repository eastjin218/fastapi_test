import os, sys, yaml, glob
import time
import soundfile
from joblib import Parallel, delayed, cpu_count
import multiprocessing
import psutil

sys.path.append(r'/home/ubuntu/ldj/Expressive-FastSpeech2/')
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
from syn_inference import Syn
conf_name = 'one'
speaker_id = '101'
emotion_id = 'happy'
exe_dir='/home/ubuntu/ldj/Expressive-FastSpeech2'
preprocess_config = yaml.load(open('{}/config/{}/preprocess.yaml'.format(exe_dir,conf_name),'r'), Loader=yaml.FullLoader)
model_config = yaml.load(open('{}/config/{}/model.yaml'.format(exe_dir,conf_name),'r'), Loader=yaml.FullLoader)
train_config = yaml.load(open('{}/config/{}/train.yaml'.format(exe_dir,conf_name),'r'), Loader=yaml.FullLoader)
configs = (preprocess_config, model_config, train_config)
element = (speaker_id, emotion_id)
tts_model = Syn(configs, element)

from espnet2.bin.asr_inference import Speech2Text
stt_model = Speech2Text(
    asr_train_config ='/home/ubuntu/ldj/espnet/egs2/ks_zeroth_korean/asr1/exp/asr_train_asr_transformer1_ddp_raw_bpe/config.yaml',
    asr_model_file ='/home/ubuntu/ldj/espnet/egs2/ks_zeroth_korean/asr1/exp/asr_train_asr_transformer1_ddp_raw_bpe/valid.acc.ave_10best.pth',
    token_type='bpe',
    device='cuda',
    batch_size = 0,
    )

## conversion api ###
def editDistance(r, h):
    import numpy
    d = numpy.zeros((len(r)+1)*(len(h)+1), dtype=numpy.uint8).reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        d[i][0] = i
    for j in range(len(h)+1):
        d[0][j] = j
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitute = d[i-1][j-1] + 1
                insert = d[i][j-1] + 1
                delete = d[i-1][j] + 1
                d[i][j] = min(substitute, insert, delete)
    return d

def getStepList(r, h, d):
    x = len(r)
    y = len(h)
    list = []
    while True:
        if x == 0 and y == 0: 
            break
        elif x >= 1 and y >= 1 and d[x][y] == d[x-1][y-1] and r[x-1] == h[y-1]: 
            list.append("e")
            x = x - 1
            y = y - 1
        elif y >= 1 and d[x][y] == d[x][y-1]+1:
            list.append("i")
            x = x
            y = y - 1
        elif x >= 1 and y >= 1 and d[x][y] == d[x-1][y-1]+1:
            list.append("s")
            x = x - 1
            y = y - 1
        else:
            list.append("d")
            x = x - 1
            y = y
    return list[::-1]

def wer(r, h):
    d = editDistance(r, h)
    list = getStepList(r, h, d)
    result = 100 - float(d[len(r)][len(h)]) / len(r) * 100
    return str("%.2f" % result), r

def wer_rate_parallel(sents,check):
    score = 0
    result = str()
    tmp=[]
    for i in sents:
        so=wer(i,check)
        tmp.append(so)
    return tmp
    
def wer_rate_p(sents,check):
    score = 0
    result = str()
    with Parallel(n_jobs=cpu_count()-2) as parallel:
        so = parallel(delayed(wer)(i, check) for i in sents)
    return so

def wer_rate(sents,check):
    score = 0
    result = str()
    check_set= [(i, check) for i in sents]
    pool=multiprocessing.Pool(processes=psutil.cpu_count(logical=False)-2)
    so = pool.starmap(wer, check_set)
    print(so)
    return so

def _get_script_dict_mission(script_path):
    with open(script_path, 'r',encoding='utf-8') as f:
        tmp = f.readlines()
    dic_qa = {}
    for i in tmp[1:1000]:
        qu = i.split(',')[0]
        an = i.split(',')[1]
        if qu in dic_qa:
            if an in dic_qa[qu]:
                pass
            else:
                dic_qa[qu].append(an)
        else:
            dic_qa[qu]=[an]
    return dic_qa


def clsfi_ans(dic_qa, qu):
    result= wer_rate_parallel(dic_qa, qu)
    if float(max(result)[0]) >50:
        answer = max(result)[1]
        return random.choice(dic_qa[answer])
    else:
        return '그것에 대해서는 생각해보지 못했어'

def stt_api(audio_path):
    speech, rate = soundfile.read(audio_path)
    assert rate==16000, 'mismatch in sampling rate'
    nbests = stt_model(speech)
    text, *_ = nbests[0]
    return text

def tts_api(text):
    result = tts_model.inference(text)
    return 'home/ubuntu/ldj/mission1/audio/%s.wav'%result

def sts_api(audio_path):
    start = time.time()
    speech, rate = soundfile.read(audio_path)
    nbests = stt_model(speech)
    text, *_ = nbests[0]
    print(text)
    print(time.time()-start)
    script_path='/home/ubuntu/ldj/mission1/conv_script.txt'
    dict_qa = _get_script_dict_mission(script_path)
    answer = clsfi_ans(dict_qa, text)
    print(answer)
    print(time.time()-start)
    check_audio = [os.path.basename(a).split('.')[0] for a in glob.glob('/home/ubuntu/ldj/mission1/audio/*.wav')]
    if answer in check_audio:
        print(time.time()-start)
        return f'/home/ubuntu/ldj/mission1/audio/{answer}.wav'
    else:
        syn_path = tts_api(answer)
        print(time.time()-start)
        return syn_pat












