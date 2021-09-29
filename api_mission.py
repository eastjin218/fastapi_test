import os, sys, yaml, glob
import time


def tts_api(text):
    sys.path.append(r'/home/ubuntu/ldj/Expressive-FastSpeech2/')
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    from syn_inference import Syn
    conf_name = 'one'
    speaker_id = '101'
    emotion_id = 'happy'
    check = [conf_name, text, speaker_id, emotion_id]
    exe_dir='/home/ubuntu/ldj/Expressive-FastSpeech2'
    preprocess_config = yaml.load(open('{}/config/{}/preprocess.yaml'.format(exe_dir,conf_name),'r'), Loader=yaml.FullLoader)
    model_config = yaml.load(open('{}/config/{}/model.yaml'.format(exe_dir,conf_name),'r'), Loader=yaml.FullLoader)
    train_config = yaml.load(open('{}/config/{}/train.yaml'.format(exe_dir,conf_name),'r'), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)
    element = (text, speaker_id, emotion_id)
    model = Syn(configs, element)
    time = model.inference()
    return 'home/ubuntu/ldj/mission1/audio/%s.wav'%time



def stt_api(audio_path):
    sys.path.append(r'/home/ubuntu/ldj/espnet/egs2/ks_zeroth_korean/asr1/')
    os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    from espnet2.bin.asr_inference import Speech2Text
    import soundfile
    ### fast version ###
    # stt_model = Speech2Text(
    # asr_train_config ='/home/ubuntu/ldj/espnet/egs2/back_zeroth_korean/asr1/exp/asr_train_asr_transformer5_raw_kr_bpe5000/config.yaml',
    # asr_model_file ='/home/ubuntu/ldj/espnet/egs2/back_zeroth_korean/asr1/exp/asr_train_asr_transformer5_raw_kr_bpe5000/valid.acc.best.pth',
    # #lm_train_config='/home/ubuntu/ldj/espnet/egs2/zeroth_korean/asr1/conf/train_lm.yaml',
    # token_type='bpe',
    # bpemodel='/home/ubuntu/ldj/espnet/egs2/back_zeroth_korean/asr1/data/kr_token_list/bpe_unigram5000/bpe.model',
    # device='cuda',
    # batch_size = 0,
    # )
    
    ## acuracy version ##
    stt_model = Speech2Text(
    asr_train_config ='/home/ubuntu/ldj/espnet/egs2/ks_zeroth_korean/asr1/exp/asr_train_asr_transformer1_ddp_raw_bpe/config.yaml',
    asr_model_file ='/home/ubuntu/ldj/espnet/egs2/ks_zeroth_korean/asr1/exp/asr_train_asr_transformer1_ddp_raw_bpe/valid.acc.ave_10best.pth',
    token_type='bpe',
    device='cuda',
    batch_size = 0,
    )
    
    speech, rate = soundfile.read(audio_path)
    assert rate==16000, 'mismatch in sampling rate'
    nbests = stt_model(speech)
    text, *_ = nbests[0]
    return text

## conversion api ###
def editDistance(r, h):
    import numpy
    d = numpy.zeros((len(r)+1, len(h)+1),  dtype=numpy.uint8)
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

def wer(r, h):
    d = editDistance(r, h)
    result = 100 - float(d[len(r)][len(h)]) / len(r) * 100
    return str("%.2f" % result)

def wer_rate(sents,check):
    score = 0
    result = str()
    for i in sents:
        ## compared part
        so = wer(i, check)
        if float(so) > score:
            score = float(so)
            result = i
    return score, result

def _get_script_dict(script_path):
    with open(script_path, 'r',encoding='utf-8') as f:
        tmp = f.readlines()
    dic_qa = {}
    for i in tmp[1:]:
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
    score, result = wer_rate(dic_qa, qu)
    if score >50:
        return random.choice(dic_qa[result])
    else:
        return '그것에 대해서는 생각해보지 못했어'



class ConversationApi():
    def __init__(self):
        self.script_path='/home/ubuntu/ldj/mission1/conv_script.txt'
        self.dict_qa = _get_script_dict(self.script_path)
    def conversation(self, audio_path):
        text = stt_api(audio_path)
        answer = clsfi_ans(self.dict_qa, text)
        check_audio = [os.path.basename(a).split('.')[0] for a in glob.glob('/home/ubuntu/ldj/mission1/audio/*.wav')]
        if answer in check_audio:
            return f'/home/ubuntu/ldj/mission1/audio/{answer}.wav'
        else:
            syn_path = tts_api(answer)
            return syn_path

#start = time.time()
#audio_path = sys.argv[1]
#con = ConversationApi()
#s_path = con.conversation(audio_path)
#print(s_path)
#print(time.time()-start)


#te_dirs = '/home/ubuntu/ldj/Expressive-FastSpeech2/con_script.txt'
#
#with open(te_dirs,'r',encoding='utf-8') as f:
#    tmp  = f.readlines()
#te_list = [a.strip() for a in tmp]
#for i in te_list:
#    tts_api(i)


audio = glob.glob('/home/ubuntu/ldj/te_dataset/*.wav')
for i in audio:
    text = stt_api(i)
    print(text)
    name = os.path.basename(i)
    with open('./test_result.txt', 'a', encoding='utf-8') as f:
        f.write(f'{name}|{text}\n')






    
