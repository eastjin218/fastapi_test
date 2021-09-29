import urllib.request
import glob, os, sys, re

audio_path = glob.glob(r'/home/ubuntu/ldj/test_dataset/*.PCM')
output_path =r'/home/ubuntu/ldj/te_dataset'
os.makedirs(output_path, exist_ok=True)
for j,i in enumerate(audio_path):
    num = str(j).zfill(4)
    os.rename(i, os.path.dirname(i)+'/'+num+'.PCM')
    #os.system(f'ffmpeg -i {os.path.dirname(i)}/{num}.PCM -acodec pcm_s16e -ac 1 -ar 16000 {output_path}/{num}.wav')
    os.system(f'ffmpeg -ar 16000 -ac 1 -f s16le -i {os.path.dirname(i)}/{num}.PCM {output_path}/{num}.wav')

re_audio_path =glob.glob(r'/home/ubuntu/ldj/te_dataset/*.wav')
for path in re_audio_path:
    data ='{"audio_path": "%s"}'%path
    url = 'http://14.49.44.132:8000/stt_api/'
    req = urllib.request.Request(url, data, {'accept':'application/json'}, {'Content-Type':'application/json'})
    f = urllib.request.urlopen(req)
    for x in f:
        print(x)
    f.close()
    break
