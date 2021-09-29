from typing import Optional
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import time
import os, sys, glob
import time

from stt2text import stt_api, tts_api,sts_api

app = FastAPI()

class Item(BaseModel):
    audio_path : str

class ttsItem(BaseModel):
    text : str

from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse

app = FastAPI()

html = """
<!DOCTYPE html>
<html>
    <head>
        <title>Chat</title>
    </head>
    <body>
        <h1>WebSocket Chat</h1>
        <form action="" onsubmit="sendMessage(event)">
            <input type="text" id="messageText" autocomplete="off"/>
            <button>Send</button>
        </form>
        <ul id='messages'>
        </ul>
        <script>
            var ws = new WebSocket("ws://14.49.44.132:8000/ws");
            ws.onmessage = function(event) {
                var messages = document.getElementById('messages')
                var message = document.createElement('li')
                var content = document.createTextNode(event.data)
                message.appendChild(content)
                messages.appendChild(message)
            };
            function sendMessage(event) {
                var input = document.getElementById("messageText")
                ws.send(input.value)
                input.value = ''
                event.preventDefault()
            }
        </script>
    </body>
</html>
"""


@app.get("/")
async def get():
    return HTMLResponse(html)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        print(data)
        await websocket.send_text(f"Message text was: {data}")



@app.get("/test_audio", response_class=FileResponse)
async def main():
    return some_file_path



@app.post('/items/')
async def create_item(item : Item):
    return item


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
    return str("%.2f" % result)

def wer_rate_jonga(sents,check):
    score = 0
    result = str()
    for i in sents:
        ## compared part
        so = wer(i, check)
        if float(so) > score:
            score = float(so)
            result = i
    if score==0:
        result = i
    return score, result


def _get_script_dict_jonga(script_path):
    with open(script_path, 'r',encoding='utf-8') as f:
        tmp = f.readlines()
    dic_qa = {}
    for i in tmp[1:]:
        qu = i.split(',')[0]
        an = i.split(',')[1].strip()
        if qu in dic_qa:
            if an in dic_qa[qu]:
                pass
            else:
                dic_qa[qu].append(int(an))
        else:
            dic_qa[qu]=[int(an)]
    return dic_qa


def clsfi_ans_jonga(dic_qa, qu):
    score, result = wer_rate_jonga(dic_qa, qu)
    return dic_qa[result]

   

@app.put('/stt_api/')
async def sttapi(item: Item):
    text = stt_api(item.audio_path)
    script_path='/home/ubuntu/ldj/mission1/jonga_script.txt'
    dict_qa = _get_script_dict_jonga(script_path)
    answer = clsfi_ans_jonga(dict_qa, text)
    return answer[0]

@app.put('/tts_api/')
async def ttsapi(item: ttsItem):
    start=time.time()
    text = tts_api(item.text)
    print(time.time()-start)
    return text


@app.put('/sts_api/')
async def sttapi(item: Item):
    return sts_api(item.audio_path)