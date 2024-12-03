import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import datetime
import os
import sys
from sys import platform
import time
import queue
import requests
import torch

from dfb.model.wdcnn import * 

# 모델 추론 결과값과 데이터 특성값을 보낼 서버의 ip
server_ip = '172.16.63.190'

# 서버에 보낼 데이터의 상단 값 설정
# 보낼 데이터의 헤더와 파라미터를 서버에서 설정해놓은 값과 맞춰서 설정
headers = {
        'Content-Type': 'text/plain',
}
params = {
        'k': 'testkey2023',
        'i': 'device01',
}

# 모델 불러오기 (학습이 완료된 모델 파일 경로는 수동 입력)
model_path = "results/model.pth" 
model = WDCNN(n_classes=4)
state_dict = torch.load(model_path, map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
    
num_classes = 4

eu_sen = np.array([100.0, 100.0]) # digital signal conditioner 사용하는 경우, sensitivity 정의
eu_units = ["g", "g"] # 가속도 단위
blocksize = 2048 # 한 번에 획득할 데이터 길이 설정
samplerate = 16000 # 48000, 44100, 32000, 22100, 16000, 11050, 8000

# TMSFindDevices
#
# returns an array of TMS compatible devices with associated information
#
# Returns dictionary items per device
#   "device"        - Device number to be used by SoundDevice stream
#   "model"         - Model number
#   "serial_number" - Serial number
#   "date"          - Calibration date
#   "format"        - format of data from device, 0 - acceleration, 1 - voltage
#   "sensitivity_int - Raw sensitivity as integer counts/EU ie Volta or m/s^2
#   "scale"         - sensitiivty scaled to float for use with a
#                     -1.0 to 1.0 scaled data.  Format returned with
#                     'float32' format to SoundDevice stream.

# 연관된 정보가 있는 TMS 장치 배열을 반환
def TMSFindDevices():
    models=["485B", "333D", "633A", "SDC0"] # 이 중에서 333D를 사용

    # window의 경우는 신호 접근하기위한 API가 많으므로 이 중에서 특정 API를 선택
    if platform == "win32":         # Windows...
        hapis=sd.query_hostapis()
        api_num=0
        for api in hapis:
            if api['name'] == "Windows WDM-KS":
                break
            api_num += 1
    else:
        api_num=0
        
    # devices: 접근 가능한 audio input들 
    devices = sd.query_devices()
    dev_info = []   # Array to store info about each compa
    dev_num=0
    # TMS 모델로 명명된 장치를 찾기 (동일한 장치더라도 여러 인스턴스 반환 - 사용 가능한 오디오 API가 다름)
    for device in devices:
        if (device['hostapi'] == api_num):
            name = device['name']
            match = next((x for x in models if x in name), False)
            if match != False:
                loc = name.find(match)
                model = name[loc:loc+6] # 모델명 추출
                fmt = name[loc+7:loc+8] # 데이터 형식 추출
                serialnum = name[loc+8:loc+14]  # 시리얼 넘버 추출
                
                # 데이터가 voltage 형식인 경우
                if fmt == "2" or fmt == '3':
                    form = 1    # Voltage
                    # sensitivity 추출
                    sens = [int(name[loc+14:loc+21]), int(name[loc+21:loc+28])]
                    if fmt == "3":  # 50mV reference for format 3
                        sens[0] *= 20 # Convert to 1V reference
                        sens[1] *= 20 
                    scale = np.array([8388608.0/sens[0],
                                      8388608.0/sens[1]],
                                     dtype='float32') # scale to volts
                    date = datetime.datetime.strptime(name[loc+28:loc+34], '%y%m%d') # Isolate the calibration date from the fullname string
                
                # 가속도 형식의 데이터를 수집하는 장치인 경우
                elif fmt == "1":
                    
                    form = 0
                    # sensitivity 추출
                    sens = [int(name[loc+14:loc+19]), int(name[loc+19:loc+24])]
                    scale = np.array([855400.0/sens[0],
                                      855400.0/sens[1]],
                                      dtype='float32') # scale to g's
                    date = datetime.datetime.strptime(name[loc+24:loc+30], '%y%m%d') # Isolate the calibration date from the fullname string
                else:
                      raise FormatError("Expecting 1, 2, or 3 format")
                 
                # 장치 정보를 dev_info에 추가 
                dev_info.append({"device":dev_num,\
                                 "model":model,\
                                 "serial_number":serialnum,\
                                 "date":date,\
                                 "format":form,\
                                 "sensitivity_int":sens,\
                                 "scale":scale,\
                                 })
        dev_num += 1
    if len(dev_info) == 0:
        raise NoDevicesFound("No compatible devices found")
    return dev_info

# sounddevice 는 PortAudio에서 콜백 활용하여 처리 지연 시간을 줄임
# 이 콜백은 다른 스레드에서 호출되며, 대기열을 사용하여 데이터를 주 처리 스레드로 전송
def callback(indata, frames, time, status):
    if status:
        print(status)
    q.put(indata[:,:])  # Place data in queue

# ms 지연에 사용되는 함수
def time_ms():
    return int(time.monotonic_ns()/1000000)

# 장치 찾기
info=TMSFindDevices()

# 첫번째로 찾은 장치를 사용
if len(info) > 1:
    print("Using first device found.")
dev=0   

# 데이터 스케일링 결정
units = ["Volts", "Volts"]
scale=info[dev]['scale']
if info[dev]['format'] == 1: # voltage 데이터
    for ch in range(len(scale)):
        if eu_sen[ch] != 0.0:
            scale[ch] *= 1.0 / (eu_sen[ch]/1000.0)
            units[ch] = eu_units[ch]
elif info[dev]['format'] == 0: # 가속도 데이터
    units = ["g", "g"]

# q를 사용하여 콜백에서 데이터를 가져옴 (콜백은 다른 스레드)
q = queue.Queue() 

# 시간 축을 정의하여 예시 플롯을 설정
x=np.linspace(0, (blocksize-1)/samplerate, blocksize)

# 입력스트림을 열어 오디오 데이터를 처리
stream = sd.InputStream(
        device=info[dev]['device'], channels=2,
        samplerate=samplerate, dtype='float32', blocksize=blocksize,
        callback=callback)

class_confidence_scores = [0] * num_classes
        
# 모델 torchscript 형식으로 변환
model = torch.jit.script(model)

model.eval()

# 모델 1000회 warmup
input_data = torch.rand(1, 1, 2048)  # 배치 크기 1, 채널 1, 길이 2048
with torch.no_grad():
    for _ in range(1000):  # 1000회 웜업
        _ = model(input_data)

# 입력 스트림 실행하여 데이터 q에 저장
stream.start()

for i in range(200):
    with torch.no_grad():
    
        data = q.get()
        # 데이터 받아올 때의 시간 기록
        stime = datetime.datetime.utcnow().isoformat()
        
        sdata = data * scale # 데이터 스케일링
        sdata = torch.FloatTensor(sdata[:,0].reshape(1,1,2048)) # 모델 입력 크기 맞춰주기
        
        # 추론
        outputs = model(sdata)
        
        # 클래스의 확률
        probabilities = torch.softmax(outputs, dim=1)
        # 가장 높은 확률을 가진 클래스와 그 확률 값
        confidence_scores, predicted_classes = torch.max(probabilities, dim=1)
        
    # 서버에 보낼 데이터
    sending_data = ""
    sending_data += "c1"+"|"+str(torch.max(sdata[0][0]).item())+"|c2|"+str(torch.min(sdata[0][0]).item())+"|c3|"+\
        str(torch.sqrt(torch.mean(torch.square(sdata[0][0]))).item())+"|c4|"+str(probabilities[0][0].item())+\
            "|c5|"+str(probabilities[0][1].item())+"|c6|"+str(probabilities[0][2].item())+\
            "|c7|"+str(predicted_classes.item())+"|c8|"+str(confidence_scores.item())+\
            "|ts|"+str(stime) + "|c9|"+str(probabilities[0][3].item())
    print(f'[{i}] {sending_data}')
    time.sleep(0.5)
    response = requests.post(f'http://{server_ip}:7896/iot/d', params=params, headers=headers, data=sending_data)
    

# 스트림 중단
stream.stop()
