import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image 
import numpy as np
import socket
import pickle
import time
import logging
import sys
from multiprocessing import Process, Queue, current_process
import multiprocessing
import fcntl
import termios
import ctypes
from algorithm_parecon import algorithm_parecon
from algorithm_parecon_old import algorithm_parecon_old
import scipy.io
import os

# multiprocessing start method
multiprocessing.set_start_method('spawn', True)

# IP information
HOST = '165.246.45.99'
PORT1 = 9999
PORT2 = 8888
ADDR1 = (HOST, PORT1)
ADDR2 = (HOST, PORT2)
buf_size = 4096

logname = './logs/mobile_fixed.log'

end_point = 9
############################

# resizing_number별 프레임 크기 매핑
resizing_map = {
    0: (114,114),
    1: (144,144),
    2: (174,174),
    3: (204,204),
    4: (234,234),
    5: (264,264),
    6: (294,294),
    7: (324,324),
    8: (354,354),
    9: (384,384)
}

def myGetLogger(logname):
    mylogger = logging.getLogger('logger')
    file_handler = logging.FileHandler(logname)
    mylogger.addHandler(file_handler)
    mylogger.setLevel(logging.INFO)
    return mylogger 


def socketOpen(ADDR1, ADDR2):    
    c1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    c1.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    c1.connect((ADDR1))
    time.sleep(1)

    c2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    c2.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    c2.connect((ADDR2))
    print("connected with server")

    return c1, c2


def predict(finalOutput):
    probabilities = torch.nn.functional.softmax(finalOutput[0], dim=0)
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    top5_prob1, top5_catid1 = torch.topk(probabilities, 1)
    print(categories[top5_catid1[0]], top5_prob1[0].item())

def load_images_from_test_folder(base_folder_path="./test"):
    """
    지정된 기본 폴더 내의 1000개 하위 폴더에서 이미지를 로드합니다.
    각 하위 폴더에는 하나의 이미지가 있어야 합니다.
    모든 이미지를 PIL Image 객체 리스트로 메모리에 로드하여 반환합니다.
    """
    images = []
    expected_image_count = 1000
    print(f"Loading images from {base_folder_path}...")

    if not os.path.isdir(base_folder_path):
        print(f"Error: Base folder '{base_folder_path}' not found.")
        return images

    subfolders = sorted(os.listdir(base_folder_path)) # 하위 폴더 정렬

    loaded_count = 0
    for subfolder_name in subfolders:
        if loaded_count >= expected_image_count:
            break # 1000개 이미지 로드 완료
        subfolder_path = os.path.join(base_folder_path, subfolder_name)
        if os.path.isdir(subfolder_path):
            image_file_name = None
            # 하위 폴더 내의 이미지 파일 찾기 (첫 번째 이미지 파일 사용)
            for f_name in sorted(os.listdir(subfolder_path)):
                if f_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    image_file_name = f_name
                    break # 첫 번째 이미지 파일 발견

            if image_file_name:
                image_path = os.path.join(subfolder_path, image_file_name)
                try:
                    img = Image.open(image_path)
                    images.append(img.convert('RGB')) # RGB로 변환하여 일관성 유지
                    loaded_count += 1
                    if loaded_count % 100 == 0:
                        print(f"Loaded {loaded_count}/{expected_image_count} images...")
                except Exception as e:
                    print(f"Error loading image {image_path}: {e}")
            else:
                # 실제로는 각 하위 폴더에 이미지가 있다고 가정하므로, 이 경고는 잘 안나올 수 있음
                print(f"Warning: No image file found in {subfolder_path}")

    if loaded_count < expected_image_count:
        print(f"Warning: Expected {expected_image_count} images, but only loaded {loaded_count}.")
    
    print(f"Total {len(images)} images loaded into memory.")
    return images


def inference(networkQueue, mobileOnlyQueue, totalTimeQueue, mobileTimeQueue, sendTimeQueue, 
              paramQueue, endPointQueue, duration, preloaded_images_list): 
    import mobileEfficientNet2Class 

    point = 4
    resizing_number = 4
    
    if not preloaded_images_list:
        print("Error in inference: No images were provided. Exiting process.")
        return

    num_total_images = len(preloaded_images_list)
    current_image_idx = 0

    def preprocess_input(resizing_number_local, current_pil_image):
        frame_size = resizing_map[resizing_number_local]
        preprocess = transforms.Compose([
            transforms.Resize(frame_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        tensor = preprocess(current_pil_image).unsqueeze(0)
        if torch.cuda.is_available():
            tensor = tensor.to('cuda')
        return tensor
    
    
    
    start_time = time.time()

    i = 0
    
    while True:
        
        current_image = preloaded_images_list[current_image_idx]

        if not paramQueue.empty():
            #current_image = preloaded_images_list[current_image_idx]
            point, resizing_number = paramQueue.get()

        input_batch = preprocess_input(resizing_number, current_image)

        if i != 0: 
            mobile_start = time.time()
            
            if point != 9:
                totalTimeQueue.put(mobile_start)
        
        endPointQueue.put(point)
                    
        with torch.no_grad():
            if point == 0:
                output1 = input_batch
            elif point == 9:
                #endPointQueue.put(9)

                model = mobileEfficientNet2Class.classify(point)
                output1 = model(input_batch)
                predict(output1)
                
                if i != 0:
                    mobile_time = time.time() - mobile_start
                    if not mobileOnlyQueue.full():
                        mobileOnlyQueue.put(mobile_time)
                
                current_image_idx = (current_image_idx + 1) % num_total_images # 다음 이미지로 인덱스 업데이트
                i = i + 1        
                continue
            else:
                model = mobileEfficientNet2Class.classify(point)
                output1 = model(input_batch)


        # Inference end
        send_x = output1.detach().cpu()
        dataInput = [point, send_x]
        
        if i != 0:
            mobile_time = time.time() - mobile_start

            if not mobileTimeQueue.full():
                mobileTimeQueue.put(mobile_time)
                
        if i != 0:
            send_start = time.time()
            
            if not sendTimeQueue.full():
                sendTimeQueue.put(send_start)

        networkQueue.put(dataInput)
        
        current_image_idx = (current_image_idx + 1) % num_total_images # 다음 이미지로 인덱스 업데이트
        i = i + 1


def sendToServer(clientSocket, networkQueue):
    i = 0
    start_time = time.time()
    while True:
        
        if i == 1:
            start_time = time.time()
        current_time = time.time()
        
        if current_time - start_time > 1501:
            break
        
        compData = networkQueue.get()
        
        sendData = pickle.dumps(compData, protocol=pickle.HIGHEST_PROTOCOL)
        clientSocket.send(len(sendData).to_bytes(length=6,byteorder='big'))
        clientSocket.sendall(sendData)    
        
        while True:
            remain_size = ctypes.c_int()
            fcntl.ioctl(clientSocket, termios.TIOCOUTQ, remain_size)
                    
            if remain_size.value == 0:
                break
            time.sleep(0.0001) 
        
        i = i + 1        
 

def receiveFromServer(clientSocket, mobileOnlyQueue, totalTimeQueue, mobileTimeQueue, sendTimeQueue, 
                      fpsLatencyQueue, accQueue, endPointQueue, run_duration, fps_arr, latency_arr):
    #first_data_received = False
    frame_count = 0
    frame_second = 0
    latency_sum = 0.0
    latency_second = 0.0
    
    start_time = time.time()
    last_update = start_time 
    
    mobile_time = 0
    total_latency = 0
    while True:
        
        if frame_count == 1:
            start_time = time.time()
            
        current_time = time.time()
        time1 = time.time()
        if current_time - start_time > run_duration:
            
            temp = 1
            accQueue.put(temp)
            time.sleep(1)
            fps = 1
            avg_latency = 1
            fpsLatencyQueue.put((fps, avg_latency))
            print(frame_count)
            
            end_time = time.time()                
            total_time = end_time - start_time 
            fps = frame_count / total_time 
            print(f"fps: {fps:.1f}")
            
            avg_latency = latency_sum / frame_count
            print(f"Avg Latency: {avg_latency:.4f}")
            break
        
        
        point = endPointQueue.get()
        if point == 9: 
            final_latency = mobileOnlyQueue.get()
        else: 
            lengthData = clientSocket.recv(6)
    
            length = int.from_bytes(lengthData, byteorder='big')
            num_byte = bytes()
            
            send_end = time.time()  
            plus_time = time.time() - time1
            count = 0
            x = 32
            while count < length:
                part = clientSocket.recv(x)
                if not part:
                    break
                num_byte += part 
                count += len(part)
                if length - count < x:
                    x = length - count

            if len(num_byte) > 0:
                finalOutput = pickle.loads(num_byte)
        
            print(finalOutput[0], finalOutput[1]) 
            
            if mobileTimeQueue.empty() or totalTimeQueue.empty() or sendTimeQueue.empty():
                continue
            
            mobile_time = mobileTimeQueue.get()                
            total_start = totalTimeQueue.get()
            total_latency = time.time() - total_start 
                
            send_start = sendTimeQueue.get() 
            send_time = send_end - send_start 
                
            if send_time > mobile_time:
                waste_time = send_time - mobile_time 
                final_latency = total_latency - waste_time + plus_time
            else: 
                final_latency = total_latency
                
        latency_second = latency_second + final_latency 
        latency_sum = latency_sum + final_latency
        frame_second = frame_second + 1
        frame_count = frame_count + 1

        if (time.time() - last_update) >= 1.0:
            if frame_second > 0:
                fps = frame_second / 1.0
                avg_latency = latency_second / frame_second
                fpsLatencyQueue.put((fps, avg_latency))
                
                fps_arr.append(fps)
                latency_arr.append(avg_latency)
                
            print("fps per second", fps)
            print("latency per second", avg_latency)
            last_update = time.time()
            latency_second = 0.0
            frame_second = 0
            

def algorithm(fpsLatencyQueue, paramQueue, accQueue, duration, F_arr, G_arr, H_arr, 
              point_arr, resizing_arr, kappa_arr):
    
    Total_general_info = scipy.io.loadmat('Total_general_info.mat')
    Total_model_info = scipy.io.loadmat('Total_model_info.mat')
    
    V = 20
    Delta_t = 1
    
    # inital value
    Total_queue = np.zeros((duration + 1, 3))  # [queue_F, queue_G, queue_H]
    Total_queue[0, :] = [10, 50, 10]
    
    t_current = 0
    accuracy_sum = 0.0
    
    fps = 30
    resizing_number = 3
    latency_t = 0.08
    accuracy_t = 0.8
    latency_max = 0.07
    accuracy_min = 0.78


    while t_current < duration:

        if not accQueue.empty():
            temp = accQueue.get()
            avg_accuracy = accuracy_sum / t_current
            print(f"Average Accuracy over {t_current} seconds: {avg_accuracy:.4f}")
            break
        
        queue_F = Total_queue[t_current, 0]
        queue_G = Total_queue[t_current, 1]
        queue_H = Total_queue[t_current, 2]

        algorithm_result = algorithm_parecon(V, t_current, Delta_t, Total_queue, Total_general_info,
                                             Total_model_info, mobile_GPU=956250000)

        point = algorithm_result['point']
        kappa_t = algorithm_result['kappa']
        resizing_number = algorithm_result['resizing_number']
        accuracy_t = Total_general_info['Total_general_info'][6,1][resizing_number][0]
        paramQueue.put((point, resizing_number))
        
        #lock 
        fps, latency_t = fpsLatencyQueue.get()
        
        if t_current < len(Total_general_info['Total_general_info'][0, 1]):
            input_frames = Total_general_info['Total_general_info'][0, 1][t_current]
        else:
            input_frames = Total_general_info['Total_general_info'][0, 1][-1]

        if input_frames > 0:
            k = fps / input_frames
        else:
            k = 0
        
        queue_F_next = float(max(0, queue_F - latency_max) + latency_t)
        queue_G_next = float(max(0, queue_G - accuracy_t) + accuracy_min)
        queue_H_next = float(max(0, queue_H - k) + kappa_t)
        
        Total_queue[t_current + 1, :] = [queue_F_next, queue_G_next, queue_H_next]
        F_arr.append(queue_F_next)
        G_arr.append(queue_G_next)
        H_arr.append(queue_H_next)
        point_arr.append(point+1)
        resizing_arr.append(resizing_number+1)
        kappa_arr.append(kappa_t)
        
        print("acc per second ", t_current, accuracy_t)
        print("partition point ", point)
        
        accuracy_sum = accuracy_sum + accuracy_t 
        
        t_current = t_current + 1 


def main():
    c1, c2 = socketOpen(ADDR1, ADDR2)
    duration = 1500
    
    manager = multiprocessing.Manager()
    F_arr               = manager.list()  # 지연시간 큐(F queue)
    G_arr               = manager.list()  # 정확도 큐(G queue)
    H_arr               = manager.list()  # 보조 변수 큐(H queue)
    point_arr           = manager.list()  # partition point
    resizing_arr        = manager.list()  # resizing factor
    kappa_arr           = manager.list()  # kappa
    fps_arr             = manager.list()  # 처리된 fps
    latency_arr         = manager.list()  # 지연시간
    
    # Queues
    networkQueue = Queue(1) 
    
    mobileOnlyQueue = Queue(1) # use when point = 9 calculate 1 frame latency 
    totalTimeQueue = Queue() # use when calculate 1 frame latency 
    mobileTimeQueue = Queue() # use when calculate inference latency 
    sendTimeQueue = Queue() # use when calculate upload latency
    fpsLatencyQueue = Queue(1) # send 1 frame latency and fps per second from recvFromServer to algorithm 
    paramQueue = Queue(1) # send point and resizing number from algorithm to inference 
    accQueue = Queue(1) # qu
    endPointQueue = Queue()
    
    # ---이미지 미리 로드---
    test_folder_main_path = "./test" # test 폴더 경로
    preloaded_images = load_images_from_test_folder(test_folder_main_path)

    if not preloaded_images:
        print("Failed to load images. Please check the './test' folder and its subdirectories.")
        print("Exiting application.")
        c1.close()
        c2.close()
        return

    process1 = Process(target=inference, args=(networkQueue, mobileOnlyQueue, totalTimeQueue, mobileTimeQueue, sendTimeQueue, paramQueue, endPointQueue, duration, preloaded_images))
    process2 = Process(target=sendToServer, args=(c1, networkQueue))
    process3 = Process(target=receiveFromServer, args=(c2, mobileOnlyQueue, totalTimeQueue, mobileTimeQueue, sendTimeQueue, fpsLatencyQueue, accQueue, endPointQueue, duration, fps_arr, latency_arr))
    process4 = Process(target=algorithm, args = (fpsLatencyQueue, paramQueue, accQueue, duration, F_arr, G_arr, H_arr, point_arr, resizing_arr, kappa_arr))

    allProcess = [process4, process3, process2, process1]

    for x in allProcess:
        x.start()

    process4.join()
    process3.join()
    
    # 3) 모든 프로세스 종료 후, 8개의 로그 작성
    with open("./logs/F_queue.log", "w") as f:
        for data in F_arr:
            f.write(str(data) + "\n")

    with open("./logs/G_queue.log", "w") as f:
        for data in G_arr:
            f.write(str(data) + "\n")

    with open("./logs/H_queue.log", "w") as f:
        for data in H_arr:
            f.write(str(data) + "\n")

    with open("./logs/latency.log", "w") as f:
        for data in latency_arr:
            f.write(str(data) + "\n")

    with open("./logs/point.log", "w") as f:
        for data in point_arr:
            f.write(str(data) + "\n")

    with open("./logs/resizing.log", "w") as f:
        for data in resizing_arr:
            f.write(str(data) + "\n")

    with open("./logs/fps.log", "w") as f:
        for data in fps_arr:
            f.write(str(data) + "\n")

    with open("./logs/kappa.log", "w") as f:
        for data in kappa_arr:
            f.write(str(data) + "\n")

    process1.join()
    process2.join()

    for x in allProcess:
        x.close()

    c1.close()
    c2.close()

    
if __name__ == '__main__':
    main()
