import torch
import socket
import pickle
import time
from multiprocessing import Process, Queue
import multiprocessing
import serverEfficientNet2Class


# multiprocessing 설정
multiprocessing.set_start_method('spawn', True)

# 고정 설정
HOST = '165.246.45.99'
PORT1 = 9999
PORT2 = 8888
ADDR1 = (HOST, PORT1)
ADDR2 = (HOST, PORT2)
BUFFER_SIZE = 4096
RUN_DURATION = 1600  # 1500초 동안 실행


# 소켓 열기
def socketOpen():
    print('Waiting for Jetson NX...')
    s1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s1.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s1.bind(ADDR1)
    s1.listen(1)
    conn1, addr1 = s1.accept()

    s2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s2.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s2.bind(ADDR2)
    s2.listen(1)
    conn2, addr2 = s2.accept()

    print("Connected with mobile.")
    return conn1, conn2


# 추론 결과 예측
def predict(finalOutput):
    probabilities = torch.nn.functional.softmax(finalOutput[0], dim=0)
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    top_prob, top_catid = torch.topk(probabilities, 1)
    return categories[top_catid[0]], top_prob[0].item()


# 데이터 수신
def recvQueue(conn, processQueue):
    while True:

        # 수신 시작
        lengthData = conn.recv(6)
        if not lengthData:
            continue

        length = int.from_bytes(lengthData, byteorder='big')
        num_byte = bytes()
        count = 0
        while count < length:
            part = conn.recv(BUFFER_SIZE)
            num_byte += part
            count += len(part)

        # 수신 데이터 처리
        data = pickle.loads(num_byte)
        processQueue.put(data)
        print("Data received from mobile.")


# 추론 수행
def inference(processQueue, networkQueue):
    while True:
        inferenceData1 = processQueue.get()

        # 추론 시작
        serverTime1 = time.time()
        index = inferenceData1[0]
        finalInput = inferenceData1[1]

        if torch.cuda.is_available():
            finalInput = finalInput.to('cuda')

        model = serverEfficientNet2Class.classify(index)
        output2 = model(finalInput)

        # 예측 결과
        catid, prob = predict(output2)
        networkQueue.put([catid, prob])

        serverTime2 = time.time()
        print(f"Inference completed. Time taken: {serverTime2 - serverTime1:.4f}s")


# 데이터 송신
def sendQueue(conn, networkQueue, run_duration):

    start_time = time.time()
    i = 0
    while True:
        if i == 1:
            start_time = time.time()
        current_time = time.time()

        if current_time - start_time > run_duration:
            print("Data sending process ended.")
            return

        try:
            networkData = networkQueue.get()
        except:
            continue

        sendData = pickle.dumps(networkData, protocol=pickle.HIGHEST_PROTOCOL)
        conn.send(len(sendData).to_bytes(length=6, byteorder='big'))
        conn.sendall(sendData)
        print("Data sent to mobile.")

        i = i + 1

# 메인 함수
def main():
    conn1, conn2 = socketOpen()
    run_duration = 1500

    processQueue = Queue(1)
    networkQueue = Queue(1)

    # 프로세스 생성
    process1 = Process(target=recvQueue, args=(conn1, processQueue))
    process2 = Process(target=inference, args=(processQueue, networkQueue))
    process3 = Process(target=sendQueue, args=(conn2, networkQueue, run_duration))
    allProcess = [process1, process2, process3]

    for x in allProcess:
        x.start()

    print("Processes started.")

    for x in allProcess:
        x.join()

    print("Processes ended.")

    for x in allProcess:
        x.close()

    conn1.close()
    conn2.close()

if __name__ == "__main__":
    main()
