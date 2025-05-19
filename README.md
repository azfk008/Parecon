논문 실험 시 프로필

모바일 기기: Jetson Xavier NX 

서버: RTX 4090 리눅스 PC 

DNN 모델: EfficientNetV2-S (Pretrained 된 모델을 Pytorch에서 직접 가져와 사용함) 

프로그램 설명 

이미지 분류 모델을 여러 레이어 그룹으로 분리한 뒤, 네트워크 상태 변화에 따라 적절하게 모바일과 서버가 나누어 처리하는 프로그램 입니다. 

알고리즘에 따라 모델 분할 지점과 이미지의 크기를 동시에 조절하며, 1프레임당 지연시간 및 top-1 정확도에 대한 제약 조건을 만족하며 추론 fps를 최대화하는 것을 목적으로 합니다. 


실행 시 유의점: 

실행 시에는 반드시 서버를 먼저 실행한 후 모바일에서 코드를 실행해야 합니다. 
프로그램이 네트워크 상태를 변경할 때 sudo를 쓰기 때문에 처음 시작 시에 본인 기기에 해당하는 비밀번호를 입력해야 할 수 있습니다. 

만약 문의사항이 있을 경우 azfk008@gmail.com으로 연락 주시기 바랍니다. 

1.	모바일 기기에 algorithm_parecon.py, mobile_parecon.py, mobileEfficientNet2Class.py, Total_general_info.mat, Total_model_info.mat, network_setting, samoyed.jpg, imagenet_classes.txt 파일 배치, 서버에 sever_parecon.py, imagenet_classes.txt, serverEfficientNet2Class.py 파일 배치. (imagenet_classes.txt는 같은 파일을 복사하여 배치)
2.	모바일 기기의 파일들이 위치한 곳에 ‘logs’이름의 폴더 생성. 로그 결과가 여기에 저장된다.
3.	모바일에서 network_setting 파일 열기. 차례대로 콘솔 창에 입력해 네트워크 초기 상태 설정. 이걸 입력해야 프로그램이 지속적으로 네트워크 상태를 변경할 수 있다. 
4.	server_parecon.py 및 mobile_parecon.py 상단 부분에 양쪽 다 서버의 IP를 입력한다. 
5.	server_parecon.py 실행. 
6.	mobile_parecon.py에서 제약 조건 및 fps 사이의 파라미터인 V값, 정확도 및 지연시간 제약조건 설정. 
7.	제약 조건을 지키는데 사용되는 가상 큐의 초기값 설정. 
8.	mobile_parecon.py 실행. 
9.	콘솔 창을 통해 1500초 동안 테스트 프로그램이 실행되는 것 확인. 실행이 끝나면 큐 3개, fps, 보조 변수 kappa, latency, partition point, resizing factor 로그 파일이 생성된다. 

※	모바일 기기와 서버의 python 버전은 3.8.10, pytorch 버전은 2.0.0이다. 
※	모바일 기기에 Jetpack을 설치한 이후, 별도로 opencv를 설치해주어야 한다. 


Mobile Device: Jetson Xavier NX
Server: RTX 4090 Linux PC
DNN Model: EfficientNetV2-S (Pretrained model directly imported from Pytorch)

Important Notes for Execution

When running, the server must be executed first, followed by running the code on the mobile device.
Since the program uses sudo to change the network status, you may need to enter the password for your device upon initial startup.

If you have any inquiries, please contact azfk008@gmail.com.
