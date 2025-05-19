실험 시 프로필

모바일 기기: Jetson Xavier NX 
서버: RTX 4090 리눅스 PC 
DNN 모델: EfficientNetV2-S (Pretrained 된 모델을 Pytorch에서 직접 가져와 사용함) 

실행 시 유의점 
실행 시에는 반드시 서버를 먼저 실행한 후 모바일에서 코드를 실행해야 합니다. 
프로그램이 네트워크 상태를 변경할 때 sudo를 쓰기 때문에 처음 시작 시에 본인 기기에 해당하는 비밀번호를 입력해야 할 수 있습니다. 

만약 문의사항이 있을 경우 azfk008@gmail.com으로 연락 주시기 바랍니다. 


Mobile Device: Jetson Xavier NX
Server: RTX 4090 Linux PC
DNN Model: EfficientNetV2-S (Pretrained model directly imported from Pytorch)

Important Notes for Execution

When running, the server must be executed first, followed by running the code on the mobile device.
Since the program uses sudo to change the network status, you may need to enter the password for your device upon initial startup.

If you have any inquiries, please contact azfk008@gmail.com.
