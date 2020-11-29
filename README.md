# landmark_classification

1. data/ 밑에 데이터셋 위치하게 하기
2. src/train.py 를 실행해서 train 진행

  * --image_size: 학습시킬 이미지의 가로, 세로 길이 (default 256)	
  * --epochs: 학습시킬 epoch의 횟수 (default 100)
  * --learning_rate: learning rate희 크기 (default 0.001)
  * --wd: 가중치 감소의 값 (default 0.00001)
  * --batch_size: batch의 크기 (default 64)
  * --train: train 모드인지 test 모드인지 정함 (default True)
  * --load_epoch: test 모드에서 가져올 모델의 학습한 epoch 수 (dafault 29)
