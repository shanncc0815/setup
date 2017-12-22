# Development Environment Test Scripts

### check if nvidia driver is properly installed on the base OS
nvidia-smi 

### check if python 2.7 is properly installed, and test with "Hello World!"
python -V
python helloworld.py 

### check if python 3.5.2 is properly installed, and test with "Hello World!"
python3 -V
python3 helloworld.py

### check available dockers on the base OS 
docker images -a

### connect to the v6.0 docker image
nvidia-docker run -it --rm nvidia/cuda bash

### check if nvidia driver is properly installed on the v6.0 docker - Ubuntu 16.04
nvidia-smi

### check if python 2.7 is properly installed, and test with "Hello World!"
python -V
python helloworld.py

### check if python 3.5.2 is properly installed, and test with "Hello World!"
python3 -V
python3 helloworld.py

### check if tensorflow is properly installed, and test with addition
python tf-add-test1.py      

### check if tensorflow tf.matmul is working properly
python tf-matmul-test2.py

### run tf.matmul 8192 x 8192 matrices 200 times, and monitor GPU utilization
python gpu-bench.py 200             

### test Multi-GPU using https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py
python cifar10_multi_gpu_train.py --max_steps 1000 --num_gpus 1


