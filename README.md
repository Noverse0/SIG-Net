# SIG-Net

# Dataset 
- KDDCUP 2015 dataset from : http://moocdata.cn/data/user-activity
- NAVER Edwith & Boostcourse dataset from : https://connect.or.kr/view/research_public.html
- Pre-processed dataset in `data` directory. 
- KDDCUP 2015 dataset directory is 'data/kddcup15'
- NAVER Edwith & Boostcourse dataset directory is 'data/naver'
<br />

# Docker Container
- Docker container use sig-net project directory as volume 
- File change will be apply directly to file in docker container

# Experiment 
1. `make up` : build docker image and start docker container
2. `python3 src/main.py` : start experiment in docker container
3. You can change the experiment config by adding a command. For example,`python3 src/main.py  --dataset naver`
4. you can check the result in `experiment_log/`

<br />