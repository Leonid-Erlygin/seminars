docker run -d --shm-size=8g --memory=80g --cpus=40 --user 1005:1005 --name erlygin_seminars --rm -it --init -v /home/l.erlygin/seminars:/app --gpus '"device=0,2"' face-eval bash
