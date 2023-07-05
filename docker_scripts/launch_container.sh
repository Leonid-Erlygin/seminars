docker run -d --shm-size=8g --memory=80g --cpus=40 --user 1133:1134 --name erlygin_seminars_new --rm -it --init -v /home/l.erlygin/seminars:/app --gpus all vision_seminars bash
