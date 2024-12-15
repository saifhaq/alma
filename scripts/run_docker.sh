docker run -itd \
--name alma5 \
--gpus=all \
--shm-size=16g \
-v /home:/home \
--entrypoint=/bin/bash \
alma:latest2
