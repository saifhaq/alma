docker run -itd \
--name alma \
--gpus=all \
--shm-size=16g \
-v /home:/home \
--entrypoint=/bin/bash \
alma:latest
