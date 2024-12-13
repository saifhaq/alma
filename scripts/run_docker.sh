docker run -itd \
--name alma4 \
--gpus=all \
-v /home:/home \
--entrypoint=/bin/bash \
alma:latest2
