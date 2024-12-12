docker run -itd \
--name alma2 \
--gpus=all \
-v /home:/home \
--entrypoint=/bin/bash \
alma
