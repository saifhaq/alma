docker run -itd \
--name alma3 \
--gpus=all \
-v /home:/home \
--entrypoint=/bin/bash \
alma:latest
