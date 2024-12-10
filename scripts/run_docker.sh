docker run -itd \
--name alma \
--gpus=all \
-v /home:/home \
--entrypoint=/bin/bash \
alma
