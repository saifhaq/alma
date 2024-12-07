docker run -itd \
--name torch2 \
--gpus=all \
-v /home:/home \
--entrypoint=/bin/bash \
torch2.5:0
