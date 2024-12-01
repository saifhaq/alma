docker run -itd \
--name torch \
--gpus=all \
-v /home:/home \
--entrypoint=/bin/bash \
torch2.4:1
