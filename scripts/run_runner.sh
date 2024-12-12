docker run -itd \
--restart=always \
--gpus=all \
-e RUNNER_TOKEN=$(cat ~/.ssh/alma_runner_token) \
-e REPO=saifhaq/alma \
-e LABELS=titan \
--name=torch_runner \
alma-runner:latest
