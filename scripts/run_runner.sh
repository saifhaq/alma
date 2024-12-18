docker run -itd \
--restart=always \
--gpus=all \
--shm-size=16g \
-e RUNNER_TOKEN=$(cat ~/.ssh/alma_runner_token) \
-e REPO=saifhaq/alma \
-e LABELS=titan \
-e RUNNER_NAME=ron \
--name=alma_runner \
alma_runner:latest

# The token can be found in the repository settings under Actions > Add runner
# ""./config.sh --url https://github.com/saifhaq/alma --token TOKEN" THIS_IS_WHAT_YOU_SHOULD_STORE_AT ~/.ssh/alma_runner_token 