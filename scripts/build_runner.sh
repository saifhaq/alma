docker build \
-f ./runner.Dockerfile \
--network=host \
-t torch2.4-runner:1 ./