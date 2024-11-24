docker build \
-f ./Dockerfile \
--no-cache \
--network=host \
-t torch:2.4 ./