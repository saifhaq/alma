docker build \
-f ./runner.Dockerfile \
--network=host \
-t alma ./