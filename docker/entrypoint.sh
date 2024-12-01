#!/bin/bash

REPO=$REPO
RUNNER_TOKEN=$RUNNER_TOKEN
LABELS=$LABELS
RUNNER_NAME=$RUNNER_NAME  

cd /home/runner/actions-runner

./config.sh --url https://github.com/${REPO} --token ${RUNNER_TOKEN} \
    --unattended --no-default-labels \
    --labels ${LABELS} --work _work \
    --name ${RUNNER_NAME}

echo "3"
cleanup() {
    echo "Removing runner..."
    ./config.sh remove --unattended --token ${RUNNER_TOKEN}
}

trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM

./run.sh & wait $!