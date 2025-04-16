#!/bin/bash

set -e
set -x

pip install gdown

gdown "1t2BjJtsejSIUZI54PKObMFG6_wMMG3bC&confirm=t" -O ./datasets/QA.zip
unzip -o ./datasets/QA.zip -d ./datasets/QA -x '*.DS_Store'

mv ./datasets/QA/processed_data/* ./datasets/QA/

rm -rf ./datasets/QA/processed_data
rm -f ./dataset/QA.zip
