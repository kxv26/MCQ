#!/bin/bash

if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

. venv/bin/activate

pip3 install —-upgrade pip
pip3 install -r requirements.txt

if [ ! -d "bleurt" ]; then
    git clone https://github.com/google-research/bleurt.git
fi

if [ -d "bleurt" ]; then
    cd bleurt
    pip3 install .
fi

deactivate
