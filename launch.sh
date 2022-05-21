#/bin/bash

source ./envsetup.sh

date >> log.txt
python3 -m discord.main
