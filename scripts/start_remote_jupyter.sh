#!/usr/bin/env bash

# trap ctrl-c and call ctrl_c()
trap ctrl_c INT

function ctrl_c() {
        echo Jupyter notebook might be still running - process ID: $PID;
        read -p "Do you want to kill the jupyter process? " yn
                case $yn in
                        [Yy]* ) echo "Killing process!"; kill $PID;;
                        [Nn]* ) exit;;
                esac
}

printf "\n\nStart Notebook\n" >> jupyter.logs;
source venv/bin/activate;
jupyter notebook &>> jupyter.logs &
PID=$!;
IP=$(hostname -I | cut -d' ' -f1);
echo My IP: $IP >> jupyter.logs;
echo Jupyter process id: $PID >> jupyter.logs;
if [ "$1" != "dont-follow" ]
then
    tail -n 3 -f jupyter.logs;
else
    echo "USER INPUT: Not Following !"
    tail -n 10 jupyter.logs;
fi
# Recommended: Create a notebook config for jupyter
# jupyter notebook --generate-config
# Line 204 -> IP='0.0.0.0'
# Line 276 -> Enter password after executing 'from notebook.auth import passwd; passwd()'
# LIne 287 -> Port
