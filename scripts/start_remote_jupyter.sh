#!/bin/bash

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
echo Jupyter process id: $PID >> jupyter.logs;
tail -n 3 -f jupyter.logs;

# Recommended: Create a notebook config for jupyter
# jupyter notebook --generate-config
# Line 204 -> IP='0.0.0.0'
# Line 276 -> Enter password after executing 'from notebook.auth import passwd; passwd()'
# LIne 287 -> Port
