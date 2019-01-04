#!/usr/bin/env bash

rm -rf /home/thomas/.jupyter

# Create default config at ~/.jupyter/jupyter_notebook_config.py
jupyter notebook --generate-config

CONFIG=~/.jupyter/jupyter_notebook_config.py

# Line 204 -> IP='0.0.0.0'
#c.NotebookApp.ip = 'localhost'
sed -i -r "s/^#(.+)NotebookApp\.ip = .+$/\1NotebookApp.ip = '0.0.0.0'/" $CONFIG

# Line 287 -> Port
#c.NotebookApp.port = 8888
sed -i -r "s/^#(.+)NotebookApp\.port = .+$/\1NotebookApp.port = 8899/" $CONFIG

# Line 276 -> Enter password after executing 'from notebook.auth import passwd; passwd()'
#c.NotebookApp.password = ''
jupyter notebook password
