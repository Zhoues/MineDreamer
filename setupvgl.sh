#!/bin/bash -e

export PATH="${PATH}:/opt/VirtualGL/bin"
export LD_LIBRARY_PATH="/usr/lib/libreoffice/program:${LD_LIBRARY_PATH}"

export DISPLAY=":0"
Xvfb "${DISPLAY}" -ac -screen "0" "1920x1200x24" -dpi "72" +extension "RANDR" +extension "GLX" +iglx +extension "MIT-SHM" +render -nolisten "tcp" -noreset -shmem &

# Wait for X11 to start
echo "Waiting for X socket"
until [ -S "/tmp/.X11-unix/X${DISPLAY/:/}" ]; do sleep 1; done
echo "X socket is ready"

export VGL_DISPLAY="egl"
export VGL_REFRESHRATE="$REFRESH"

echo "Session Running."

"$@"
