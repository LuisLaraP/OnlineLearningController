#! /bin/bash

: ${VREP_HOME:=$1}
: ${VREP_HOME:=/usr/share/vrep}

cp $VREP_HOME/programming/remoteApiBindings/python/python/vrep.py .
cp $VREP_HOME/programming/remoteApiBindings/python/python/vrepConst.py .
cp $VREP_HOME/programming/remoteApiBindings/lib/lib/Linux/64Bit/remoteApi.so .
