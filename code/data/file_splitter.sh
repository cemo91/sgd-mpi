#!/bin/bash

# USAGE: ./file_splitter.sh -f=FILENAME -s=SPLITCOUNT
# USAGE: ./file_splitter.sh --filename=FILENAME --split=SPLITCOUNT

for i in "$@"
do
case $i in
    -f=*|--filename=*)
    FILENAME="${i#*=}"
    shift # past argument=value
    ;;
    -s=*|--split=*)
    SPLIT="${i#*=}"
    shift # past argument=value
    ;;
esac
done
split -da 1 -l $((`wc -l < ${FILENAME}`/${SPLIT})) ${FILENAME} ${FILENAME}.part