#!/bin/sh

PATH_ROOT=../..
PATH_TOOL=${PATH_ROOT}/tools
PATH_LIB=${PATH_ROOT}/lib
PATH_INI=${PATH_ROOT}/expr/test/model

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${PATH_LIB}/x86

${PATH_TOOL}/generate_fix8_pt -ini_file ${PATH_INI}/convert_fix8.ini
