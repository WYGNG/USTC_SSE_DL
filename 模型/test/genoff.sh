#!/bin/sh

PATH_ROOT=../..
PATH_TOOL=${PATH_ROOT}/tools
PATH_LIB=${PATH_ROOT}/lib
PATH_E01=${PATH_ROOT}/expr/test
PATH_MODEL=${PATH_E01}/model/test.fix8.prototxt
PATH_WEIGHTS=${PATH_E01}/model/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${PATH_LIB}/x86

${PATH_TOOL}/caffe genoff -model ${PATH_MODEL} -weights ${PATH_WEIGHTS} \
                          -mname ${PATH_E01}/model/test_group6 -mcore 1H8
