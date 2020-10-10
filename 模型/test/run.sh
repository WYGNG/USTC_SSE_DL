#!/bin/sh

./e02_detection_ssd.bin -offlinemodel model/test_group6.cambricon -output_dir ./ -file_list ./ssd_image_file_list -confidence_threshold 0.6 


