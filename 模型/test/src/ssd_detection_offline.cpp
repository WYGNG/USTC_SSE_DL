// Copyright [2018] <cambricon>
// This is a demo code for using a SSD model to do detection.
// The code is modified from examples/cpp_classification/classification.cpp.
// Usage:
//    ssd_detect [FLAGS] model_file weights_file list_file
//
// where model_file is the .prototxt file defining the network architecture, and
// weights_file is the .caffemodel file containing the network parameters, and
// list_file contains a list of image files with the format as follows:
//    folder/img1.JPEG
//    folder/img2.JPEG
// list_file can also contain a list of video files with the format as follows:
//    folder/video1.mp4
//    folder/video2.mp4
//
#ifndef USE_OPENCV
#define USE_OPENCV 1
#endif
#ifdef USE_OPENCV
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#endif  // USE_OPENCV
#include <algorithm>
#include <sys/time.h>
#include <iomanip>
#include <iosfwd>
#include <iostream>
#include <fstream>
#include <memory>
#include <string>
#include <map>
#include <utility>
#include <vector>
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "cnrt.h"
#include "yolov2.pb.h"

using namespace cv;
using namespace std;

#ifdef USE_OPENCV

DEFINE_string(offlinemodel, "",
    "The prototxt file used to find net configuration");
DEFINE_string(mean_file, "",
    "The mean file used to subtract from the input image.");
DEFINE_string(mean_value, "0,0,0",
    "If specified, can be one value or can be same as image channels"
    " - would subtract from the corresponding channel). Separated by ','."
    "Either mean_file or mean_value should be provided, not both.");
DEFINE_string(file_type, "image",
    "The file type in the list_file. Currently support image and video.");
DEFINE_string(out_file, "",
    "If provided, store the detection results in the out_file.");
DEFINE_double(confidence_threshold, 0.01,
    "Only store detections with score higher than the threshold.");
DEFINE_string(file_list, "",
    "The input file list");
DEFINE_string(output_dir, "",
    "The directoy used to save output images");
DEFINE_string(label_map_file, "",
    "prototxt with infomation about mapping from label to name");

cv::Scalar HSV2RGB(const float h, const float s, const float v) {
  const int h_i = static_cast<int>(h * 6);
  const float f = h * 6 - h_i;
  const float p = v * (1 - s);
  const float q = v * (1 - f*s);
  const float t = v * (1 - (1 - f) * s);
  float r, g, b;
  switch (h_i) {
    case 0:
      r = v; g = t; b = p;
      break;
    case 1:
      r = q; g = v; b = p;
      break;
    case 2:
      r = p; g = v; b = t;
      break;
    case 3:
      r = p; g = q; b = v;
      break;
    case 4:
      r = t; g = p; b = v;
      break;
    case 5:
      r = v; g = p; b = q;
      break;
    default: 
      r = 1; g = 1; b = 1;
      break;
  }
  return cv::Scalar(r * 255, g * 255, b * 255);
}

// http://martin.ankerl.com/2009/12/09/how-to-create-random-colors-programmatically
vector<cv::Scalar> GetColors(const int n) {
  vector<cv::Scalar> colors;
  cv::RNG rng(12345);
  const float golden_ratio_conjugate = 0.618033988749895;
  const float s = 0.3;
  const float v = 0.99;
  for (int i = 0; i < n; ++i) {
    const float h = std::fmod(rng.uniform(0.f, 1.f) + golden_ratio_conjugate,
                              1.f);
    colors.push_back(HSV2RGB(h, s, v));
  }
  return colors;
}

void WriteVisualizeBBox_offline(const vector<cv::Mat>& images,
                   const vector<vector<vector<float > > > detections,
                   const float threshold, const vector<cv::Scalar>& colors,
                   const map<int, string>& label_to_display_name,
                   const vector<string>& img_names) {
  // Retrieve detections.
  const int num_img = images.size();
  vector< map< int, vector<NormalizedBBox> > > all_detections(num_img);
  for (int i = 0; i < num_img; ++i) {
    for (int j = 0; j < detections[i].size(); j++) {
      const int img_idx = i;
      const int label = detections[i][j][1];
      const float score = detections[i][j][2];
      if (score < threshold) {
        continue;
      }
      NormalizedBBox bbox;
      bbox.set_xmin(detections[i][j][3] *
                    images[i].cols);
      bbox.set_ymin(detections[i][j][4] *
                    images[i].rows);
      bbox.set_xmax(detections[i][j][5] *
                    images[i].cols);
      bbox.set_ymax(detections[i][j][6] *
                    images[i].rows);
      bbox.set_score(score);
      all_detections[img_idx][label].push_back(bbox);
    }
  }

  int fontface = cv::FONT_HERSHEY_SIMPLEX;
  double scale = 1;
  int thickness = 2;
  int baseline = 0;
  char buffer[50];
  for (int i = 0; i < num_img; ++i) {
    cv::Mat image = images[i];
    // Show FPS.
//    snprintf(buffer, sizeof(buffer), "FPS: %.2f", fps);
//    cv::Size text = cv::getTextSize(buffer, fontface, scale, thickness,
//                                    &baseline);
//    cv::rectangle(image, cv::Point(0, 0),
//                  cv::Point(text.width, text.height + baseline),
//                  CV_RGB(255, 255, 255), CV_FILLED);
//    cv::putText(image, buffer, cv::Point(0, text.height + baseline / 2.),
//                fontface, scale, CV_RGB(0, 0, 0), thickness, 8);
    // Draw bboxes.
    std::string name = img_names[i];
    int pos = img_names[i].rfind("/");
    if (pos > 0 && pos < img_names[i].size()) {
      name = name.substr(pos + 1);
    }
    pos = name.rfind(".");
    if (pos > 0 && pos < name.size()) {
      name = name.substr(0, pos);
    }
//    name = name + ".txt";
    name = "output.txt";
//    name = name.substr(4);
    std::ofstream file(name.c_str());
    for (map<int, vector<NormalizedBBox> >::iterator it =
         all_detections[i].begin(); it != all_detections[i].end(); ++it) {
      int label = it->first;
      string label_name = "Unknown";
      if (label_to_display_name.find(label) != label_to_display_name.end()) {
        label_name = label_to_display_name.find(label)->second;
      }
      const cv::Scalar& color = colors[label];
      const vector<NormalizedBBox>& bboxes = it->second;
      for (int j = 0; j < bboxes.size(); ++j) {
        cv::Point top_left_pt(bboxes[j].xmin(), bboxes[j].ymin());
        cv::Point bottom_right_pt(bboxes[j].xmax(), bboxes[j].ymax());
        cv::rectangle(image, top_left_pt, bottom_right_pt, color, 4);
        cv::Point bottom_left_pt(bboxes[j].xmin(), bboxes[j].ymax());
        snprintf(buffer, sizeof(buffer), "%s: %.2f", label_name.c_str(),
                 bboxes[j].score());
        cv::Size text = cv::getTextSize(buffer, fontface, scale, thickness,
                                        &baseline);
        cv::rectangle(
            image, bottom_left_pt + cv::Point(0, 0),
            bottom_left_pt + cv::Point(text.width, -text.height - baseline),
            color, CV_FILLED);
        cv::putText(image, buffer, bottom_left_pt - cv::Point(0, baseline),
                    fontface, scale, CV_RGB(0, 0, 0), thickness, 8);
        file << label_name << " " << bboxes[j].score() << " "
            << bboxes[j].xmin() / image.cols << " "
            << bboxes[j].ymin() / image.rows << " "
            << bboxes[j].xmax() / image.cols
            << " " << bboxes[j].ymax() / image.rows << std::endl;
      }
    }
    file.close();
    cv::imwrite("output.jpg", image);
  }
};

class Detector {
  public:
  Detector(const string& model_file,
           const string& mean_file,
           const string& mean_value);
  // void Detect_img(const std::vector<cv::Mat>& imgs);
  std::vector<std::vector<vector<float> > > \
    Detect(const std::vector<cv::Mat>& imgs);
  std::vector<std::vector<vector<float> > > Get_results(int n);
  int batch_size() { return batch_size_; }
  cv::Size input_geometry() { return input_geometry_; }
  private:
  void SetMean(const string& mean_file, const string& mean_value);
  void WrapInputLayer(std::vector<std::vector<cv::Mat> >* input_imgs);
  void Preprocess(const std::vector<cv::Mat>& imgs,
                  std::vector<std::vector<cv::Mat> >* input_imgs);
  private:
  cv::Size input_geometry_;
  int batch_size_;
  int num_channels_;
  cv::Mat mean_;
  void** inputCpuPtrS;
  void** outputCpuPtrS;
  void** inputMluPtrS;
  void** outputMluPtrS;
  cnrtDataDescArray_t inputDescS, outputDescS;
  cnrtStream_t stream;
  int inputNum, outputNum;
  cnrtFunction_t function;
  unsigned int out_n, out_c, out_h, out_w;
  int out_count;
};
Detector::Detector(const string& model_file,
                   const string& mean_file,
                   const string& mean_value) {
  // offline model
  // 1. init runtime_lib and device
  unsigned dev_num;
  cnrtGetDeviceCount(&dev_num);
  if (dev_num == 0){
    std::cout<<"no device found"<<std::endl;
    exit(-1);
  }
  cnrtDev_t dev;
  cnrtGetDeviceHandle(&dev, 0);
  cnrtSetCurrentDevice(dev);
  // 2. load model and get function
  cnrtModel_t model;
  printf("load file: %s\n", model_file.c_str());
  cnrtLoadModel(&model, model_file.c_str());
  string name="subnet0";
  cnrtCreateFunction(&function);
  cnrtExtractFunction(&function, model, name.c_str());
  cnrtInitFunctionMemory(function, CNRT_FUNC_TYPE_BLOCK);
  // 3. get function's I/O DataDesc
  cnrtGetInputDataDesc (&inputDescS , &inputNum , function);
  cnrtGetOutputDataDesc(&outputDescS, &outputNum, function);
  // 4. allocate I/O data space on CPU memory and prepare Input data
  inputCpuPtrS  = (void**) malloc (sizeof(void*) * inputNum);
  outputCpuPtrS = (void**) malloc (sizeof(void*) * outputNum);
  int in_count;

  std::cout << "input blob num is " << inputNum << std::endl;
  for (int i = 0; i < inputNum; i++) {
    unsigned int in_n, in_c, in_h, in_w;
    cnrtDataDesc_t inputDesc = inputDescS[i];
    cnrtGetHostDataCount(inputDesc, &in_count);
    inputCpuPtrS[i] = (void*)malloc(sizeof(float) * in_count);
    cnrtSetHostDataLayout(inputDesc, CNRT_FLOAT32, CNRT_NCHW);
    cnrtGetDataShape(inputDesc, &in_n, &in_c, &in_h, &in_w);
    std::cout << "shape " << in_n << std::endl;
    std::cout << "shape " << in_c << std::endl;
    std::cout << "shape " << in_h << std::endl;
    std::cout << "shape " << in_w << std::endl;
    if (i == 0) {
      batch_size_ = in_n;
      num_channels_ = in_c;
      input_geometry_ = cv::Size(in_w, in_h);
    } else {
      cnrtGetHostDataCount(inputDesc, &in_count);
      float* data = (float*) inputCpuPtrS[1];
      for (int j = 0; j < in_count; j++) {
        data[j] = 1;
      }
    }
  }

  for (int i = 0; i < outputNum; i++) {
    cnrtDataDesc_t outputDesc = outputDescS[i];
    cnrtSetHostDataLayout(outputDesc, CNRT_FLOAT32, CNRT_NCHW);
    cnrtGetHostDataCount(outputDesc, &out_count);
    cnrtGetDataShape(outputDesc, &out_n, &out_c, &out_h, &out_w);
    outputCpuPtrS[i] = (void*)malloc (sizeof(float) * out_count);
    std::cout << "output shape " << out_n << std::endl;
    std::cout << "output shape " << out_c << std::endl;
    std::cout << "output shape " << out_h << std::endl;
    std::cout << "output shape " << out_w << std::endl;
  }

  // 5. allocate I/O data space on MLU memory and copy Input data
  cnrtMallocByDescArray(&inputMluPtrS ,  inputDescS,  inputNum);
  cnrtMallocByDescArray(&outputMluPtrS, outputDescS, outputNum);
  cnrtCreateStream(&stream);

  /* Load the binaryproto mean file. */
  SetMean(mean_file, mean_value);
}

std::vector<std::vector<vector<float> > > \
    Detector::Detect(const std::vector<cv::Mat>& imgs) {
  std::vector<std::vector<cv::Mat> > input_imgs;
  WrapInputLayer(&input_imgs);
  Preprocess(imgs, &input_imgs);
  float time_use;
  struct timeval tpend, tpstart;
  gettimeofday(&tpstart, NULL);

  std::cout << __LINE__ << "----------ipuMemcpy: copy input-----------" << std::endl;
  cnrtMemcpyByDescArray(inputMluPtrS, inputCpuPtrS, inputDescS,
                        inputNum, CNRT_MEM_TRANS_DIR_HOST2DEV);

  // net_->ForwardPrefilled();
  cnrtDim3_t dim = {1, 1, 1};
  void *param[] = {inputMluPtrS[0],
                   outputMluPtrS[0]};
  cnrtInvokeFunction(function, dim, param,
                     CNRT_FUNC_TYPE_BLOCK, stream, NULL);
  cnrtSyncStream(stream);
  gettimeofday(&tpend, NULL);
  time_use = 1000000 * (tpend.tv_sec - tpstart.tv_sec)
    + tpend.tv_usec - tpstart.tv_usec;
  std::cout << "Forward execution time: "
    << time_use << " us" << std::endl;

  std::cout << __LINE__ << "----------ipuMemcpy-----------" << std::endl;
  cnrtMemcpyByDescArray(outputCpuPtrS, outputMluPtrS, outputDescS,
                        outputNum, CNRT_MEM_TRANS_DIR_DEV2HOST);
  vector<vector<vector<float> > > detections(imgs.size());
  float* result = (float*)outputCpuPtrS[0];
  for (int k = 0; k < out_count / 6; ++k) {
    if (result[0] == 0 && result[1] == 0 &&
        result[2] == 0 && result[3] == 0 &&
        result[4] == 0 && result[5] == 1) {
      // Skip invalid detection.
      result += 6;
      continue;
    }
    int batch = k * 6 / (out_c * out_h * out_w);
    vector<float> detection(7, 0);
    detection[0] = batch;
    detection[1] = result[5];
    detection[2] = result[4];
    detection[3] = result[0];
    detection[4] = result[1];
    detection[5] = result[2];
    detection[6] = result[3];
    detections[batch].push_back(detection);
    result += 6;
  }
  return detections;
}
/* Load the mean file in binaryproto format. */
void Detector::SetMean(const string& mean_file, const string& mean_value) {
  cv::Scalar channel_mean;
  if (!mean_value.empty()) {
    CHECK(mean_file.empty()) <<
      "Cannot specify mean_file and mean_value at the same time";
    stringstream ss(mean_value);
    vector<float> values;
    string item;
    while (getline(ss, item, ',')) {
      float value = std::atof(item.c_str());
      values.push_back(value);
    }
    CHECK(values.size() == 1 || values.size() == num_channels_) <<
      "Specify either 1 mean_value or as many as channels: " << num_channels_;
    std::vector<cv::Mat> channels;
    for (int i = 0; i < num_channels_; ++i) {
      /* Extract an individual channel. */
      cv::Mat channel(input_geometry_.height, input_geometry_.width, CV_32FC1,
          cv::Scalar(values[i]));
      channels.push_back(channel);
    }
    cv::merge(channels, mean_);
  }
}
void Detector::WrapInputLayer(std::vector<std::vector<cv::Mat> >* input_imgs) {
  int width = input_geometry_.width;
  int height = input_geometry_.height;
  float* input_data = (float*)inputCpuPtrS[0];
  for (int i = 0; i < batch_size_; ++i) {
    (*input_imgs).push_back(std::vector<cv::Mat> ());
    for (int j = 0; j < num_channels_; ++j) {
      cv::Mat channel(height, width, CV_32FC1, input_data);
      (*input_imgs)[i].push_back(channel);
      input_data += width * height;
    }
  }
}
void Detector::Preprocess(const std::vector<cv::Mat>& imgs,
    std::vector<std::vector<cv::Mat> >* input_imgs) {
  /* Convert the input image to the input image format of the network. */
  CHECK(imgs.size() == input_imgs->size())
    << "Size of imgs and input_imgs doesn't match";
  for (int i = 0; i < imgs.size(); ++i) {
    cv::Mat sample;
    if (imgs[i].channels() == 3 && num_channels_ == 1)
      cv::cvtColor(imgs[i], sample, cv::COLOR_BGR2GRAY);
    else if (imgs[i].channels() == 4 && num_channels_ == 1)
      cv::cvtColor(imgs[i], sample, cv::COLOR_BGRA2GRAY);
    else if (imgs[i].channels() == 4 && num_channels_ == 3)
      cv::cvtColor(imgs[i], sample, cv::COLOR_BGRA2BGR);
    else if (imgs[i].channels() == 1 && num_channels_ == 3)
      cv::cvtColor(imgs[i], sample, cv::COLOR_GRAY2BGR);
    else
      sample = imgs[i];
    cv::Mat sample_resized;
    if (sample.size() != input_geometry_)
      cv::resize(sample, sample_resized, input_geometry_);
    else
      sample_resized = sample;
    cv::Mat sample_float;
    if (num_channels_ == 3)
      sample_resized.convertTo(sample_float, CV_32FC3);
    else
      sample_resized.convertTo(sample_float, CV_32FC1);
    /* This operation will write the separate BGR planes directly to the
     * input layer of the network because it is wrapped by the cv::Mat
     * objects in input_channels. */
    cv::split(sample_float, (*input_imgs)[i]);
  }
}
int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;
#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif
  gflags::SetUsageMessage("Do detection using SSD mode.\n"
        "Usage:\n"
        "    ssd_detect [FLAGS] model_file list_file\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (argc == 0) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "examples/ssd/ssd_detect");
    return 1;
  }
  map<int, string> label_to_display_name;

  label_to_display_name[0] = "background";
  label_to_display_name[1] = "aeroplane";
  label_to_display_name[2] = "bicycle";
  label_to_display_name[3] = "bird";
  label_to_display_name[4] = "boat";
  label_to_display_name[5] = "bottle";
  label_to_display_name[6] = "bus";
  label_to_display_name[7] = "car";
  label_to_display_name[8] = "cat";
  label_to_display_name[9] = "chair";
  label_to_display_name[10] = "cow";
  label_to_display_name[11] = "diningtable";
  label_to_display_name[12] = "dog";
  label_to_display_name[13] = "horse";
  label_to_display_name[14] = "motorbike";
  label_to_display_name[15] = "person";
  label_to_display_name[16] = "pottedplant";
  label_to_display_name[17] = "sheep";
  label_to_display_name[18] = "sofa";
  label_to_display_name[19] = "train";
  label_to_display_name[20] = "tvmonitor";

  cnrtInit(0);
  // Initialize the network.
  Detector* detector = new Detector(FLAGS_offlinemodel,
                                    FLAGS_mean_file,
                                    FLAGS_mean_value);
  // Set the output mode.
  std::streambuf* buf = std::cout.rdbuf();
  std::ofstream outfile;
  if (!FLAGS_out_file.empty()) {
    outfile.open(FLAGS_out_file.c_str());
    if (outfile.good()) {
      buf = outfile.rdbuf();
    }
  }
  std::ostream out(buf);
  // Process image by batch_size in .prototxt.
  std::ifstream infile(FLAGS_file_list.c_str());
  std::string file;
  std::vector<cv::Mat> imgs;
  std::vector<std::string> img_names;
  std::vector<vector<vector<float> > > multi_detections;
  if (FLAGS_file_type == "image") {
    int image_num = 0;
    std::string line_tmp;
    std::ifstream files_tmp(FLAGS_file_list.c_str(), std::ios::in);
    if (files_tmp.fail()) {
      std::cout << "open " << FLAGS_file_list  << " file fail!" << std::endl;
      return 1;
    } else {
      while (getline(files_tmp, line_tmp)) {
        image_num++;
      }
    }
    files_tmp.close();
    std::cout << "there are " << image_num
      << " figures in " << FLAGS_file_list << std::endl;
    int batches_num = image_num / detector->batch_size();
    float time_use;
    struct timeval tpend, tpstart;
    for ( int iter = 1; (batches_num > 0) && (iter <= batches_num); iter++ ) {
      gettimeofday(&tpstart, NULL);
      for (int i = 0; i < detector->batch_size(); i++) {
        getline(infile, file);
        cv::Mat img = cv::imread(file, -1);
        CHECK(!img.empty()) << "Unable to decode image " << file;

        if (!FLAGS_output_dir.empty()) {
          stringstream ss;
          string out_file;
          int pos = file.find_last_of('/');
          string file_name(file.substr(pos+1));
          ss << FLAGS_output_dir << "ssd_" << file_name;
          ss >> out_file;
          img_names.push_back(out_file);
        }
        imgs.push_back(img);
      }
      std::vector<vector<vector<float> > > \
        multi_detections = detector->Detect(imgs);
      if (!FLAGS_output_dir.empty()) {
        vector<cv::Scalar> colors = GetColors(label_to_display_name.size());
        WriteVisualizeBBox_offline(imgs, multi_detections,
        FLAGS_confidence_threshold, colors, label_to_display_name, img_names);
      }
      for (int img_id = 0; img_id < multi_detections.size(); ++img_id) {
        std::vector<std::vector<float> >& detections = multi_detections[img_id];
        /* Print the detection results. */
        for (int i = 0; i < detections.size(); ++i) {
          const vector<float>& d = detections[i];
          // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
          CHECK_EQ(d.size(), 7);
          const float score = d[2];
          if (score >= FLAGS_confidence_threshold) {
            out << img_names[img_id] << " ";
            out << static_cast<int>(d[1]) << " ";
            out << score << " ";
            out << static_cast<int>(d[3] * imgs[img_id].cols) << " ";
            out << static_cast<int>(d[4] * imgs[img_id].rows) << " ";
            out << static_cast<int>(d[5] * imgs[img_id].cols) << " ";
            out << static_cast<int>(d[6] * imgs[img_id].rows) << std::endl;
          }
        }
      }
      imgs.clear();
      img_names.clear();
      gettimeofday(&tpend, NULL);
      time_use = 1000000 * (tpend.tv_sec - tpstart.tv_sec)
        + tpend.tv_usec - tpstart.tv_usec;
      std::cout << "ssd_detection() execution time: "
        << time_use << " us" << std::endl;
    }
    if ( image_num % detector->batch_size() > 0 ) {
      for ( int i = batches_num * detector->batch_size(); i < image_num; i++ ) {
        getline(infile, file);
        cv::Mat img = cv::imread(file, -1);
        CHECK(!img.empty()) << "Unable to decode image " << file;
        imgs.push_back(img);
        if (!FLAGS_output_dir.empty()) {
          stringstream ss;
          string out_file;
          int pos = file.find_last_of('/');
          string file_name(file.substr(pos+1));
          ss << FLAGS_output_dir << "ssd_" << file_name;
          ss >> out_file;
          img_names.push_back(out_file);
        }
      }
      std::vector<vector<vector<float> > >
        multi_detections = detector->Detect(imgs);
      if (!FLAGS_output_dir.empty()) {
        vector<cv::Scalar> colors = GetColors(label_to_display_name.size());
        WriteVisualizeBBox_offline(imgs, multi_detections,
        FLAGS_confidence_threshold, colors, label_to_display_name, img_names);
      }
      for (int img_id = 0; img_id < multi_detections.size(); ++img_id) {
        std::vector<std::vector<float> >& detections = multi_detections[img_id];
        /* Print the detection results. */
        for (int i = 0; i < detections.size(); ++i) {
          const vector<float>& d = detections[i];
          // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
          CHECK_EQ(d.size(), 7);
          const float score = d[2];
          if (score >= FLAGS_confidence_threshold) {
            out << img_names[img_id] << " ";
            out << static_cast<int>(d[1]) << " ";
            out << score << " ";
            out << static_cast<int>(d[3] * imgs[img_id].cols) << " ";
            out << static_cast<int>(d[4] * imgs[img_id].rows) << " ";
            out << static_cast<int>(d[5] * imgs[img_id].cols) << " ";
            out << static_cast<int>(d[6] * imgs[img_id].rows) << std::endl;
          }
        }
      }
    }
  } else if (FLAGS_file_type == "video") {
    cv::VideoCapture cap(file);
    if (!cap.isOpened()) {
      LOG(FATAL) << "Failed to open video: " << file;
    }
    cv::Mat img;
    int frame_count = 0;
    while (true) {
      bool success = cap.read(img);
      if (!success) {
        LOG(INFO) << "Process " << frame_count << " frames from " << file;
        break;
      }
      CHECK(!img.empty()) << "Error when read frame";
      stringstream ss;
      string s;
      ss << file << "_frame_" << frame_count;
      ss >> s;
      imgs.push_back(img);
      if (!FLAGS_output_dir.empty()) {
        stringstream ss;
        string out_file;
        int pos = file.find_last_of('/');
        string file_name(file.substr(pos+1));
        ss << FLAGS_output_dir << "/ssd_" << file
          << "_frame_" << frame_count << ".jpg";
        ss >> out_file;
        img_names.push_back(out_file);
      }
      if (imgs.size() == detector->batch_size()) {
        vector<vector<vector<float> > >
          multi_detections = detector->Detect(imgs);
        for (int img_id = 0; img_id < multi_detections.size(); ++img_id) {
          std::vector<std::vector<float> >&
            detections = multi_detections[img_id];
          /* Print the detection results. */
          for (int i = 0; i < detections.size(); ++i) {
            const vector<float>& d = detections[i];
            CHECK_EQ(d.size(), 7);
            const float score = d[2];
            if (score >= FLAGS_confidence_threshold) {
              out << img_names[img_id] << "_";
              out << static_cast<int>(d[1]) << " ";
              out << score << " ";
              out << static_cast<int>(d[3] * imgs[img_id].cols) << " ";
              out << static_cast<int>(d[4] * imgs[img_id].rows) << " ";
              out << static_cast<int>(d[5] * imgs[img_id].cols) << " ";
              out << static_cast<int>(d[6] * imgs[img_id].rows) << std::endl;
            }
          }
        }
        imgs.clear();
        img_names.clear();
      }
      ++frame_count;
    }
    if (!imgs.empty()) {
      std::vector<vector<vector<float> > >
        multi_detections = detector->Detect(imgs);
      for (int img_id = 0; img_id < multi_detections.size(); ++img_id) {
        std::vector<std::vector<float> >& detections = multi_detections[img_id];
        /* Print the detection results. */
        for (int i = 0; i < detections.size(); ++i) {
          const vector<float>& d = detections[i];
          // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
          CHECK_EQ(d.size(), 7);
          const float score = d[2];
          if (score >= FLAGS_confidence_threshold) {
            out << img_names[img_id] << "_";
//            out << std::setfill('0') << std::setw(6) << frame_count << " ";
            out << static_cast<int>(d[1]) << " ";
            out << score << " ";
            out << static_cast<int>(d[3] * imgs[img_id].cols) << " ";
            out << static_cast<int>(d[4] * imgs[img_id].rows) << " ";
            out << static_cast<int>(d[5] * imgs[img_id].cols) << " ";
            out << static_cast<int>(d[6] * imgs[img_id].rows) << std::endl;
          }
        }
      }
      imgs.clear();
      img_names.clear();
    }
#if 1
    if (cap.isOpened()) {
      cap.release();
    }
#else
    cap.release();
#endif
  } else {
    LOG(FATAL) << "Unknown file_type: " << FLAGS_file_type;
  }
  delete detector;
  cnrtDestroy();
  return 0;
}
#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV
