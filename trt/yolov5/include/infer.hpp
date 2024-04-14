#ifndef INFER_HPP
#define INFER_HPP

#include <vector>
#include <string>
#include <future>
#include <fstream>
#include <string>
#include <iostream>

#include <common.h>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <logger.h>
#include "NvInfer.h"
#include <cuda_runtime_api.h>
#include "NvOnnxParser.h"
using namespace std;
using namespace sample;

int inferEngine(char* ege_file_path, string input_path);
void doInference(IExecutionContext* context, float* input, float* output);
void preprocess(cv::Mat src, float *data);

class detect_result
{
public:
    int classId;
    float confidence;
    cv::Rect_<float> box;
};
void draw_frame(cv::Mat & frame, std::vector<detect_result> &results);
void yolopostprocess(float* netout, std::vector<detect_result> &results);
std::string Getfilename(std::string &Orgfolder, cv::String &Orgfilenames);
#endif