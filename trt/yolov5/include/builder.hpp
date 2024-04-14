#ifndef BUILDER_HPP
#define BUILDER_HPP

#include <vector>
#include <string>
#include <future>
#include <fstream>
#include <string>
#include <iostream>

#include <common.h>
#include <opencv2/opencv.hpp>
#include <logger.h>
#include "NvInfer.h"
#include <cuda_runtime_api.h>
#include "NvOnnxParser.h"
#include "calibrator.h"

int buildEngine(char* onnx_file_path);

#endif

