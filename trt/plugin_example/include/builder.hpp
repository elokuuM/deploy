#ifndef _BUILDER_HPP
#define _BUILDER_HPP

#include <common.h>
#include <logger.h>

#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include "NvOnnxParser.h"

#include "custom.h"

int builder(char* onnx_file_path);

#endif