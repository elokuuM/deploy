#ifndef _NMS_H
#define _NMS_H

#include <string>
#include <vector>
#include <iostream>

#include "NvInfer.h"
#include "common.h"

#include <NvInferRuntimeCommon.h>

using namespace std;

struct Box{
    float left, top, right, bottom, confidence;
    int class_label;

    Box() = default;

    Box(float left, float top, float right, float bottom, float confidence, int class_label)
    :left(left), top(top), right(right), bottom(bottom), confidence(confidence), class_label(class_label){}
};

void decode_kernel_invoker(float* predict, int num_bboxes, int num_classes, float confidence_threshold, float nms_threshold, float* parray, int max_objects);

#endif