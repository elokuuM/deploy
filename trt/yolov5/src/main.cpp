#include <builder.hpp>
#include <infer.hpp>


int main() {
    // buildEngine("yolov5s.onnx");
    inferEngine("./workspace/Debug/yolov5s.trt", "./workspace/Debug/images/");
    return 0;
}