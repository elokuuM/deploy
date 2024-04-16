#include <infer.hpp>

int main() {
    inferEngine("./workspace/Debug/yolov8x_30p(ppqint8).engine", "./workspace/Debug/images/");
    return 0;
}