#include <iostream>
#include <fstream>
#include <stdio.h>
#include <algorithm>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "net.h"

using namespace std;              
//这个函数是官方提供的用于打印输出的tensor
void pretty_print(const ncnn::Mat& m)
{
    for (int q=0; q<m.c; q++)
    {
        const float* ptr = m.channel(q);
        for (int y=0; y<m.h; y++)
        {
            cout<<"----------------------"<<endl;
            for (int x=0; x<m.w; x++)
            {
                printf("%f ", ptr[x]);
            }
            ptr += m.w;
            printf("\n");
        }
        printf("------------------------\n");
    }
}
void visualize(const char* title, const ncnn::Mat& m)
{
    std::vector<cv::Mat> normed_feats(m.c);
 
    for (int i=0; i<m.c; i++)
    {
        cv::Mat tmp(m.h, m.w, CV_32FC1, (void*)(const float*)m.channel(i));
 
        cv::normalize(tmp, normed_feats[i], 0, 255, cv::NORM_MINMAX, CV_8U);
 
        cv::cvtColor(normed_feats[i], normed_feats[i], cv::COLOR_GRAY2BGR);
 
        // check NaN
        for (int y=0; y<m.h; y++)
        {
            const float* tp = tmp.ptr<float>(y);
            uchar* sp = normed_feats[i].ptr<uchar>(y);
            for (int x=0; x<m.w; x++)
            {
                float v = tp[x];
                if (v != v)
                {
                    sp[0] = 0;
                    sp[1] = 0;
                    sp[2] = 255;
                }
 
                sp += 3;
            }
        }
    }
 
    int tw = m.w < 10 ? 32 : m.w < 20 ? 16 : m.w < 40 ? 8 : m.w < 80 ? 4 : m.w < 160 ? 2 : 1;
    int th = (m.c - 1) / tw + 1;
 
    cv::Mat show_map(m.h * th, m.w * tw, CV_8UC3);
    show_map = cv::Scalar(127);
 
    // tile
    for (int i=0; i<m.c; i++)
    {
        int ty = i / tw;
        int tx = i % tw;
 
        normed_feats[i].copyTo(show_map(cv::Rect(tx * m.w, ty * m.h, m.w, m.h)));
    }
 
    cv::resize(show_map, show_map, cv::Size(0,0), 2, 2, cv::INTER_NEAREST);
    cv::imshow(title, show_map);
    cv::waitKey(3000);
}

class detect_result
{
public:
    int classId;
    float confidence;
    cv::Rect_<float> box;
};

void yolopostprocess(const ncnn::Mat &netout, std::vector<detect_result> &results){
    const float confidence_threshold_ =0.25f;
    const float nms_threshold_ = 0.4f;
    cv::Mat det_output(netout.h, netout.w, CV_32FC1);
    memcpy((uchar*)det_output.data, netout.data, netout.w * netout.h * sizeof(float));
    std::vector<cv::Rect> boxes;
    std::vector<int> classIds;
    std::vector<float> confidences;
    for (int i = 0; i < det_output.rows; i++)
    {
        float box_conf = det_output.at<float>(i, 4);
        if (box_conf < nms_threshold_)
        {
            continue;
        }

        cv::Mat classes_confidences = det_output.row(i).colRange(5, 85);
        cv::Point classIdPoint;
        double cls_conf;
        cv::minMaxLoc(classes_confidences, 0, &cls_conf, 0, &classIdPoint);


        if (cls_conf > confidence_threshold_)
        {
            float cx = det_output.at<float>(i, 0);
            float cy = det_output.at<float>(i, 1);
            float ow = det_output.at<float>(i, 2);
            float oh = det_output.at<float>(i, 3);
            int x = static_cast<int>((cx - 0.5 * ow) * 1);
            int y = static_cast<int>((cy - 0.5 * oh) * 1);
            int width = static_cast<int>(ow * 1);
            int height = static_cast<int>(oh * 1);
            cv::Rect box;
            box.x = x;
            box.y = y;
            box.width = width;
            box.height = height;

            boxes.push_back(box);
            classIds.push_back(classIdPoint.x);
            confidences.push_back(cls_conf * box_conf);
        }
    }

    std::vector<int> indexes;
    cv::dnn::NMSBoxes(boxes, confidences, confidence_threshold_, nms_threshold_, indexes);
    for (size_t i = 0; i < indexes.size(); i++)
    {
        detect_result dr;
        int index = indexes[i];
        int idx = classIds[index];
        dr.box = boxes[index];
        dr.classId = idx;
        dr.confidence = confidences[index];
        results.push_back(dr);
    }
}

void draw_frame(cv::Mat & frame, std::vector<detect_result> &results)
{
    static const string class_names[] = {
		"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
		"fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
		"elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
		"skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
		"tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
		"sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
		"potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
		"microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
		"hair drier", "toothbrush"
	};

    for(auto dr : results)
    {
        // cout<< class_names[dr.classId ]<<endl;
        cv::rectangle(frame, dr.box , cv::Scalar(0, 0, 255), 2, 8);
        cv::rectangle(frame, cv::Point(dr.box .tl().x, dr.box .tl().y - 20), cv::Point(dr.box .br().x, dr.box .tl().y), cv::Scalar(255, 0, 0), -1);

        std::string label = cv::format("%.2f", dr.confidence);
        label = class_names[dr.classId ] + ":" + label;

        cv::putText(frame, label, cv::Point(dr.box.x, dr.box.y + 6), 1, 2, cv::Scalar(0, 255, 0),2);

    }
}

std::string Getfilename(std::string &Orgfolder, std::string &Orgfilenames)
{
	char a[20];//保存字符
	int i = 0;
	int n = Orgfilenames.size() - Orgfolder.size();//文件名长度
	for ( i; i < 16; i++)
	{
		a[i] = Orgfilenames[Orgfolder.size() + i];
	}
	a[i] = '\0';//结束标志
	std::string filename = a;//构造String类型数据
	return filename;
}
//main函数模板
int main(){
    std::string folder_path = "./images/"; //path of folder, you can replace "*.*" by "*.jpg" or "*.png"
    std::vector<cv::String> file_names;
    cv::glob(folder_path, file_names);
    ncnn::Net net;
    // 加载网络 ncnn::Extractor需在循环中实例(开新空间)
    // net.opt.num_threads=1;
    net.opt.use_vulkan_compute = true; 
    net.load_param("yolov5s.param");
    net.load_model("yolov5s.bin");

    // for(auto input:net.input_names()){
    //     cout<< "input_name: "<<input << endl;
    // }

    for (int i = 0; i < file_names.size(); i++){
        cv::Mat img = cv::imread(file_names[i], cv::IMREAD_COLOR);
        cv::Mat img2;
        int input_width = 640;//转onnx时指定的输入大小
        int input_height = 640;
        // resize
        cv::resize(img, img2, cv::Size(input_width, input_height));


        // 把opencv的mat转换成ncnn的mat
        ncnn::Mat input = ncnn::Mat::from_pixels(img2.data, ncnn::Mat::PIXEL_BGR, img2.cols, img2.rows);
        const float norm_vals[3] = { 1 / 255.f, 1 / 255.f, 1 / 255.f };
        input.substract_mean_normalize(0, norm_vals);
        // visualize("img",input);
        // ncnn前向计算
        // cout<< "C:"<< input.c<<" H:"<<input.h<<" W:"<<input.w<<endl;
        ncnn::Extractor extractor = net.create_extractor(); 
        // extractor.input("in0", input);
        extractor.input(net.input_names()[0], input);
        ncnn::Mat output;
        // extractor.extract("out0", output);
        extractor.extract(net.output_names()[0], output);
        // pretty_print(output);
        // cout<< "C:"<< output.c<<" H:"<<output.h<<" W:"<<output.w<<endl;
        // postprocess
        std::vector<detect_result> results;
        yolopostprocess(output, results);
        draw_frame(img2, results);
        cout<<"["<<i+1<<"/"<<file_names.size()<<"] "<<"./pred/" + Getfilename(folder_path, file_names[i])<<endl;
        cv::imwrite("./pred/" + Getfilename(folder_path, file_names[i]), img2);
        // cv::imshow("YOLOv5-7.0", img2);
        // cv::waitKey(0);
    }
    cout<<"done"<<endl;
    return 0;
}