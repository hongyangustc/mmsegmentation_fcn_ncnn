//
// Created by hy on 2020/9/15.
//

#include "net.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <vector>
#include "iostream"

static const unsigned char colors[19][3] = {
    {128, 64, 128},
    {244, 35, 232},
    {70, 70, 70},
    {102, 102, 156},
    {190, 153, 153},
    {153, 153, 153},
    {250, 170, 30},
    {220, 220, 0},
    {107, 142, 35},
    {152, 251, 152},
    {70, 130, 180},
    {220, 20, 60},
    {255, 0, 0},
    {0, 0, 142},
    {0, 0, 70},
    {0, 60, 100},
    {0, 80, 100},
    {0, 0, 230},
    {119, 11, 32}
};

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }

    const char* imagepath = argv[1];

    cv::Mat img = cv::imread(imagepath, 1);
    if (img.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    ncnn::Net fcn;
    fcn.load_param("fcn_mbv2-sim-opt.param");
    fcn.load_model("fcn_mbv2-sim-opt.bin");

    const int target_width = 512;
    const int target_height = 512;

    int img_w = img.cols;
    int img_h = img.rows;
    std::cout << img_w <<" "<< img_h<< std::endl;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, target_width, target_height);

    const float mean_vals[3] = {123.68f, 116.28f, 103.53f};
    const float norm_vals[3] = {1.0 / 58.40f, 1.0 / 57.12f, 1.0 / 57.38f};
    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = fcn.create_extractor();

    ex.input("input.1", in);
    ncnn::Mat maskmaps;
    ex.extract("581", maskmaps);
    std::cout << maskmaps.c <<" "<< maskmaps.w <<" "<< maskmaps.h << std::endl;

    cv::Mat mask(maskmaps.h, maskmaps.w, CV_8UC1);
    cv::Mat color(maskmaps.h, maskmaps.w, CV_8UC3);

    float *maskmapsdata = (float*)maskmaps.data;
    unsigned char *maskdata = mask.data;
    unsigned char *colordata = color.data;

    for(int i=0; i<maskmaps.h; i++)
        for(int j=0; j<maskmaps.w; j++){
            float tmp = maskmapsdata[0*maskmaps.h*maskmaps.w+i*maskmaps.w+j];
            int maxk = 0;
            for(int k=0; k<maskmaps.c; k++){
                if (tmp < maskmapsdata[k*maskmaps.h*maskmaps.w+i*maskmaps.w+j]) {
                    tmp = maskmapsdata[k*maskmaps.h*maskmaps.w+i*maskmaps.w+j];
                    maxk = k;
                }
            }
            maskdata[i*maskmaps.w + j] = maxk;
            for(int c=0; c<3; c++)
            {
                colordata[i*maskmaps.w*3 + j*3 + 2-c] = colors[maxk][c];
            }
        }

    cv::Mat mask_resize(img_h, img_w, CV_8UC1);
    cv::Mat color_resize(img_h, img_w, CV_8UC3);
    cv::resize(mask, mask_resize, mask_resize.size(), 0 , 0, cv::INTER_NEAREST);
    cv::resize(color, color_resize, mask_resize.size(), 0 , 0, cv::INTER_NEAREST);
    cv::imwrite("mask.png", mask_resize);
    cv::imwrite("color.png", color_resize);
    cv::imshow("mask", mask_resize);
    cv::imshow("color", color_resize);
    cv::waitKey(0);

    return 0;
}


