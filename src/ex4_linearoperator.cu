// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// ########################################################################

#include <iostream>
#include <string>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "helper.cuh"
#include "gradient.cuh"
#include "divergence.cuh"
#include "norm.cuh"


int main(int argc,char **argv)
{
    // parse command line parameters
    const char *params = {
        "{i|image| |input image}"
        "{b|bw|false|load input image as grayscale/black-white}"
        "{r|repeats|1|number of computation repetitions}"
    };
    cv::CommandLineParser cmd(argc, argv, params);

    // input image
    std::string inputImage = cmd.get<std::string>("image");
    // number of computation repetitions to get a better run time measurement
    size_t repeats = (size_t)cmd.get<int>("repeats");
    // load the input image as grayscale
    bool gray = cmd.get<bool>("bw");

    // init camera
    bool useCam = inputImage.empty();
    cv::VideoCapture camera;
    if (useCam && !openCamera(camera, 0))
    {
        std::cerr << "ERROR: Could not open camera" << std::endl;
        return 1;
    }

    // read input frame
    cv::Mat mIn;
    if (useCam)
    {
        // read in first frame to get the dimensions
        camera >> mIn;
    }
    else
    {
        // load the input image using opencv (load as grayscale if "gray==true", otherwise as is (may be color or grayscale))
        mIn = cv::imread(inputImage.c_str(), (gray ? CV_LOAD_IMAGE_GRAYSCALE : -1));
    }
    // check
    if (mIn.empty())
    {
        std::cerr << "ERROR: Could not retrieve frame " << inputImage << std::endl;
        return 1;
    }
    // convert to float representation (opencv loads image values as single bytes by default)
    mIn.convertTo(mIn, CV_32F);

    // get image dimensions
    int w = mIn.cols;         // width
    int h = mIn.rows;         // height
    int nc = mIn.channels();  // number of channels
    std::cout << "Image: " << w << " x " << h << std::endl;

    // initialize CUDA context
    cudaDeviceSynchronize();  CUDA_CHECK;

    // ### Set the output image format
    cv::Mat mOut_lapNorm(h,w,CV_32FC1);    // grayscale, 1 layer
    cv::Mat mOut_u(h,w,mIn.type());    // rgb, 1 layer
    cv::Mat mOut_v(h,w,mIn.type());    // rgb, 1 layer
    cv::Mat mOut_w(h,w,mIn.type());    // rgb, 1 layer

    // ### Allocate arrays
    // allocate raw input image array
    float *imgIn = NULL;    // TODO allocate array
    // allocate raw output array (the computation result will be stored in this array, then later converted to mOut for displaying)
    float *imgOut_lapNorm = NULL;   // TODO allocate array
    float *imgOut_u = NULL;         // TODO allocate array
    float *imgOut_v = NULL;         // TODO allocate array
    float *imgOut_w = NULL;         // TODO allocate array

    // allocate arrays on GPU
    float *d_imgIn = NULL;
    float *d_lapNorm = NULL;
    float *d_u = NULL;
    float *d_v = NULL;
    float *d_w = NULL;
    // TODO alloc cuda memory for device arrays

    do
    {
        // convert range of each channel to [0,1]
        mIn /= 255.0f;

        // init raw input image array (and convert to layered)
        convertMatToLayered (imgIn, mIn);
        // upload to GPU
        // TODO copy from imgIn to d_imgIn

        Timer timer;
        timer.start();
        for(size_t i = 0; i < repeats; ++i)
        {
            // TODO (4.1) implement computeGradientCuda() in gradient.cu
            computeGradientCuda(d_u, d_v, d_imgIn, w, h, nc);
            cudaDeviceSynchronize();

            // TODO (4.2) implement computeDivergenceCuda() in divergence.cu
            computeDivergenceCuda(d_w, d_u, d_v, w, h, nc);
            cudaDeviceSynchronize();

            // TODO (4.3) implement computeNormCuda() in norm.cu
            computeNormCuda(d_lapNorm, d_w, w, h, nc);
            cudaDeviceSynchronize();
        }
        timer.end();
        float t = timer.get()/repeats;
        std::cout << "average time: " << t*1000 << " ms" << std::endl;

        // copy back to CPU
        // TODO download from device arrays to host arrays

        // show input image
        showImage("Input", mIn, 100, 100);  // show at position (x_from_left=100,y_from_above=100)

        // show output image: first convert to interleaved opencv format from the layered raw array
        // TODO (4.4) show gradient, divergence and laplacian

        if (useCam)
        {
            // wait 30ms for key input
            if (cv::waitKey(30) >= 0)
            {
                mIn.release();
            }
            else
            {
                // retrieve next frame from camera
                camera >> mIn;
                // convert to float representation (opencv loads image values as single bytes by default)
                mIn.convertTo(mIn, CV_32F);
            }
        }
    }
    while (useCam && !mIn.empty());

    if (!useCam)
    {
        cv::waitKey(0);

        // save input and result
        //cv::imwrite("image_input.png",mIn*255.f);  // "imwrite" assumes channel range [0,255]
        cv::imwrite("image_result.png",mOut_lapNorm*255.f);
    }

    // ### Free allocated arrays
    // TODO free cuda memory of all device arrays
    // TODO free memory of all host arrays

    // close all opencv windows
    cv::destroyAllWindows();

    return 0;
}
