// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// ########################################################################

#include <iostream>
#include <string>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "helper.cuh"
#include "convolution.cuh"
#include "structure_tensor.cuh"


int main(int argc,char **argv)
{
    // parse command line parameters
    const char *params = {
        "{i|image| |input image}"
        "{w|bw|false|load input image as grayscale/black-white}"
        "{s|sigma|1.0|sigma}"
        "{a|alpha|0.005|alpha}"
        "{b|beta|0.001|beta}"
    };
    cv::CommandLineParser cmd(argc, argv, params);

    // input image
    std::string inputImage = cmd.get<std::string>("image");
    // load the input image as grayscale
    bool gray = cmd.get<bool>("bw");
    float sigma = cmd.get<float>("sigma");
    std::cout << "sigma: " << sigma << std::endl;
    float alpha = cmd.get<float>("alpha");
    std::cout << "alpha: " << alpha << std::endl;
    float beta = cmd.get<float>("beta");
    std::cout << "beta: " << beta << std::endl;

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

    // init kernel
    int kradius = 0;    // TODO (7.1) calculate kernel radius using sigma
    std::cout << "kradius: " << kradius << std::endl;
    int k_diameter = 0;     // TODO (7.1) calculate kernel diameter from radius
    int kn = k_diameter*k_diameter;
    float *kernel = NULL;    // TODO (7.1) allocate array
    createConvolutionKernel(kernel, kradius, sigma);

    // get image dimensions
    int w = mIn.cols;         // width
    int h = mIn.rows;         // height
    int nc = mIn.channels();  // number of channels
    std::cout << "Image: " << w << " x " << h << std::endl;

    // initialize CUDA context
    cudaDeviceSynchronize();  CUDA_CHECK;

    // ### Set the output image format
    //cv::Mat mOut(h,w,mIn.type());  // grayscale or color depending on input image, nc layers
    cv::Mat mOut(h,w,CV_32FC3);    // color, 3 layers
    cv::Mat mM11(h,w,CV_32FC1);    // grayscale, 1 layer
    cv::Mat mM21(h,w,CV_32FC1);    // grayscale, 1 layer
    cv::Mat mM22(h,w,CV_32FC1);    // grayscale, 1 layer

    // ### Allocate arrays
    // allocate raw input image array
    float *imgIn = NULL;    // TODO allocate array
    // allocate raw output array (the computation result will be stored in this array, then later converted to mOut for displaying)
    float *imgOut = NULL;    // TODO allocate array
    float *t11 = NULL;    // TODO allocate array
    float *t22 = NULL;    // TODO allocate array
    float *t12 = NULL;    // TODO allocate array

    // allocate arrays on GPU
    // kernel
    float *d_kernelGauss = NULL;
    // TODO alloc cuda memory for device arrays
    float kernelDx[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};    // TODO (7.2) fill
    float kernelDy[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};    // TODO (7.2) fill
    float *d_kernelDx = NULL;
    float *d_kernelDy = NULL;
    // TODO alloc cuda memory for device arrays
    // input
    float *d_imgIn = NULL;
    // TODO alloc cuda memory for device arrays
    // output
    float *d_imgOut = NULL;
    // TODO alloc cuda memory for device arrays
    // temp
    float *d_inSmooth = NULL;
    float *d_dx = NULL;
    float *d_dy = NULL;
    float *d_tensor11Nonsmooth = NULL;
    float *d_tensor12Nonsmooth = NULL;
    float *d_tensor22Nonsmooth = NULL;
    float *d_tensor11 = NULL;
    float *d_tensor12 = NULL;
    float *d_tensor22 = NULL;
    float *d_lmb1 = NULL;
    float *d_lmb2 = NULL;
    // TODO alloc cuda memory for device arrays

    do
    {
        // convert range of each channel to [0,1]
        mIn /= 255.0f;

        // init raw input image array (and convert to layered)
        convertMatToLayered (imgIn, mIn);

        // TODO (7.1) upload kernel to device
        // TODO (7.2) upload kernelDx and kernelDy to device
        // TODO upload input to device

        Timer timer;
        timer.start();

        // TODO (7.1) smooth imgIn using computeConvolutionGlobalMemCuda()

        // TODO (7.2) compute derivatives d_dx and d_dy using computeConvolutionGlobalMemCuda()

        // compute tensor
        // TODO (7.3) implement computeStructureTensorCuda() in structure_tensor.cu
        computeStructureTensorCuda(d_tensor11Nonsmooth, d_tensor12Nonsmooth, d_tensor22Nonsmooth, d_dx, d_dy, w, h, nc);  CUDA_CHECK;
        cudaThreadSynchronize();

        // blur tensor
        // TODO (7.4) blur non-smooth tensor images using computeConvolutionGlobalMemCuda()

        // compute detector
        // TODO (8.2) implement computeDetectorCuda() in structure_tensor.cu
        computeDetectorCuda(d_lmb1, d_lmb2, d_tensor11, d_tensor12, d_tensor22, w, h);
        cudaThreadSynchronize();

        // set output image
        // TODO (8.3) implement computeTensorOutputCuda() in structure_tensor.cu
        computeTensorOutputCuda(d_imgOut, d_lmb1, d_lmb2, d_imgIn, w, h, nc, alpha, beta);
        cudaThreadSynchronize();

        timer.end();
        float t = timer.get();
        std::cout << "time: " << t*1000 << " ms" << std::endl;

        // TODO copy all necessary arrays from device to host

        // show input image
        showImage("Input", mIn, 100, 100);  // show at position (x_from_left=100,y_from_above=100)

        // show output image: first convert to interleaved opencv format from the layered raw array
        convertLayeredToMat(mOut, imgOut);
        showImage("Output", mOut, 100+w+40, 100);

        // TODO (7.5) visualize tensor images t11, t12, t22 (incl. scaling)

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
        cv::imwrite("image_result.png", mOut*255.f);
        cv::imwrite("image_m11.png", mM11*10*255.f);
        cv::imwrite("image_m21.png", mM21*10*255.f);
        cv::imwrite("image_m22.png", mM22*10*255.f);
    }

    // ### Free allocated arrays
    // TODO free cuda memory of all device arrays
    // TODO free memory of all host arrays

    // close all opencv windows
    cv::destroyAllWindows();

    return 0;
}



