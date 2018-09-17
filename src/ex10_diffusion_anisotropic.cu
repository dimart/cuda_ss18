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
#include "gradient.cuh"
#include "divergence.cuh"
#include "diffusion.cuh"
#include "structure_tensor.cuh"


int main(int argc,char **argv)
{
    // parse command line parameters
    const char *params = {
        "{i|image| |input image}"
        "{w|bw|false|load input image as grayscale/black-white}"
        "{n|iter|100|iterations}"
        "{d|dt|0.01|dt}"
        "{s|sigma|0.25|sigma (for smoothing input)}"
        "{a|alpha|0.5|alpha}"
        "{c|c|0.1|c}"
        "{r|rho|5.0|rho (for smoothing tensor)}"
    };
    cv::CommandLineParser cmd(argc, argv, params);

    // input image
    std::string inputImage = cmd.get<std::string>("image");
    // load the input image as grayscale
    bool gray = cmd.get<bool>("bw");
    size_t iter = (size_t)cmd.get<int>("iter");
    std::cout << "iterations: " << iter << std::endl;
    float dt = cmd.get<float>("dt");
    std::cout << "dt: " << dt << std::endl;
    float sigma = cmd.get<float>("sigma");
    std::cout << "sigma: " << sigma << std::endl;
    float alpha = cmd.get<float>("alpha");
    std::cout << "alpha: " << alpha << std::endl;
    float C = cmd.get<float>("c");
    std::cout << "C: " << C << std::endl;
    float rho = cmd.get<float>("rho");
    std::cout << "rho: " << rho << std::endl;

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

    // init kernels
    int kradius2 = 0;    // TODO calculate kernel radius using rho
    std::cout << "kradius2: " << kradius2 << std::endl;
    int k_diameter2 = 0;     // TODO calculate kernel diameter from radius
    int kn2 = k_diameter2*k_diameter2;
    float *kernelGaussTensor = NULL;    // TODO allocate array
    createConvolutionKernel(kernelGaussTensor, kradius2, rho);

    int kradius = 0;    // TODO calculate kernel radius using sigma
    std::cout << "kradius: " << kradius << std::endl;
    int k_diameter = 0;     // TODO calculate kernel diameter from radius
    int kn = k_diameter*k_diameter;
    float *kernelGauss = NULL;    // TODO allocate array
    createConvolutionKernel(kernelGauss, kradius, sigma);

    // gradient convolution kernels
    float kernelDx[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};    // TODO fill
    float kernelDy[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};    // TODO fill

    // get image dimensions
    int w = mIn.cols;         // width
    int h = mIn.rows;         // height
    int nc = mIn.channels();  // number of channels
    std::cout << "Image: " << w << " x " << h << std::endl;

    // initialize CUDA context
    cudaDeviceSynchronize();  CUDA_CHECK;

    // ### Set the output image format
    cv::Mat mOut(h,w,mIn.type());  // grayscale or color depending on input image, nc layers
    cv::Mat mOutT1(h,w,CV_32FC1);    // grayscale, 1 layer
    cv::Mat mOutT2(h,w,CV_32FC1);    // grayscale, 1 layer
    cv::Mat mOutT3(h,w,CV_32FC1);    // grayscale, 1 layer

    // ### Allocate arrays
    // allocate raw input image array
    float *imgIn = NULL;    // TODO allocate array
    // allocate raw output array (the computation result will be stored in this array, then later converted to mOut for displaying)
    float *imgOut = NULL;    // TODO allocate array
    float *outT1 = NULL;    // TODO allocate array
    float *outT2 = NULL;    // TODO allocate array
    float *outT3 = NULL;    // TODO allocate array

    // allocate arrays on GPU
    float *d_imgIn = NULL;
    float *d_v1 = NULL;
    float *d_v2 = NULL;
    float *d_div = NULL;
    float *d_kernelGauss = NULL;
    float *d_kernelGaussTensor = NULL;
    float *d_kernelDx = NULL;
    float *d_kernelDy = NULL;
    float *d_inSmooth = NULL;
    float *d_dx = NULL;
    float *d_dy = NULL;
    float *d_tensor11Nonsmooth = NULL;
    float *d_tensor12Nonsmooth = NULL;
    float *d_tensor22Nonsmooth = NULL;
    float *d_tensor11 = NULL;
    float *d_tensor12 = NULL;
    float *d_tensor22 = NULL;
    float *d_difftensor11 = NULL;
    float *d_difftensor12 = NULL;
    float *d_difftensor22 = NULL;

    // TODO alloc cuda memory for device arrays

    // TODO allocate and upload convolution kernels

    do
    {
        // convert range of each channel to [0,1]
        mIn /= 255.0f;

        // init raw input image array (and convert to layered)
        convertMatToLayered (imgIn, mIn);
        // upload to GPU
        // TODO copy from imgIn to d_imgIn

        // TODO (10.1) smooth d_imgIn using d_kernelGauss (result in d_inSmooth)
        cudaThreadSynchronize();

        // smooth d_imgIn using d_kernelGauss (result in d_inSmooth)
        // TODO (10.1) compute derivatives of d_inSmooth using d_kernelDx (result in d_dx)
        cudaThreadSynchronize();
        // TODO (10.1) compute derivatives of d_inSmooth using d_kernelDy (result in d_dy)
        cudaThreadSynchronize();

        // compute tensor
        // TODO (10.1) compute structure tensor using computeStructureTensorCuda() in structure_tensor.cu
        cudaThreadSynchronize();

        // TODO (10.1) blur tensor images (d_tensor11Nonsmooth etc) using computeConvolutionGlobalMemCuda(), result in d_tensor11 etc
        cudaThreadSynchronize();

        // compute diffusion tensor
        // TODO (10.1) implement computeDiffusionTensorCuda() in diffusion.cu
        computeDiffusionTensorCuda(d_difftensor11, d_difftensor12, d_difftensor22, d_tensor11, d_tensor12, d_tensor22, alpha, C, w, h, nc);  CUDA_CHECK;
        cudaThreadSynchronize();

        Timer timer;
        timer.start();
        for(size_t i = 0; i < iter; ++i)
        {
            // TODO (10.2) compute gradient of d_imgIn using computeGradientCuda() in gradient.cu
            cudaDeviceSynchronize();

            // TODO (10.2) implement multDiffusivityAnisotropicCuda() in diffusion.cu
            multDiffusivityAnisotropicCuda(d_v1, d_v2, d_difftensor11, d_difftensor12, d_difftensor22, w, h, nc);
            cudaDeviceSynchronize();

            // TODO (10.2) compute divergence of d_v1, d_v2 using computeDivergenceCuda() in divergence.cu
            cudaDeviceSynchronize();

            // TODO (10.2) update d_imgIn using updateDiffusivityCuda() in diffusion.cu
            cudaDeviceSynchronize();
        }
        timer.end();
        float t = timer.get();
        std::cout << "time: " << t*1000 << " ms" << std::endl;

        // download from GPU
        // TODO download from device arrays to host arrays

        // show input image
        showImage("Input", mIn, 100, 100);  // show at position (x_from_left=100,y_from_above=100)
        // show output image: first convert to interleaved opencv format from the layered raw array
        convertLayeredToMat(mOut, imgOut);
        showImage("Output", mOut, 100+w+40, 100);

        // show tensor
        convertLayeredToMat(mOutT1, outT1);
        convertLayeredToMat(mOutT2, outT2);
        convertLayeredToMat(mOutT3, outT3);
        showImage("t11", mOutT1, 100+2*w+40, 100);
        showImage("t12", mOutT2, 100+2*w+40, 100+h+10);
        showImage("t22", mOutT3, 100+w+40, 100+h+10);

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
        cv::imwrite("image_input.png",mIn*255.f);  // "imwrite" assumes channel range [0,255]
        cv::imwrite("image_result.png",mOut*255.f);
    }

    // ### Free allocated arrays
    // TODO free cuda memory of all device arrays
    // TODO free memory of all host arrays

    // close all opencv windows
    cv::destroyAllWindows();

    return 0;
}
