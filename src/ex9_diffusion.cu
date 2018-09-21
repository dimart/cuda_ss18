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
#include "diffusion.cuh"
#include "convolution.cuh"


int main(int argc,char **argv)
{
    // parse command line parameters
    const char *params = {
        "{i|image| |input image}"
        "{w|bw|false|load input image as grayscale/black-white}"
        "{n|iter|100|iterations}"
        "{e|epsilon|0.01|epsilon}"
        "{d|dt|0.001|dt}"
        "{g|0|diffusivity: 0=constant, 1=max, 2=exp}"
        "{c|cmp|false|compare with Gaussian}"
    };
    cv::CommandLineParser cmd(argc, argv, params);

    // input image
    std::string inputImage = cmd.get<std::string>("image");
    // load the input image as grayscale
    bool gray = cmd.get<bool>("bw");
    size_t iter = (size_t)cmd.get<int>("iter");
    std::cout << "iterations: " << iter << std::endl;
    float epsilon = cmd.get<float>("epsilon");
    std::cout << "epsilon: " << epsilon << std::endl;
    size_t diffusivity_mode = (size_t)cmd.get<int>("g");
    if (diffusivity_mode == 1)
        std::cout << "diffusivity: max" << std::endl;
    else if (diffusivity_mode == 2)
        std::cout << "diffusivity: exponential" << std::endl;
    else
        std::cout << "diffusivity: constant" << std::endl;

    bool cmp_with_gauss = cmd.get<bool>("cmp");

    float dt = cmd.get<float>("dt");
    if (dt == 0.0f)
        dt = 0.225f/funcDiffusivity(0, epsilon, diffusivity_mode);
    std::cout << "dt: " << dt << std::endl;

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
    cv::Mat mOut(h,w,mIn.type());  // grayscale or color depending on input image, nc layers

    // ### Allocate arrays
    // allocate raw input image array
    float *imgIn = new float[w * h * nc];
    // allocate raw output array (the computation result will be stored in this array, then later converted to mOut for displaying)
    float *imgOut = new float[w * h * nc];

    // allocate arrays on GPU
    size_t nbytes_fullCh = (size_t)(h * w * nc)*sizeof(float);
    size_t nbytes_oneCh = (size_t)(h * w)*sizeof(float);

    float *d_imgIn = NULL;
    float *d_v1 = NULL;
    float *d_v2 = NULL;
    float *d_div = NULL;
    cudaMalloc(&d_imgIn, nbytes_fullCh);
    cudaMalloc(&d_v1, nbytes_fullCh);
    cudaMalloc(&d_v2, nbytes_fullCh);
    cudaMalloc(&d_div, nbytes_fullCh);

    // setup Gaussian kernel for comparision
    float *gauss_img = new float[w * h * nc];
    float *d_gauss_img = NULL;
    float *d_gauss_kernel = NULL;
    cv::Mat mGaussImg(h,w,mIn.type());

    if (cmp_with_gauss) {
        float sigma = sqrt(2 * dt * iter);
        int kradius = ceil(3 * sigma);
        std::cout << "Gaussian kernel radius = " << kradius << std::endl;
        std::cout << "Gaussian kernel sigma = " << sigma << std::endl;
        int k_diameter = 2 * kradius + 1;
        int kn = k_diameter * k_diameter;
        float *gauss_kernel = new float[kn];
        createConvolutionKernel(gauss_kernel, kradius, sigma);

        // allocate memory for the result
        cudaMalloc(&d_gauss_img, nbytes_fullCh);
        cudaMalloc(&d_gauss_kernel, kn * sizeof(float));

        // copy gaussian kernel
        convertMatToLayered (imgIn, mIn / 255.0f);
        cudaMemcpy(d_imgIn, imgIn, nbytes_fullCh, cudaMemcpyHostToDevice); CUDA_CHECK;
        cudaMemcpy(d_gauss_kernel, gauss_kernel, kn * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;

        // apply gaussian to the input image
        computeConvolutionSharedMemCuda(d_gauss_img, d_imgIn, d_gauss_kernel, kradius, w, h, nc);
    }

    do
    {
        // convert range of each channel to [0,1]
        mIn /= 255.0f;

        // init raw input image array (and convert to layered)
        convertMatToLayered (imgIn, mIn);
        // CPU => GPU
        cudaMemcpy(d_imgIn, imgIn, nbytes_fullCh, cudaMemcpyHostToDevice); CUDA_CHECK;

        Timer timer;
        timer.start();
        for(size_t i = 0; i < iter; ++i)
        {
            // compute gradient of d_imgIn
            computeGradientCuda(d_v1, d_v2, d_imgIn, w, h, nc);

            // compute diffusivity and store it back into d_v
            multDiffusivityCuda(d_v1, d_v2, w, h, nc, epsilon, diffusivity_mode);
            cudaDeviceSynchronize();

            // compute divergence of d_v1, d_v2
            computeDivergenceCuda(d_div, d_v1, d_v2, w, h, nc);

            // perform the update
            updateDiffusivityCuda(d_imgIn, d_div, w, h, nc, dt);
            cudaDeviceSynchronize();
        }
        timer.end();
        float t = timer.get();
        std::cout << "time: " << t*1000 << " ms" << std::endl;

        // GPU => CPU
        cudaMemcpy(imgOut, d_imgIn, nbytes_fullCh, cudaMemcpyDeviceToHost); CUDA_CHECK;

        // show input image
        showImage("Input", mIn, 100, 100);  // show at position (x_from_left=100,y_from_above=100)

        // show output image: first convert to interleaved opencv format from the layered raw array
        convertLayeredToMat(mOut, imgOut);
        showImage("Output", mOut, 100+w+40, 100);

        if (cmp_with_gauss) {
            // show the input image blurred with gaussian
            cudaMemcpy(gauss_img, d_gauss_img, nbytes_fullCh, cudaMemcpyDeviceToHost); CUDA_CHECK;
            convertLayeredToMat(mGaussImg, gauss_img);
            showImage("Blurred with Gaussian", mGaussImg, 100+w+40, 100+h+40);
        }
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
    delete[] imgIn;
    delete[] imgOut;
    cudaFree(d_imgIn); CUDA_CHECK;
    cudaFree(d_v1); CUDA_CHECK;
    cudaFree(d_v2); CUDA_CHECK;
    cudaFree(d_div); CUDA_CHECK;

    // close all opencv windows
    cv::destroyAllWindows();

    return 0;
}



