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
    // try to run it with:
    // -i ../../images/gaudi.png -s 0.5 -a 1e-3 -b 0.0005
    // :)
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
    int kradius = ceil(3 * sigma);
    std::cout << "kradius: " << kradius << std::endl;
    int k_diameter = 2 * kradius + 1;
    int kn = k_diameter*k_diameter;
    float *kernel = new float[kn];
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
    cv::Mat mM12(h,w,CV_32FC1);    // grayscale, 1 layer
    cv::Mat mM22(h,w,CV_32FC1);    // grayscale, 1 layer

    // ### Allocate arrays

    // allocate raw input image array
    float *imgIn = new float[h * w * nc];

    // allocate raw output array (the computation result will be stored in this array, then later converted to mOut for displaying)
    float *imgOut = new float[h * w * nc];
    float *t11 = new float[h * w];
    float *t12 = new float[h * w];
    float *t22 = new float[h * w];

    // allocate kernels
    int kernelDxDy_radius = 1;
    size_t nbytes_kernelDxDy = (size_t)(9)*sizeof(float);
    float kernelDx[9] = { -3.0f/32.0f,  0.0f,  3.0f/32.0f,
                         -10.0f/32.0f,  0.0f, 10.0f/32.0f,
                          -3.0f/32.0f,  0.0f,  3.0f/32.0f};

    float kernelDy[9] = { -3.0f/32.0f, -10.0f/32.0f,  -3.0f/32.0f,
                                 0.0f,        0.0f,          0.0f,
                           3.0f/32.0f,  10.0f/32.0f,   3.0f/32.0f};


    // allocate arrays on GPU
    size_t nbytes_fullCh = (size_t)(h * w * nc)*sizeof(float);
    size_t nbytes_oneCh = (size_t)(h * w)*sizeof(float);
    size_t nbytes_gauss_kernel = (size_t)(kn)*sizeof(float);

    // kernel
    float *d_kernelGauss = NULL;
    float *d_kernelDx = NULL;
    float *d_kernelDy = NULL;
    cudaMalloc(&d_kernelGauss, nbytes_gauss_kernel);
    cudaMalloc(&d_kernelDx, nbytes_kernelDxDy);
    cudaMalloc(&d_kernelDy, nbytes_kernelDxDy);

    // input
    float *d_imgIn = NULL;
    cudaMalloc(&d_imgIn, nbytes_fullCh);

    // output
    float *d_imgOut = NULL;
    cudaMalloc(&d_imgOut, nbytes_fullCh);

    // temp
    float *d_inSmooth = NULL;
    cudaMalloc(&d_inSmooth, nbytes_fullCh);

    float *d_dx = NULL;
    float *d_dy = NULL;
    cudaMalloc(&d_dx, nbytes_fullCh);
    cudaMalloc(&d_dy, nbytes_fullCh);

    float *d_tensor11Nonsmooth = NULL;
    float *d_tensor12Nonsmooth = NULL;
    float *d_tensor22Nonsmooth = NULL;
    cudaMalloc(&d_tensor11Nonsmooth, nbytes_oneCh);
    cudaMalloc(&d_tensor12Nonsmooth, nbytes_oneCh);
    cudaMalloc(&d_tensor22Nonsmooth, nbytes_oneCh);

    float *d_tensor11 = NULL;
    float *d_tensor12 = NULL;
    float *d_tensor22 = NULL;
    cudaMalloc(&d_tensor11, nbytes_oneCh);
    cudaMalloc(&d_tensor12, nbytes_oneCh);
    cudaMalloc(&d_tensor22, nbytes_oneCh);

    float *d_lmb1 = NULL;
    float *d_lmb2 = NULL;
    cudaMalloc(&d_lmb1, nbytes_oneCh);
    cudaMalloc(&d_lmb2, nbytes_oneCh);

    do
    {
        // convert range of each channel to [0,1]
        mIn /= 255.0f;

        // init raw input image array (and convert to layered)
        convertMatToLayered(imgIn, mIn);

        // CPU => GPU
        cudaMemcpy(d_imgIn, imgIn, nbytes_fullCh, cudaMemcpyHostToDevice); CUDA_CHECK;
        cudaMemcpy(d_kernelGauss, kernel, nbytes_gauss_kernel, cudaMemcpyHostToDevice); CUDA_CHECK;
        cudaMemcpy(d_kernelDx, kernelDx, nbytes_kernelDxDy, cudaMemcpyHostToDevice); CUDA_CHECK;
        cudaMemcpy(d_kernelDy, kernelDy, nbytes_kernelDxDy, cudaMemcpyHostToDevice); CUDA_CHECK;

        Timer timer;
        timer.start();

        // S = G_sigma * u
        computeConvolutionGlobalMemCuda(d_inSmooth, d_imgIn, d_kernelGauss, kradius, w, h, nc);

        // v1 = dxS, v2 = dyS
        computeConvolutionGlobalMemCuda(d_dx, d_inSmooth, d_kernelDx, kernelDxDy_radius, w, h, nc);
        computeConvolutionGlobalMemCuda(d_dy, d_inSmooth, d_kernelDy, kernelDxDy_radius, w, h, nc);

        // compute tensor M = {{dxS*dxS, dxS*dyS}, {dyS*dxS, dyS}}, a.k.a. m11, m12, m22
        computeStructureTensorCuda(d_tensor11Nonsmooth, d_tensor12Nonsmooth, d_tensor22Nonsmooth, d_dx, d_dy, w, h, nc);  CUDA_CHECK;

        // blur tensor, T = G_sigma * M
        computeConvolutionGlobalMemCuda(d_tensor11, d_tensor11Nonsmooth, d_kernelGauss, kradius, w, h, 1);
        computeConvolutionGlobalMemCuda(d_tensor12, d_tensor12Nonsmooth, d_kernelGauss, kradius, w, h, 1);
        computeConvolutionGlobalMemCuda(d_tensor22, d_tensor22Nonsmooth, d_kernelGauss, kradius, w, h, 1);

        // compute detector
        computeDetectorCuda(d_lmb1, d_lmb2, d_tensor11, d_tensor12, d_tensor22, w, h);

        // set output image
        computeTensorOutputCuda(d_imgOut, d_lmb1, d_lmb2, d_imgIn, w, h, nc, alpha, beta);
        cudaThreadSynchronize();

        timer.end();
        float t = timer.get();
        std::cout << "time: " << t*1000 << " ms" << std::endl;

        // GPU => CPU
        cudaMemcpy(imgOut, d_imgOut, nbytes_fullCh, cudaMemcpyDeviceToHost); CUDA_CHECK;
        cudaMemcpy(t11, d_tensor11, nbytes_oneCh, cudaMemcpyDeviceToHost); CUDA_CHECK;
        cudaMemcpy(t12, d_tensor12, nbytes_oneCh, cudaMemcpyDeviceToHost); CUDA_CHECK;
        cudaMemcpy(t22, d_tensor22, nbytes_oneCh, cudaMemcpyDeviceToHost); CUDA_CHECK;

        // show input image
        showImage("Input", mIn, 100, 100);  // show at position (x_from_left=100,y_from_above=100)

        // show output image: first convert to interleaved opencv format from the layered raw array
        convertLayeredToMat(mOut, imgOut);
        showImage("Output", mOut, 100, 100);

        // visualize tensor images t11, t12, t22 (incl. scaling)
        convertLayeredToMat(mM11, t11);
        convertLayeredToMat(mM12, t12);
        convertLayeredToMat(mM22, t22);
        float f = 10.f;
        mM11 *= f;
        mM12 *= f;
        mM22 *= f;
        showImage("M11", mM11, 100+w+40, 100);
        showImage("M12", mM12, 100, 100+w+40);
        showImage("M22", mM22, 100+w+40, 100+w+40);

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
        cv::imwrite("image_m12.png", mM12*10*255.f);
        cv::imwrite("image_m22.png", mM22*10*255.f);
    }

    // ### Free allocated arrays
    delete[] imgIn;
    delete[] imgOut;
    delete[] kernel;
    delete[] t11;
    delete[] t12;
    delete[] t22;
    cudaFree(d_imgIn); CUDA_CHECK;
    cudaFree(d_imgOut); CUDA_CHECK;
    cudaFree(d_kernelGauss); CUDA_CHECK;
    cudaFree(d_kernelDx); CUDA_CHECK;
    cudaFree(d_kernelDy); CUDA_CHECK;
    cudaFree(d_dx); CUDA_CHECK;
    cudaFree(d_dy); CUDA_CHECK;
    cudaFree(d_lmb1); CUDA_CHECK;
    cudaFree(d_lmb2); CUDA_CHECK;
    cudaFree(d_tensor11); CUDA_CHECK;
    cudaFree(d_tensor12); CUDA_CHECK;
    cudaFree(d_tensor22); CUDA_CHECK;
    cudaFree(d_tensor11Nonsmooth); CUDA_CHECK;
    cudaFree(d_tensor12Nonsmooth); CUDA_CHECK;
    cudaFree(d_tensor22Nonsmooth); CUDA_CHECK;
    cudaFree(d_inSmooth); CUDA_CHECK;

    // close all opencv windows
    cv::destroyAllWindows();

    return 0;
}



