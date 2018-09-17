// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// ########################################################################

#include <iostream>
#include <string>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "helper.cuh"
#include "diffusion.cuh"
#include "energy.cuh"
#include "reduction.cuh"

#include "cublas_v2.h"


int main(int argc,char **argv)
{
    // parse command line parameters
    const char *params = {
        "{i|image| |input image}"
        "{w|bw|false|load input image as grayscale/black-white}"
        "{n|iter|100|iterations}"
        "{e|epsilon|0.01|epsilon}"
        "{l|lambda|1.0|lambda}"
        "{t|theta|0.9|theta}"
        "{g|noise|0.1|gaussian noise}"
        "{r|repeats|1|number of computation repetitions}"
        "{j|jacobistep|1|jacobi step (1=jacobi, 0=SOR)}"
        "{p|reduce|false|parallel reduction}"
    };
    cv::CommandLineParser cmd(argc, argv, params);

    // input image
    std::string inputImage = cmd.get<std::string>("image");
    // number of computation repetitions to get a better run time measurement
    size_t repeats = (size_t)cmd.get<int>("repeats");
    // load the input image as grayscale
    bool gray = cmd.get<bool>("bw");
    size_t iter = (size_t)cmd.get<int>("iter");
    std::cout << "iterations: " << iter << std::endl;
    float epsilon = cmd.get<float>("epsilon");
    std::cout << "epsilon: " << epsilon << std::endl;
    float lambda = cmd.get<float>("lambda");
    std::cout << "lambda: " << lambda << std::endl;
    float theta = cmd.get<float>("theta");
    std::cout << "theta: " << theta << std::endl;
    float noise = cmd.get<float>("noise");
    std::cout << "noise: " << noise << std::endl;
    bool jacobiStep = (bool)cmd.get<int>("jacobistep");
    std::cout << "jacobi step: " << jacobiStep << std::endl;
    bool reduce = cmd.get<bool>("reduce");

    if (reduce)
    {
        // ### PARALLEL_REDUCE_TEST
        // specify command line parameter -p to run parallel reduction
        // TODO (12.1) complete runParallelReduction() in reduction.cu
        runParallelReduction(100000, repeats);
        return 0;
    }

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
    float *imgIn = NULL;    // TODO allocate array
    // allocate raw output array (the computation result will be stored in this array, then later converted to mOut for displaying)
    float *imgOut = NULL;    // TODO allocate array

    // allocate arrays on GPU
    // input
    float *d_imgData = NULL;
    // temp
    float *d_imgIn = NULL;
    float *d_imgOut = NULL;
    float *d_diffusivity = NULL;    // TODO allocate array
    float *d_energy = NULL;    // TODO allocate array

    // create cublas handle
    cublasHandle_t handle;
    // TODO create handle using cublasCreate()

    do
    {
        // convert range of each channel to [0,1]
        mIn /= 255.0f;

        // add noise to input image
        // TODO (11.1)

        // init raw input image array (and convert to layered)
        convertMatToLayered (imgIn, mIn);

        // fixed input image
        // TODO upload input to device and copy to working image

        Timer timer;
        timer.start();

        float *a_in = d_imgIn;
        float *a_out = d_imgOut;
        for(size_t i = 0; i < iter; ++i)
        {
            // TODO (11.2) compute diffusivity using computeDiffusivityCuda() in diffusion.cu
            computeDiffusivityCuda(d_diffusivity, a_in, w, h, nc, epsilon);  CUDA_CHECK;

            // if jacobi
            if (jacobiStep)
            {
                // TODO (11.2) implement minimizeEnergyJacobiStep() in energy.cu
                minimizeEnergyJacobiStepCuda(a_out, a_in, d_diffusivity, d_imgData, w, h, nc, lambda);  CUDA_CHECK;
                std::swap(a_in, a_out);  // the output is always in "a_in" after this
            }
            else
            {
                // if SOR
                // TODO (11.3) implement minimizeEnergySorStep() in energy.cu
                minimizeEnergySorStepCuda(a_in, a_in, d_diffusivity, d_imgData, w, h, nc, lambda, theta, 0);  CUDA_CHECK;
                minimizeEnergySorStepCuda(a_in, a_in, d_diffusivity, d_imgData, w, h, nc, lambda, theta, 1);  CUDA_CHECK;
            }

            // calculate energy
            // TODO (12.2) implement computeEnergyCuda() in energy.cu
            computeEnergyCuda(d_energy, a_in, d_imgData, w, h, nc, lambda, epsilon); CUDA_CHECK;

            float energy = 0.0f;
            // TODO (12.2) compute energy from d_energy using cublasSasum()
            std::cout << i << "," << energy << std::endl;
        }
        timer.end();
        float t = timer.get();
        std::cout << "time: " << t*1000 << " ms" << std::endl;

        // download from GPU
        // TODO copy all necessary arrays from device to host

        // show input image
        showImage("Input", mIn, 100, 100);  // show at position (x_from_left=100,y_from_above=100)

        // show output image: first convert to interleaved opencv format from the layered raw array
        convertLayeredToMat(mOut, imgOut);
        showImage("Output", mOut, 100+w+40, 100);

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
        cv::imwrite("image_result.png",mOut*255.f);
    }

    // ### Free allocated arrays
    // TODO free cuda memory of all device arrays
    // TODO free memory of all host arrays

    // close all opencv windows
    cv::destroyAllWindows();

    return 0;
}



