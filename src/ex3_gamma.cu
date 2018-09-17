// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// ########################################################################

#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "gamma.cuh"
#include "helper.cuh"



int main(int argc,char **argv)
{
    // parse command line parameters
    const char *params = {
        "{i|image| |input image}"
        "{b|bw|false|load input image as grayscale/black-white}"
        "{g|gamma|2.2|gamma value}"
        "{r|repeats|1|number of computation repetitions}"
        "{c|cpu|false|compute on CPU}"
    };
    cv::CommandLineParser cmd(argc, argv, params);

    // input image
    std::string inputImage = cmd.get<std::string>("image");
    // number of computation repetitions to get a better run time measurement
    size_t repeats = (size_t)cmd.get<int>("repeats");
    // load the input image as grayscale
    bool gray = cmd.get<bool>("bw");
    // gamma correction value
    float gamma = cmd.get<float>("gamma");
    // compute on CPU
    bool cpu = cmd.get<bool>("cpu");
    std::cout << "mode: " << (cpu ? "CPU" : "GPU") << std::endl;

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

    // Before the GPU can process your kernels, a so called "CUDA context" must be initialized
    // This happens on the very first call to a CUDA function, and takes some time (around half a second)
    // We will do it right here, so that the run time measurements are accurate
    if (!cpu) {
        cudaDeviceSynchronize();  CUDA_CHECK;
    }
    // ### Set the output image format
    // Let mOut have the same number of channels as the input image (e.g. for the "invert image" or the "convolution" exercise)
    // To let mOut be a color image with 3 channels: CV_32FC3 instead of mIn.type() (e.g. for "visualization of the laplacian" exercise)
    // To let mOut be a grayscale image: use CV_32FC1 instead of mIn.type() (e.g. for the "visualization of the gradient absolute value" exercise)
    //
    // ### TODO: Change the output image format as needed by the exercise (CV_32FC1 for grayscale, CV_32FC3 for color, mIn.type() for same as input)
    cv::Mat mOut(h, w, mIn.type());  // grayscale or color depending on input image, nc layers
    //cv::Mat mOut(h, w, CV_32FC3);    // color, 3 layers
    //cv::Mat mOut(h, w, CV_32FC1);    // grayscale, 1 layer
    //
    // If you want to display other images, define them here as needed, e.g. the opencv image for the convolution kernel

    // ### Allocate arrays
    // input/output image width: w
    // input/output image height: h
    // input image number of channels: nc
    // output image number of channels: mOut.channels(), as defined above depending on the exercise (1 for grayscale, 3 for color, nc for general)
    //

    // allocate arrays on CPU
    // allocate raw input image array
    float *imgIn = new float[h * w * nc];
    // allocate raw output array (the computation result will be stored in this array, then later converted to mOut for displaying)
    float *imgOut = new float[h * w * nc];

    float *d_imgIn = NULL;
    float *d_imgOut = NULL;
    size_t nbytes = (size_t)(h * w * nc)*sizeof(float);

    if (!cpu) {
        // allocate arrays on GPU
        cudaMalloc(&d_imgIn, nbytes); CUDA_CHECK;
        cudaMalloc(&d_imgOut, nbytes); CUDA_CHECK;
    }

    do
    {
        // convert range of each channel to [0,1] (opencv default is [0,255])
        mIn /= 255.0f;

        // ### Init raw input image array
        // opencv images are interleaved: rgb rgb rgb...  (actually bgr bgr bgr...)
        // But for CUDA it's better to work with layered images: rrr... ggg... bbb...
        // So we will convert as necessary, using interleaved "opencv::Mat" for loading/saving/displaying, and layered "float*" for CUDA computations
        convertMatToLayered(imgIn, mIn);

        // ###
        // ### Notes:
        // ### 1. Input CPU image imgIn has nc channels. Do not assume nc=3, write the computation for a general nc
        // ### 2. Output CPU image imgOut has 1, 3, or nc channels, depending on how you defined it above. Use may assume 3 channels only if you have used CV_F32FC3.
        // ### 3. Images are layered: access imgIn(x,y,channel c) as imgIn[x + (size_t)w*y + nOmega*c],  where: size_t nOmega = (size_t)w*h;
        // ###
        // ### 4. Allocate arrays as necessary and remember to free them
        // ### 5. Always use the macro CUDA_CHECK after each CUDA call, e.g. "cudaMalloc(...); CUDA_CHECK;"
        // ###
        // ### 6. Use the Timer class to measure the run time:
        // ###    Timer timer; timer.start();
        // ###    ...
        // ###    timer.end();  float t = timer.get();  // elapsed time in seconds
        // ###    std::cout << "time: " << t*1000 << " ms" << std::endl;
        // ###

        if (cpu)
        {
            Timer timer;
            timer.start();
            for (size_t i = 0; i < repeats; ++i)
            {
                computeGamma(imgOut, imgIn, gamma, w, h, nc);
            }
            float t = timer.get()/repeats;
            std::cout << "average time: " << t * 1000 << " ms" << std::endl;
        }
        else
        {
            // CPU => GPU
            cudaMemcpy(d_imgIn, imgIn, nbytes, cudaMemcpyHostToDevice); CUDA_CHECK;

            // 2. Execute kernel
            Timer timer;
            timer.start();
            for(size_t i = 0; i < repeats; ++i)
            {
                computeGammaCuda(d_imgOut, d_imgIn, gamma, w, h, nc);
                cudaDeviceSynchronize(); CUDA_CHECK;
            }
            float t = timer.get()/repeats;
            std::cout << "average time: " << t*1000 << " ms" << std::endl;

            // GPU => CPU
            cudaMemcpy(imgOut, d_imgOut, nbytes, cudaMemcpyDeviceToHost); CUDA_CHECK;
        }

        // show input image
        showImage("Input", mIn, 100, 100);  // show at position (x_from_left=100,y_from_above=100)

        // show output image: first convert to interleaved opencv format from the layered raw array
        convertLayeredToMat(mOut, imgOut);
        showImage("Output", mOut, 100+w+40, 100);

        if (useCam)
        {
            // Read a camera image frames every 30 milliseconds:
            // cv::waitKey(30) waits 30 milliseconds for a keyboard input,
            // returns a value <0 if no key is pressed during this time, returns immediately with a value >=0 if a key is pressed
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



