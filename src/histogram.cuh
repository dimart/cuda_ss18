// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// ########################################################################
#ifndef TUM_HISTOGRAM_H
#define TUM_HISTOGRAM_H

#include <iostream>


void computeHistogramCuda(int *histogram, float *imgIn, int nbins, int w, int h, int nc);

void computeHistogramCudaShared(int *histogram, float *imgIn, int w, int h, int nc);

#endif
