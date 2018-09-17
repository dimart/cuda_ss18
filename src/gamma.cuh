// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// ########################################################################
#ifndef TUM_GAMMA_H
#define TUM_GAMMA_H

void computeGamma(float *imgOut, const float *imgIn, float gamma, size_t w, size_t h, size_t nc);
void computeGammaCuda(float *imgOut, const float *imgIn, float gamma, int w, int h, int nc);

#endif
