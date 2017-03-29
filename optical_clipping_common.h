#ifndef OPTICAL_CLIPPING_COMMON_H
#define OPTICAL_CLIPPING_COMMON_H

#include <cuda_runtime.h>


typedef struct {
	float x[8];
	float y[8];
} clippedGeom;

typedef struct {
	int numPoints;
	float x[8], y[8];
} FootPrint;

typedef struct {
	float hclip[4];
	float magdx;
	float magdy;
	float selfitsnx[4];
	float selfitsny[4];
	float sortedselfitsnx[4];
	float sortedselfitsny[4];
	float lineEdge[8][2];
	FootPrint newPoly;
	FootPrint newTex;
	int inpolyTest[4];
	float perpLines[4][4];
	float newpln_1[4];
	float newpln_2[4];
	float newpln_3[4];
	float newpln_4[4];
	float xv[4];
	float yv[4];
	int uniqueSelfItsn;
	
} experimentData;


////////////////////////////////////////////////////////////////////////////////
// Reference CPU clipping
////////////////////////////////////////////////////////////////////////////////
extern "C" void computeTriClipCPU(
    float *h_Dst,
    float *h_Src,
    float *h_Kernel,
    int imageW,
    int imageH,
    int kernelR
);

extern "C" void initCuda(cudaArray *boxfilter_Src, cudaChannelFormatDesc channelDesc);


////////////////////////////////////////////////////////////////////////////////
// GPU clipping
////////////////////////////////////////////////////////////////////////////////


/*
extern "C"
void computeTriClipGPU(
	int filterFrame, int numExposures,int spatialImgWidth, int spatialImgHeight, 
	int *d_sensorX, int *d_sensorY, int *d_plane,
	clippedGeom *d_resultsGeom, clippedGeom *d_resultsTex, int *d_resultsGeomLen, 
	int numSensors);
*/	

extern "C"
void callKernel(int filterFrames, int numExposures, int spatialWidth, int spatialHeight, int *d_sensorX, int *d_sensorY, int* d_plane,
	clippedGeom *d_resultsGeom,clippedGeom *d_resultsGeom, int *d_resultsGeomLen, int maxSensors, experimentData *debug);




#endif
