#include "tiffconf.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "tiffio.h"
#include <math.h>

#include <GL/glut.h>
#include <time.h>


#include <cuda_runtime.h>
#include <cutil_inline.h>

#include "optical_clipping_common.h"

typedef  struct {
	int numSensors;
	int * sensorList ; // 2D array
} SetofSensor;

typedef struct {
	int numSets;
	SetofSensor *sensors;
	
} SensorsDB;


typedef struct {
	int numSets;
	FootPrint *polys;
	FootPrint *polysTex;
	
} SensorsFP;

typedef struct {
	int numPlanes;
	SensorsFP *plane;
} FootPrintDB;

typedef struct {
	int width;
	int height;
	unsigned short *img;
} ImageDB;

typedef struct {
	int width;
	int height;
	int frames;
	void **imgs;
} ImageStackDB;


SensorsDB *load_sensorSet(const char * fname) {
	
	FILE *fp = fopen(fname, "r");
	
	SensorsDB *set = (SensorsDB *)malloc(sizeof(SensorsDB));
	
	int numFrames, numSensors;
	
	int result;
	
	char preamble[100];
	
	result = fscanf(fp, "%[^\n]",&preamble[0]);
	result = fscanf(fp, "%d",&numFrames);
	result = fscanf(fp, "%[^\n]",&preamble[0]);

	set->numSets = numFrames;
	set->sensors = (SetofSensor*)malloc(sizeof(SetofSensor)*numFrames);
	
	for (int f=0; f<numFrames; f++) {
		result = fscanf(fp, "%d",&numSensors);
		
		set->sensors[f].numSensors = numSensors;
		set->sensors[f].sensorList = (int*)malloc(sizeof(int)*2*numSensors);

		unsigned int col, row;
		
		for (int s=0; s<numSensors; s++) {
			result = fscanf(fp, "%u %u",&col,&row);
	
			set->sensors[f].sensorList[s*2] =  col;
			set->sensors[f].sensorList[s*2+1] =  row;
		}
	}
	
	return set;
}

void load_BoxFilterImgs(const char *fname, ImageStackDB *imgsdb) {
	TIFF *in;
	int dirCount = 0;
	
	in = TIFFOpen(fname, "r");
	
	if (in == NULL) {
		fprintf(stderr, "Bad BoxFilterImg file name");
		exit(-1);
	}
	
	// Count number of images
    if (in) {
		do {
			dirCount++;
		} while (TIFFReadDirectory(in));
		TIFFClose(in);
    }
	
	if (dirCount <= 1)
		fprintf(stderr, "Error: Expected more than %d BoxFilter Image\n", dirCount);
	else {
		fprintf(stdout, "Loaded %d BoxFilter images\n",dirCount);
	}

	
	// Reopen to load images
	in = TIFFOpen(fname, "r");
	
	uint8* raster;			
    uint32  width, height;		/* image width & height */
    size_t pixel_count;
	uint16 nsamples,nbits;
	
    TIFFGetField(in, TIFFTAG_IMAGEWIDTH, &width);
    TIFFGetField(in, TIFFTAG_IMAGELENGTH, &height);
	TIFFGetField(in, TIFFTAG_SAMPLESPERPIXEL, &nsamples);
	TIFFGetField(in, TIFFTAG_BITSPERSAMPLE, &nbits);
	
    pixel_count = width * height;
		
	// byte images to return
	float **imgs = (float**)malloc(sizeof(float*)*dirCount);

	// Scanline buffer read from byte encoded images
	raster = (uint8*)_TIFFmalloc(width* sizeof(uint8));
	
	for (int i=0; i<dirCount; i++) {
		// allocate an image
		imgs[i] = (float *)malloc(sizeof(float)*pixel_count);
		
		unsigned int row;
		for (row = 0; row < height; row++) {
			
			TIFFReadScanline(in, raster, row, 0);
			
			for (unsigned int col=0; col<width; col++)
				imgs[i][row*width + col] = (float)raster[col]/255.0;
				
		}
		TIFFReadDirectory(in);
	}
	
	TIFFClose(in);
	_TIFFfree(raster);
	
	imgsdb->imgs = (void **)imgs;
	imgsdb->width = width;
	imgsdb->height = height;
	imgsdb->frames = dirCount;

}


void load_BoxFilterImgsLinear(const char *fname, ImageStackDB *imgsdb) {
	TIFF *in;
	int dirCount = 0;
	
	in = TIFFOpen(fname, "r");
	
	if (in == NULL) {
		fprintf(stderr, "Bad BoxFilterImg file name");
		exit(-1);
	}
	
	// Count number of images
    if (in) {
		do {
			dirCount++;
		} while (TIFFReadDirectory(in));
		TIFFClose(in);
    }
	
	if (dirCount <= 1)
		fprintf(stderr, "Error: Expected more than %d BoxFilter Image\n", dirCount);
	else {
		fprintf(stdout, "Loaded %d BoxFilter images\n",dirCount);
	}

	
	// Reopen to load images
	in = TIFFOpen(fname, "r");
	
    uint32  width, height;		/* image width & height */
    size_t pixel_count;
	uint16 nsamples,nbits;
	
    TIFFGetField(in, TIFFTAG_IMAGEWIDTH, &width);
    TIFFGetField(in, TIFFTAG_IMAGELENGTH, &height);
	TIFFGetField(in, TIFFTAG_SAMPLESPERPIXEL, &nsamples);
	TIFFGetField(in, TIFFTAG_BITSPERSAMPLE, &nbits);
	
    pixel_count = width * height;
		
	// byte images to return, one linear chunk of memory
	uint8 *imgs = (uint8*)malloc(sizeof(uint8)*dirCount*pixel_count);
	
	for (int i=0; i<dirCount; i++) {
		
		unsigned int row;
		for (row = 0; row < height; row++) {
			
			TIFFReadScanline(in, &imgs[i*width*height + row*width], row, 0);
				
		}
		TIFFReadDirectory(in);
	}
	
	TIFFClose(in);
	
	imgsdb->imgs = (void **)imgs; // cast is clearly wrong but still is a pointer
	imgsdb->width = width;
	imgsdb->height = height;
	imgsdb->frames = dirCount;

}


void load_spatial_image(const char *fname, ImageDB *imdb) {
	TIFF *in;
	
	in = TIFFOpen(fname, "r");

    uint32  width, height;		/* image width & height */
    size_t pixel_count;
	uint16 nsamples,nbits;
	
    TIFFGetField(in, TIFFTAG_IMAGEWIDTH, &width);
    TIFFGetField(in, TIFFTAG_IMAGELENGTH, &height);
	TIFFGetField(in, TIFFTAG_SAMPLESPERPIXEL, &nsamples);
	TIFFGetField(in, TIFFTAG_BITSPERSAMPLE, &nbits);
	
    pixel_count = width * height;
	
	// floating point images to return
	unsigned short *img = (unsigned short*)malloc(sizeof(unsigned short)*pixel_count);
	
	unsigned int row;
	for (row = 0; row < height; row++)
		TIFFReadScanline(in, &img[row*width], row, 0);
	
	TIFFClose(in);
	
	imdb->img = img;
	imdb->width = width;
	imdb->height = height;
}

// This may be parallelized in CUDA
// Method to find linear exposure range pixel image
void bestExpImage(int plane, float **exposures, float *img, int width, int height, int numExposures, 
				   unsigned short *sList, int numSensors) {
		
	for (int p=0; p<numSensors; p++) {
		int c = sList[p*2] - 1; // Matlab to C++ index shift
		int r = sList[p*2+1] - 1; // Matlab to C++ index shift
		int sIndex = r*width + c;
		
		int found=0;
		float imgtest=0.0;
		
		for (int j=0; j<numExposures; j++) { // Loop over exposures for linear range
			imgtest = exposures[plane*numExposures + j][sIndex];
			if (imgtest < .9 && imgtest > .3) {
				img[sIndex] = imgtest;
				found = 1;
			}
		}
		
		if (!found) {
			img[sIndex] = exposures[plane*numExposures + numExposures-1][sIndex]; // choose the last exposure
		}
	}
		
}




void gradientImgX(unsigned short *I, float *gdx, int width, int height) {
	// Gradient is 3 neighbors wide
	
	for (int r=0; r<height; r++) {
		for (int c=0; c<width; c++) {
			
			// Boundary conditions
			if (c>0 && c<width-1)  // horizontal
				gdx[r*width + c] = (float)I[r*width + c+1] - (float)I[r*width + c-1];

		}
	}
}

void gradientImgY(unsigned short *I, float *gdy, int width, int height) {
	// Gradient is 3 neighbors wide
	
	for (int r=0; r<height; r++) {
		for (int c=0; c<width; c++) {
			
			// Boundary conditions
			if (r>0 && r<height-1)  // vertical
				gdy[r*width + c] = (float)I[(r+1)*width + c] - (float)I[(r-1)*width + c];
			
		}
	}
}

void magnitudeImgs(float *uImg, float *vImg, float *mImg, int width, int height) {
	
	// m = sqrt(u^2 + v^2)
	for (int r=0; r<height; r++) {
		for (int c=0; c<width; c++) {
			float u = uImg[r*width+c], v = vImg[r*width+c];
			mImg[r*width+c] = sqrtf(u*u+v*v);
		}
	}
	
}

void normalizeImgs( float *uImg, float *mImg,float *gImg, int width, int height) {
	
	// g = u/m
	for (int r=0; r<height; r++) {
		for (int c=0; c<width; c++) {
			gImg[r*width+c] = uImg[r*width+c]/mImg[r*width+c];
		}
	}	
	
}



int midRangeImage(unsigned short col, unsigned short row, float **exposures, 
			 int filterImgWidth, int filterImgHeight, int numExposures) {
	
	int sIndex = row*filterImgWidth + col;
	
	int found=0;
	float imgtest=0.0;
	int expIndex=0;
	
	int j;
	
	for (j=0; j<numExposures; j++) { // Loop over exposures for linear range
		imgtest = exposures[j][sIndex];
		if (imgtest < .9 && imgtest > .3) {
			expIndex = j;
			found = 1;
		}
	}
	
	if (!found) {
		return numExposures-1; // choose the last exposure
	}
	
	return expIndex;
	
}

extern "C" {
 void centroidConformalCoords(float *xv,float *yv, float *intx, float *inty, 
							 float **perpLines, int *inpolyTest, int len, FootPrint *newPoly, FootPrint*newTexcoords);

 void pntsnpoly(float *intx, float *inty, float *xv, float *yv, int len, int *test, int tlen);

 int pnpoly(int nvert, float *vertx, float *verty, float testx, float testy);

 int uniqueRowsCount(float **list, int len);

 void insertSortVectors(float ** list, int len, int vectorLen);

 int isAnyInf(float *t, int len);

 int isAnyNaN(float *t, int len);

 void IntrLine2Line(float *line0, float *line1, float *inter );

 void IntrLine2Segment(float *line, float *seg, float *inter );

 void normalizeVec(float vec[]);
 float vecMag(float vec[]);

 float dotPerp(float v1[], float v2[]);

 float dotProd(float v1[], float v2[]);

 void intersectLineEdgeTol(float * line, float *edge, float *inter, float tol );

 void intersectLines(float *line1, float*line2, float *point);

 float distancePoints(float *p1, float *p2);

 int isPointOnEdgeTol(float *point, float * edge, float tol);

 int isPointOnEdge(float *point, float * edge);

 void projPointOnLine(float *point, float *line, float *projPoint);

 float polygonArea(float x[], float y[],  int len);

 int selfitsComp( const void * a, const void *b );
 
}

extern "C" float vectorNorm(float point[]);



FootPrintDB * computeTriClip(SensorsDB * sensorSet, unsigned short *Ix, unsigned short *Iy, int spatialImgWidth, int spatialImgHeight,
					float **BoxFilterImgs, int filterImgWidth, int filterImgHeight, int filterFrameCount) {
	
	float *dudx, *dudy, *dvdx, *dvdy;
	float *ndudx, *ndudy, *ndvdx, *ndvdy;
	float *magdx, *magdy;
	
	dudx = (float *)malloc(sizeof(float)*spatialImgWidth*spatialImgHeight);
	dudy = (float *)malloc(sizeof(float)*spatialImgWidth*spatialImgHeight);
	dvdx = (float *)malloc(sizeof(float)*spatialImgWidth*spatialImgHeight);
	dvdy = (float *)malloc(sizeof(float)*spatialImgWidth*spatialImgHeight);

	gradientImgX(Ix, dudx, spatialImgWidth, spatialImgHeight);
	gradientImgY(Ix, dudy, spatialImgWidth, spatialImgHeight);
	gradientImgX(Iy, dvdx, spatialImgWidth, spatialImgHeight);
	gradientImgY(Iy, dvdy, spatialImgWidth, spatialImgHeight);

	magdx = (float *)malloc(sizeof(float)*spatialImgWidth*spatialImgHeight);
	magdy = (float *)malloc(sizeof(float)*spatialImgWidth*spatialImgHeight);
	
	magnitudeImgs(dudx, dvdx, magdx, spatialImgWidth, spatialImgHeight);
	magnitudeImgs(dudy, dvdy, magdy, spatialImgWidth, spatialImgHeight);

	ndudx = (float *)malloc(sizeof(float)*spatialImgWidth*spatialImgHeight);
	ndudy = (float *)malloc(sizeof(float)*spatialImgWidth*spatialImgHeight);
	ndvdx = (float *)malloc(sizeof(float)*spatialImgWidth*spatialImgHeight);
	ndvdy = (float *)malloc(sizeof(float)*spatialImgWidth*spatialImgHeight);
	
	normalizeImgs(dudx, magdx, ndudx, spatialImgWidth, spatialImgHeight);
	normalizeImgs(dvdx, magdx, ndvdx, spatialImgWidth, spatialImgHeight);
	normalizeImgs(dudy, magdy, ndudy, spatialImgWidth, spatialImgHeight);
	normalizeImgs(dvdy, magdy, ndvdy, spatialImgWidth, spatialImgHeight);
	
	int numPlanes = sensorSet->numSets;
	int expCount = filterFrameCount/numPlanes;
		
	// allocate clipping dataset
	FootPrintDB *lines = (FootPrintDB *)malloc(sizeof(FootPrintDB)); 
	lines->numPlanes = numPlanes;
	lines->plane = (SensorsFP*)malloc(sizeof(SensorsFP)*numPlanes);

	for (int o=0; o<numPlanes; o++) { // This may be parallelized in CUDA
		
		printf("Processing sensorSet %d\n", o);

		int numSensors = sensorSet->sensors[o].numSensors;
		int *sensorList = sensorSet->sensors[o].sensorList;
		
		lines->plane[o].numSets = numSensors;
		lines->plane[o].polys = (FootPrint*)malloc(sizeof(FootPrint)*numSensors);
		lines->plane[o].polysTex = (FootPrint*)malloc(sizeof(FootPrint)*numSensors);		
		
	
		// may be parallelized in CUDA
		
		// This is parallelized in CUDA
		for (int s=0; s<numSensors; s++) { 

			int n1r,n1c,n2r,n2c,n3r,n3c,n4c,n4r; // sensor neighborhood coordinates
			int Cr, Cc; // sensor coordinates
			int n5r,n5c,n6r,n6c,n7r,n7c,n8r,n8c; // corner sensors
			int ic,i1,i2,i3,i4,i5,i6,i7,i8; // convert coordinates to indices
			float centroid_r,centroid_c;
						
			Cc = sensorList[s*2] - 1;
			Cr = sensorList[s*2 + 1] - 1;
			
			
			// sensor neighborhood coordinates
			n1r = Cr-1;
			n1c = Cc;
			n2r = Cr;
			n2c = Cc-1;
			n3r = Cr+1;
			n3c = Cc;
			n4r = Cr;
			n4c = Cc+1;
			
			// corner sensors
			n5r = n1r;
			n5c = n2c;
			n6r = n3r;
			n6c = n2c;
			n7r = n3r;
			n7c = n4c;
			n8r = n1r;
			n8c = n4c;
			
			// convert coordinates to indices
			ic = Cr*filterImgWidth + Cc;
			i1 = n1r*filterImgWidth + n1c;
			i2 = n2r*filterImgWidth + n2c;
			i3 = n3r*filterImgWidth + n3c;
			i4 = n4r*filterImgWidth + n4c;
			
			i5 = n5r*filterImgWidth + n5c;
			i6 = n6r*filterImgWidth + n6c;
			i7 = n7r*filterImgWidth + n7c;
			i8 = n8r*filterImgWidth + n8c;
			
			// Find the best linear response  pixel
			int Iexp = midRangeImage(Cc, Cr, &BoxFilterImgs[o*expCount], filterImgWidth, filterImgHeight, expCount);

			// Warning, we assume the spatial image has the same dimensions as the filter response images
			centroid_c = 0.25*(Ix[i1] + Ix[i2] + Ix[i3] + Ix[i4]);
			centroid_r = 0.25*(Iy[i1] + Iy[i2] + Iy[i3] + Iy[i4]);
			
			// Sample image at linear exposure
			float rIi1,rIi2,rIi3,rIi4,rIi5,rIi6,rIi7,rIi8,rIic;
			
			int expIndex = o*expCount + Iexp;

			rIi1 = BoxFilterImgs[expIndex][i1];
			rIi2 = BoxFilterImgs[expIndex][i2];
			rIi3 = BoxFilterImgs[expIndex][i3];
			rIi4 = BoxFilterImgs[expIndex][i4];
			rIi5 = BoxFilterImgs[expIndex][i5];
			rIi6 = BoxFilterImgs[expIndex][i6];
			rIi7 = BoxFilterImgs[expIndex][i7];
			rIi8 = BoxFilterImgs[expIndex][i8];
			rIic = BoxFilterImgs[expIndex][ic];
			
			// Clipping computations
			float Rtotal, areaL, kIinv, uneighbors[4], vneighbors[4];
			
			Rtotal = rIi1 + rIi2 + rIic + rIi3 + rIi4;
			Rtotal = Rtotal + rIi5 + rIi6 + rIi7 + rIi8;

			uneighbors[0] = Ix[i1];
			vneighbors[0] = Iy[i1];
			uneighbors[1] = Ix[i2];
			vneighbors[1] = Iy[i2];
			uneighbors[2] = Ix[i3];
			vneighbors[2] = Iy[i3];
			uneighbors[3] = Ix[i4];
			vneighbors[3] = Iy[i4];			
			
			areaL = fabsf(polygonArea(uneighbors, vneighbors, 4));
			
			kIinv = areaL/Rtotal;
			
			float diagvec1[4], diagvec2[4], diagvec3[4], diagvec4[4];
			
			diagvec1[0] = Ix[i1];
			diagvec1[1] = Iy[i1];
			diagvec1[2] = ndudy[ic];
			diagvec1[3] = ndvdy[ic];
			
			diagvec2[0] = Ix[i2];
			diagvec2[1] = Iy[i2];
			diagvec2[2] = ndudx[ic];
			diagvec2[3] = ndvdx[ic];
			
			diagvec3[0] = Ix[i3];
			diagvec3[1] = Iy[i3];
			diagvec3[2] = -ndudy[ic];
			diagvec3[3] = -ndvdy[ic];
			
			diagvec4[0] = Ix[i4];
			diagvec4[1] = Iy[i4];
			diagvec4[2] = -ndudx[ic];
			diagvec4[3] = -ndvdx[ic];
			
			float point1[2], point2[2], point3[2], point4[2];
			
			point1[0] = Ix[i1];
			point1[1] = Iy[i1];
			point2[0] = Ix[i2];
			point2[1] = Iy[i2];
			point3[0] = Ix[i3];
			point3[1] = Iy[i3];
			point4[0] = Ix[i4];
			point4[1] = Iy[i4];
			
			float projLeft1[2],projRight1[2],projDist1,projLeft2[2],projRight2[2],projDist2;
			float projLeft3[2],projRight3[2],projDist3,projLeft4[2],projRight4[2],projDist4;
			
			float tempVecA[2], tempVecB[2];
			
			
			// Centroid based hclip
			
			// Make a line for the base of a triangle,
			// perpendicular to diagonal vector (vertex to centroid) or
			// reference triangle altitiude
				
			projPointOnLine(point4, diagvec1, projLeft1);
			projPointOnLine(point2, diagvec1, projRight1);
			tempVecA[0] = Ix[i1] - projLeft1[0]; tempVecA[1] = Iy[i1] - projLeft1[1];
			tempVecB[0] = Ix[i1] - projRight1[0]; tempVecB[1] = Iy[i1] - projRight1[1];
		//	projDist1 = fminf(vectorNorm(tempVecA), vectorNorm(tempVecB));
			projDist1 = fminf(vecMag(tempVecA), vecMag(tempVecB));

			projPointOnLine(point1, diagvec2, projLeft2);
			projPointOnLine(point3, diagvec2, projRight2);
			tempVecA[0] = Ix[i2] - projLeft2[0]; tempVecA[1] = Iy[i2] - projLeft2[1];
			tempVecB[0] = Ix[i2] - projRight2[0]; tempVecB[1] = Iy[i2] - projRight2[1];
			//projDist2 = fminf(vectorNorm(tempVecA), vectorNorm(tempVecB));
			projDist2 = fminf(vecMag(tempVecA), vecMag(tempVecB));

			projPointOnLine(point2, diagvec3, projLeft3);
			projPointOnLine(point4, diagvec3, projRight3);
			tempVecA[0] = Ix[i3] - projLeft3[0]; tempVecA[1] = Iy[i3] - projLeft3[1];
			tempVecB[0] = Ix[i3] - projRight3[0]; tempVecB[1] = Iy[i3] - projRight3[1];
			//projDist3 = fminf(vectorNorm(tempVecA), vectorNorm(tempVecB));
			projDist3 = fminf(vecMag(tempVecA), vecMag(tempVecB));

			projPointOnLine(point3, diagvec4, projLeft4);
			projPointOnLine(point1, diagvec4, projRight4);
			tempVecA[0] = Ix[i4] - projLeft4[0]; tempVecA[1] = Iy[i4] - projLeft4[1];
			tempVecB[0] = Ix[i4] - projRight4[0]; tempVecB[1] = Iy[i4] - projRight4[1];
			//projDist4 = fminf(vectorNorm(tempVecA), vectorNorm(tempVecB));
			projDist4 = fminf(vecMag(tempVecA), vecMag(tempVecB));

			// projDist is the altitude length and 
			// convex_pln is the perpendicular base line equation of the
			// reference triangle
			//
			
			float convex_pln_1[4], convex_pln_2[4], convex_pln_3[4], convex_pln_4[4];
			
			convex_pln_1[0] = Ix[i1]+projDist1*ndudy[ic]; 
			convex_pln_1[1] = Iy[i1]+projDist1*ndvdy[ic]; 
			convex_pln_1[2] = -ndvdy[ic];
			convex_pln_1[3] = ndudy[ic];
			
			convex_pln_2[0] = Ix[i2]+projDist2*ndudx[ic]; 
			convex_pln_2[1] = Iy[i2]+projDist2*ndvdx[ic]; 
			convex_pln_2[2] = -ndvdx[ic];
			convex_pln_2[3] = ndudx[ic];
			
			convex_pln_3[0] = Ix[i3]-projDist3*ndudy[ic]; 
			convex_pln_3[1] = Iy[i3]-projDist3*ndvdy[ic];
			convex_pln_3[2] = -ndvdy[ic]; 
			convex_pln_3[3] = ndudy[ic];
			
			convex_pln_4[0] = Ix[i4]-projDist4*ndudx[ic]; 
			convex_pln_4[1] = Iy[i4]-projDist4*ndvdx[ic];
			convex_pln_4[2] = -ndvdx[ic]; 
			convex_pln_4[3] = ndudx[ic];
			
			// Compute intersections between base line and edges 
			// about a  vertex of the quadrilateral
			// a Nan may result if outside edge end points
			float lineEdgeTol = .0001;
			
			float intLeft1[2], intRight1[2], intLeft2[2], intRight2[2], intLeft3[2], intRight3[2], intLeft4[2], intRight4[2];
			
			float edge41[4], edge12[4], edge23[4], edge34[4];
			
			edge41[0] = Ix[i4]; edge41[1] = Iy[i4]; edge41[2] = Ix[i1]; edge41[3] = Iy[i1];
			edge12[0] = Ix[i1]; edge12[1] = Iy[i1]; edge12[2] = Ix[i2]; edge12[3] = Iy[i2];
			edge23[0] = Ix[i2]; edge23[1] = Iy[i2]; edge23[2] = Ix[i3]; edge23[3] = Iy[i3];
			edge34[0] = Ix[i3]; edge34[1] = Iy[i3]; edge34[2] = Ix[i4]; edge34[3] = Iy[i4];

			intersectLineEdgeTol(convex_pln_1, edge41, intLeft1, lineEdgeTol);
			intersectLineEdgeTol(convex_pln_1, edge12, intRight1, lineEdgeTol);
			intersectLineEdgeTol(convex_pln_2, edge12, intLeft2, lineEdgeTol);
			intersectLineEdgeTol(convex_pln_2, edge23, intRight2, lineEdgeTol);
			intersectLineEdgeTol(convex_pln_3, edge23, intLeft3, lineEdgeTol);
			intersectLineEdgeTol(convex_pln_3, edge34, intRight3, lineEdgeTol);
			intersectLineEdgeTol(convex_pln_4, edge34, intLeft4, lineEdgeTol);
			intersectLineEdgeTol(convex_pln_4, edge41, intRight4, lineEdgeTol);

			// Compute length of base edge-to-edge intersections
			float base1, base2, base3, base4;
			
			base1 = distancePoints(intLeft1, intRight1);
			base2 = distancePoints(intLeft2, intRight2);
			base3 = distancePoints(intLeft3, intRight3);
			base4 = distancePoints(intLeft4, intRight4);
			
			// Area of a triangle is .5*base*height
			// height is magnitude of vertex-centroid vector
			float A1, A2, A3, A4;
			
			A1 = .5*base1*projDist1;
			A2 = .5*base2*projDist2;
			A3 = .5*base3*projDist3;
			A4 = .5*base4*projDist4;
			
			float hclip1, hclip2, hclip3, hclip4;
						
			hclip1 = sqrtf(rIi1*kIinv*projDist1*projDist1/A1);
			hclip2 = sqrtf(rIi2*kIinv*projDist2*projDist2/A2);
			hclip3 = sqrtf(rIi3*kIinv*projDist3*projDist3/A3);
			hclip4 = sqrtf(rIi4*kIinv*projDist4*projDist4/A4);
			
			float pln_1[4], pln_2[4], pln_3[4], pln_4[4];
			
			pln_1[0] = Ix[i1] + hclip1*ndudy[ic];
			pln_1[1] = Iy[i1] + hclip1*ndvdy[ic];
			pln_1[2] = -ndvdy[ic];
			pln_1[3] = ndudy[ic];
			
			pln_2[0] = Ix[i2] + hclip2*ndudx[ic];
			pln_2[1] = Iy[i2] + hclip2*ndvdx[ic];
			pln_2[2] = -ndvdx[ic];
			pln_2[3] = ndudx[ic];
	
			pln_3[0] = Ix[i3] - hclip3*ndudy[ic];
			pln_3[1] = Iy[i3] - hclip3*ndvdy[ic];
			pln_3[2] = -ndvdy[ic];
			pln_3[3] = ndudy[ic];
			
			pln_4[0] = Ix[i4] - hclip4*ndudx[ic];
			pln_4[1] = Iy[i4] - hclip4*ndvdx[ic];
			pln_4[2] = -ndvdx[ic];
			pln_4[3] = ndudx[ic];
			
			/* Check 
				   no response: rIic(s) = 0
				   or 
				   nonconvex quad (area == NaN) or 
				   or
				   response profile is off	
			*/
			if (rIic < .1 || isnan(areaL)) {
				lines->plane[o].polys[s].numPoints=0;
				lines->plane[o].polysTex[s].numPoints=0;

			} else {
				float xv[4], yv[4];
				
				xv[0]=Ix[i1]; xv[1]=Ix[i2]; xv[2]=Ix[i3]; xv[3]=Ix[i4];
				yv[0]=Iy[i1]; yv[1]=Iy[i2]; yv[2]=Iy[i3]; yv[3]=Iy[i4];
				
				float newpln_1[4], newpln_2[4], newpln_3[4], newpln_4[4];

				// Now we check for prevous computation faults, Nan or Inf
				
				// Constrain pln_x to vertex if nan in reference triangle
				if (isAnyNaN(pln_1,4))
					memcpy(newpln_1,convex_pln_1,sizeof(float)*4);
				else 
					memcpy(newpln_1,pln_1,sizeof(float)*4);
				
				if (isAnyNaN(pln_2,4))
					memcpy(newpln_2,convex_pln_2,sizeof(float)*4);
				else 
					memcpy(newpln_2,pln_2,sizeof(float)*4);
				
				if (isAnyNaN(pln_3,4))
					memcpy(newpln_3,convex_pln_3,sizeof(float)*4);
				else 
					memcpy(newpln_3,pln_3,sizeof(float)*4);
			
				if (isAnyNaN(pln_4,4))
					memcpy(newpln_4,convex_pln_4,sizeof(float)*4);
				else 
					memcpy(newpln_4,pln_4,sizeof(float)*4);
				
				float lineEdgeTol = .0001;
				
				float lineEdges[8][2];
				
				intersectLineEdgeTol(newpln_1, edge41, &lineEdges[0][0], lineEdgeTol);
				intersectLineEdgeTol(newpln_1, edge12, &lineEdges[1][0], lineEdgeTol);
				intersectLineEdgeTol(newpln_2, edge12, &lineEdges[2][0], lineEdgeTol);
				intersectLineEdgeTol(newpln_2, edge23, &lineEdges[3][0], lineEdgeTol);
				intersectLineEdgeTol(newpln_3, edge23, &lineEdges[4][0], lineEdgeTol);
				intersectLineEdgeTol(newpln_3, edge34, &lineEdges[5][0], lineEdgeTol);
				intersectLineEdgeTol(newpln_4, edge34, &lineEdges[6][0], lineEdgeTol);
				intersectLineEdgeTol(newpln_4, edge41, &lineEdges[7][0], lineEdgeTol);

				float xple[8], yple[8];
				
				xple[0]=lineEdges[0][0];xple[1]=lineEdges[1][0];xple[2]=lineEdges[2][0];xple[3]=lineEdges[3][0];
				yple[0]=lineEdges[0][1];yple[1]=lineEdges[1][1];yple[2]=lineEdges[2][1];yple[3]=lineEdges[3][1];

				float selfitsnlist[4][2]; // Self intersection of perpendicular lines (of base triangles)
				
				// Compute intersection of footprint border, used later for
				// polygon coords and texture coords
					
				intersectLines(newpln_1, newpln_2, &selfitsnlist[0][0]);
				intersectLines(newpln_2, newpln_3, &selfitsnlist[1][0]);
				intersectLines(newpln_3, newpln_4, &selfitsnlist[2][0]);
				intersectLines(newpln_4, newpln_1, &selfitsnlist[3][0]);

				float intx[4], inty[4]; // Self intersection list of verices

				intx[0]=selfitsnlist[0][0];intx[1]=selfitsnlist[1][0];intx[2]=selfitsnlist[2][0];intx[3]=selfitsnlist[3][0]; 
				inty[0]=selfitsnlist[0][1];inty[1]=selfitsnlist[1][1];inty[2]=selfitsnlist[2][1];inty[3]=selfitsnlist[3][1]; 
				
				float *sorteditsnlist[4];
				
				for (int l=0; l<4; l++) {
					sorteditsnlist[l] = &selfitsnlist[l][0]; // Do not use for anything other than uniqueRowsCount
				}
				
				
				if (isAnyNaN(intx, 4) || isAnyInf(intx, 4) || isAnyNaN(inty, 4) || isAnyInf(inty, 4)) {
					printf("Exception: NAN poly detected. line o=%d, sensor=%d\n",o,s);
					lines->plane[o].polys[s].numPoints = 0;
					lines->plane[o].polysTex[s].numPoints = 0;
				} else if (uniqueRowsCount(sorteditsnlist, 4)<4){
					printf("Exception: Degenerate poly detected. line o=%d, sensor=%d\n",o,s);
					lines->plane[o].polys[s].numPoints = 0;
					lines->plane[o].polysTex[s].numPoints = 0;
				} else {
					// Determine footprint polygon coords and texture map coords
					// check if perp lines self-intersect within the polygon, if they do then use
					// them as footprint coords instead of line-quad intersections
					
					int inpolyTest[4];
					
					pntsnpoly(intx, inty, xv, yv, 4, inpolyTest, 4);

					// At this point we have perpLines that intersect edges within valid range
		
					float *perpLines[4];
					
					perpLines[0] = &newpln_1[0];
					perpLines[1] = &newpln_2[0]; 
					perpLines[2] = &newpln_3[0];
					perpLines[3] = &newpln_4[0]; 

					FootPrint newPoly, newTexcoords;
					
					// Compute clipped polygon coords and texture coords
					centroidConformalCoords(xv,yv, intx, inty, perpLines, inpolyTest, 4, &newPoly, &newTexcoords);
					
					if (newPoly.numPoints==0) {
						printf("Empty new poly detected. line o=%d,  sensor=%d \n", o,s);             
						lines->plane[o].polys[s].numPoints = 0;
					} else if (newPoly.numPoints >= 4) {
						lines->plane[o].polys[s].numPoints = newPoly.numPoints;
						memcpy(&(lines->plane[o].polys[s].x), &(newPoly.x), sizeof(float)*newPoly.numPoints);
						memcpy(&(lines->plane[o].polys[s].y), &(newPoly.y), sizeof(float)*newPoly.numPoints);
						memcpy(&(lines->plane[o].polysTex[s].x), &(newTexcoords.x), sizeof(float)*newPoly.numPoints);
						memcpy(&(lines->plane[o].polysTex[s].y), &(newTexcoords.y), sizeof(float)*newPoly.numPoints);
					} else {
						printf("Lower orer new poly detected. line o=%d, sensor=%d \n",o,s);
						lines->plane[o].polys[s].numPoints = 0;
					}				
					
				}
				
			}

			
		}
			
	}
	
	free(dudx);
	free(dvdx);
	free(dudy);
	free(dvdy);
	free(magdx);
	free(magdy);
	free(ndudx);
	free(ndudy);
	free(ndvdx);
	free(ndvdy);
	
	
	return lines;
	
}

void idle(void) {
	
	
}

int tile = 0;
SensorsDB *sensorSet;
clippedGeom *h_resultsGeom;
clippedGeom *h_resultsTex;
int *h_resultsGeomLen;
experimentData *h_debug;

FootPrintDB *fpCPU ;

void display(void) 
{
	
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
	glColor3f(1.0,1.0,1.0);
	
	if (tile==0) {
		
	
	for (int p=0, c=0; p<sensorSet->numSets; p++) {

		int x, y;
		
		for (int s=0; s< sensorSet->sensors[p].numSensors; s++,c++) {
			int anyOut=false;
			if (h_resultsGeomLen[c]>0 && h_resultsGeomLen[c]<=8 )
			for (int n=0; n<h_resultsGeomLen[c]; n++) {
				x = (int)h_resultsGeom[c].x[n];
				y = (int)h_resultsGeom[c].y[n];
				if (x<1 || y<1 || x>1022 || y>766)
					anyOut = true;
			}
			
			if (anyOut) continue;
			
			glBegin(GL_LINE_LOOP);
			if (h_resultsGeomLen[c]>0 && h_resultsGeomLen[c]<=8)
			for (int n=0; n<h_resultsGeomLen[c]; n++) {
				x = (int)h_resultsGeom[c].x[n];
				y = (int)h_resultsGeom[c].y[n];
				glVertex2i(x,y);
			}
			glEnd();
		}
	}
	} else {
		
		int x, y;
		
		int p, c;
		
		for (p=0,c=0; p<sensorSet->numSets && p<tile; p++)
			c+=sensorSet->sensors[p].numSensors;
		
		for (int s=0; s< sensorSet->sensors[tile-1].numSensors; s++,c++) {
			int anyOut=false;
			if (h_resultsGeomLen[c]>0 && h_resultsGeomLen[c]<=8 )
			for (int n=0; n<h_resultsGeomLen[c]; n++) {
				x = (int)h_resultsGeom[c].x[n];
				y = (int)h_resultsGeom[c].y[n];
				if (x<1 || y<1 || x>1022 || y>766)
					anyOut = true;
			}
			
			if (anyOut) continue;
			
			glBegin(GL_LINE_LOOP);
			if (h_resultsGeomLen[c]>0 && h_resultsGeomLen[c]<=8)
			for (int n=0; n<h_resultsGeomLen[c]; n++) {
				x = (int)h_resultsGeom[c].x[n];
				y = (int)h_resultsGeom[c].y[n];
				glVertex2i(x,y);
			}
			glEnd();
		}
	}
	
	
	glutSwapBuffers();

}


void keyboard(unsigned char key, int x, int y)
{
	switch (key) {
		case 27:
			exit(0);
			break;
			
		case '+':
			if (tile < 16)
				tile++;
			printf("Tile number = %d\n",tile);
			break;
		case '-':
			if (tile>0) tile--;
			printf("Tile number = %d\n",tile);
			break;		
		default:
			break;
	}


display();	
}

/* Return 1 if the difference is negative, otherwise 0.  */
int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1)
{
    long int diff = (t2->tv_usec + 1000000 * t2->tv_sec) - (t1->tv_usec + 1000000 * t1->tv_sec);
    result->tv_sec = diff / 1000000;
    result->tv_usec = diff % 1000000;

    return (diff<0);
}

void timeval_print(struct timeval *tv)
{
    char buffer[30];
    time_t curtime;

    printf("%ld.%06ld", tv->tv_sec, tv->tv_usec);
    curtime = tv->tv_sec;
    strftime(buffer, 30, "%m-%d-%Y  %T", localtime(&curtime));
    printf(" = %s.%06ld\n", buffer, tv->tv_usec);
}

 

int main (int argc,  char ** argv) {
	
	if (argc != 5) {
		fprintf(stderr, "usage: OpticalClip sensorSet.txt Ix.tiff Iy.tiff boxfilterImages.tif \n");
		return (-1);
	}
	
	fprintf(stdout, "Loading sensorSet file %s\n",argv[1]);
	
	sensorSet = load_sensorSet(argv[1]);
	
	fprintf(stdout, "Loading Ix spatial file %s\n",argv[2]);
	
	ImageDB Ix;
	
	load_spatial_image(argv[2], &Ix);

	fprintf(stdout, "Loading Iy spatial file %s\n",argv[3]);
	
	ImageDB Iy;
	
	load_spatial_image(argv[3], &Iy);
	
	fprintf(stdout, "Loading BoxFilter images in file %s\n",argv[4]);

	ImageStackDB stackDB;
	
	load_BoxFilterImgsLinear(argv[4], &stackDB );
//	load_BoxFilterImgs(argv[4], &stackDB );
	
	int filterImgDepth = stackDB.frames;
	int filterImgWidth = stackDB.width;
	int filterImgHeight = stackDB.height;
	

	float **BoxFilterArray = (float**)malloc(sizeof(float*)*stackDB.frames);

	int pixelcount = stackDB.width*stackDB.height;
	
	for (int k=0;k<stackDB.frames; k++)
		BoxFilterArray[k] = &stackDB.imgs[pixelcount*k];
	
	int totalSensors = 0;
	
	// Count number of sensors
	for (int p=0; p<sensorSet->numSets; p++) 
	  totalSensors += sensorSet->sensors[p].numSensors;
  
  
  	printf("The total number of sensors is %d\n",totalSensors);

  	gettimeofday(&tvBegin, NULL);

  	//timeval_print(&tvBegin);

  	fpCPU = computeTriClip(sensorSet, Ix.img,Iy.img,Ix.width,Iy.height,
  				(float **)stackDB.imgs,filterImgWidth, filterImgHeight,filterImgDepth);

  	gettimeofday(&tvEnd, NULL);

  	// diff
  	timeval_subtract(&tvDiff, &tvEnd, &tvBegin);
  	printf("%ld.%06ld\n", tvDiff.tv_sec, tvDiff.tv_usec);

  	//return 0;
 	
  	
  	// Allocate device and host memory 

  	int *d_sensorX; // ints are aligned properly
  	int *d_sensorY;
  	  
  	cudaMalloc(&d_sensorX,totalSensors*sizeof(int));
  	cudaMalloc(&d_sensorY,totalSensors*sizeof(int));
  	
  	int *h_sensorX = (int *)malloc(totalSensors*sizeof(int));
  	int *h_sensorY = (int *)malloc(totalSensors*sizeof(int));
 	
  	int *d_plane; // ints are aligned properly, chars would have sufficed here
  	
  	cudaMalloc(&d_plane,totalSensors*sizeof(int));
  	
  	int *h_plane = (int *)malloc(totalSensors*sizeof(int));
  	
  	clippedGeom *d_resultsGeom;
  	clippedGeom *d_resultsTex;
  	int *d_resultsGeomLen;
  	
  	experimentData *d_debug;
  	cudaMalloc(&d_debug,totalSensors*sizeof(experimentData));

  	cudaMalloc(&d_resultsGeom,totalSensors*sizeof(clippedGeom));
   	cudaMalloc(&d_resultsTex,totalSensors*sizeof(clippedGeom));
 	cudaMalloc(&d_resultsGeomLen,totalSensors*sizeof(int));

  	h_resultsGeom = (clippedGeom *)malloc(totalSensors*sizeof(clippedGeom));
  	h_resultsTex = (clippedGeom *)malloc(totalSensors*sizeof(clippedGeom));
  	h_resultsGeomLen = (int *)malloc(totalSensors*sizeof(int));
  	
  	h_debug = (experimentData *)malloc(totalSensors*sizeof(experimentData));
  	
  	// Initialize host memory
  	for (int p=0,k=0; p<sensorSet->numSets;p++) {
  	   int pSensors = sensorSet->sensors[p].numSensors;
  	   
  	   for (int s=0; s<pSensors;s++,k++) {
  	     h_plane[k] = p;
  	     h_sensorX[k] = sensorSet->sensors[p].sensorList[s*2];
  	     h_sensorY[k] = sensorSet->sensors[p].sensorList[s*2+1];
  	   }
  	}
  	
  	//memset(h_sensorY,0,totalSensors*sizeof(int));
  	
  	
  	 unsigned int dTimer;
    

    cutilCheckError( cutCreateTimer(&dTimer) );

    
        cutilSafeCall( cudaThreadSynchronize() );
        cutilCheckError( cutResetTimer(dTimer) );
        cutilCheckError( cutStartTimer(dTimer) ); 

        // Time CUDA memory operations
  	
  	cudaArray *boxfilter_Src;
  	cudaArray *Ix_Src, *Iy_Src;
  	
  	  // Copy host data to device
  	cudaMemcpy(d_plane,h_plane,sizeof(int)*totalSensors,cudaMemcpyHostToDevice);
  	cudaMemcpy(d_sensorX,h_sensorX,sizeof(int)*totalSensors,cudaMemcpyHostToDevice);
  	cudaMemcpy(d_sensorY,h_sensorY,sizeof(int)*totalSensors,cudaMemcpyHostToDevice);

  	
  	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<unsigned char>();
  	
  	struct cudaExtent extent;
  	
  	extent.width = filterImgWidth;
  	extent.height = filterImgHeight;
  	extent.depth = filterImgDepth;
  	
  	cutilSafeCall(cudaMalloc3DArray(&boxfilter_Src,&channelDesc,extent));

  	//initCuda(boxfilter_Src,channelDesc);
  
  	 // copy data to 3D array
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr   = make_cudaPitchedPtr((void*)stackDB.imgs, filterImgWidth*sizeof(unsigned char), filterImgWidth, filterImgHeight);
    copyParams.dstArray = boxfilter_Src;
    copyParams.extent   = make_cudaExtent(filterImgWidth,filterImgHeight,filterImgDepth);
    copyParams.kind     = cudaMemcpyHostToDevice;
 
  	cutilSafeCall( cudaMemcpy3D(&copyParams) ); 
    
     	const textureReference *texRefFilterPtr, *texRefIxPtr, *texRefIyPtr;
  	
  	cudaError_t result;
  	result = cudaGetTextureReference(&texRefFilterPtr, "texRefFilter");
  	result = cudaGetTextureReference(&texRefIxPtr, "texRefIx");
  	result = cudaGetTextureReference(&texRefIyPtr, "texRefIy");
  	
  	// if not in scope?
  	// Bind memory to texRef symbol
	cudaChannelFormatDesc cDescIxy = cudaCreateChannelDesc<unsigned short>();
  	cudaMallocArray(&Ix_Src,&cDescIxy, Ix.width, Ix.height);
  	cudaMemcpy2DToArray(Ix_Src, 0,0,Ix.img, Ix.width*sizeof(unsigned short),Ix.width*sizeof(unsigned short),Ix.height, cudaMemcpyHostToDevice);
 
   	cudaMallocArray(&Iy_Src,&cDescIxy, Iy.width, Iy.height);
  	cudaMemcpy2DToArray(Iy_Src, 0,0,Iy.img, Iy.width*sizeof(unsigned short),Iy.width*sizeof(unsigned short), Ix.height, cudaMemcpyHostToDevice);
  	
  	cudaBindTextureToArray(texRefFilterPtr, boxfilter_Src, &channelDesc);
  	cudaBindTextureToArray(texRefIxPtr, Ix_Src, &cDescIxy);
  	cudaBindTextureToArray(texRefIyPtr, Iy_Src, &cDescIxy);

  	
  	
   
        cutilCheckError( cutStopTimer(dTimer) );
    
    
    
    
        printf( "Processing time MemCPY: %f (ms)\n", cutGetTimerValue(dTimer));

  
  	
  	// For 917 sensors per plane and 16 planes we can divide the total
  	// sensor count to total/(x_16 & y_16 & blockx remainder ~6
  	dim3 threadsPerBlock(16,16);
  	dim3 numBlocks(totalSensors/256+1,1,1); // plus 1 for remainder
 
 
  	// Warm up
  	callKernel(filterImgDepth, 3, Ix.width, Ix.height, 
        	d_sensorX, d_sensorY, d_plane, d_resultsGeom, d_resultsTex, d_resultsGeomLen, totalSensors,d_debug);
  	
 
        cutilSafeCall( cudaThreadSynchronize() );
        cutilCheckError( cutResetTimer(dTimer) );
        cutilCheckError( cutStartTimer(dTimer) );
  
        // call optical clip
        printf("Spawning kernel\n");
        callKernel(filterImgDepth, 3, Ix.width, Ix.height, 
        	d_sensorX, d_sensorY, d_plane, d_resultsGeom, d_resultsTex, d_resultsGeomLen, totalSensors,d_debug);
        cutilSafeCall( cudaThreadSynchronize() );

        cutilCheckError( cutStopTimer(dTimer) );
         cutilSafeCall( cudaThreadSynchronize() );
        printf("Return from kernel call\n");
                                            
    
printf( "Processing time OpticalClip: %f (ms)\n", cutGetTimerValue(dTimer));

	// Copy device data to host
  	cudaMemcpy(h_debug,d_debug,totalSensors*sizeof(experimentData),cudaMemcpyDeviceToHost);
  	cudaMemcpy(h_resultsGeom,d_resultsGeom,totalSensors*sizeof(clippedGeom),cudaMemcpyDeviceToHost);
  	cudaMemcpy(h_resultsTex,d_resultsTex,totalSensors*sizeof(clippedGeom),cudaMemcpyDeviceToHost);  
 	cudaMemcpy(h_resultsGeomLen,d_resultsGeomLen,totalSensors*sizeof(int),cudaMemcpyDeviceToHost);
  
 	//float vectorvals[][2] = {{4.8,47},{4.8,37},{11,37},{11,46}};
 	float vectorvals[4][2];
 	
 	for (int q=0; q< 4; q++) {
 	   vectorvals[q][0] = h_debug[0].selfitsnx[q];
 	   vectorvals[q][1] = h_debug[0].selfitsny[q]; 		
 	}
 	
 	float *tosort[4];
 	
 	tosort[0] = &vectorvals[0][0];
  	tosort[1] = &vectorvals[1][0];
 	tosort[2] = &vectorvals[2][0];
 	tosort[3] = &vectorvals[3][0];
 	//insertSortVectors(tosort,4,2);
 	int ur = uniqueRowsCount(tosort,4);
 	
printf("\nJust before glutInit\n");
	  
    	/* start of glut windowing and control functions */
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
	int w, h;

	w = 1024;
	h = 768;
	
	printf("\nJust before gluInitWindowSize\n");
	glutInitWindowSize(w,h); 
	printf("\nJust before gluCreateWindow\n");

	glutCreateWindow("Projector Coordinates Image");
	/*
	* Disable stuff that's likely to slow down glDrawPixels.
	* (Omit as much of this as possible, when you know in advance
	* that the OpenGL state will already be set correctly.)
	*/

	glDisable(GL_ALPHA_TEST);
	glDisable(GL_BLEND);
	//   glDisable(GL_DEPTH_TEST);

	glDisable(GL_DITHER);

	glDisable(GL_FOG);
	glDisable(GL_LIGHTING);
	glDisable(GL_LOGIC_OP);
	glDisable(GL_STENCIL_TEST);
	glDisable(GL_TEXTURE_1D);
//	glViewport(0,0,1023,767);
	
	gluOrtho2D(0,1023,0,767);
	
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutIdleFunc(idle);
printf("\nJust before GlutMainLoop\n");
	glutMainLoop();

 	cutilExit(argc, argv);

    cudaThreadExit();
  

}


