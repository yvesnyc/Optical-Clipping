#include "optical_clipping_common.h"

#include <cutil_inline.h>
#include <cutil_math.h>
#include <math.h>
#include <math_constants.h>


extern "C" {

__device__ __host__ float vectorNorm(float point[]) {
	return sqrtf(point[0]*point[0]+point[1]*point[1]);
}


__device__ __host__ float vecMag(float vec[]) {
	return sqrtf(vec[0]*vec[0] + vec[1]*vec[1]);
}


__device__ __host__  float polygonArea(float x[], float y[],  int len) {
	float sum = 0;

	int l;
	for (l=0; l<len-1; l++) {
		sum = sum + x[l]*y[l+1] - y[l]*x[l+1];
	}
	
	sum = sum + x[l]*y[0] - y[l]*x[0];
	
	return 0.5*sum;
	
}
// Can be made parallel in CUDA
__device__ __host__ void projPointOnLine(float *point, float *line, float *projPoint) {
	
	// line: x y dx dy
	// point: x y
	
	float dx, dy, tp;
	
	dx = line[2];
	dy = line[3];
	
	tp = ((point[1] - line[1])*dy + (point[0] - line[0])*dx) / (dx*dx+dy*dy);
	
	projPoint[0] = line[0] + tp*dx;
	projPoint[1] = line[1] + tp*dy;
	
}

__device__ __host__ int isPointOnEdge(float *point, float * edge) {
	
	// extract computation tolerance
	float tol = 1e-14;
	
	float x0 = edge[0];
	float y0 = edge[1];
	float dx = edge[2]-x0;
	float dy = edge[3]-y0;
	
	float xp = point[0];
	float yp = point[1];
	
	// test if point is located on supporting line
	int b1 = fabsf((xp-x0)*dy - (yp-y0)*dx)/(dx*dx+dy*dy)<tol;

	// check if point is located between edge bounds
	int ind     = fabsf(dx)>abs(dy);
	float t;
	
	if (ind) 
	t = (xp-x0)/dx;
	else
	t = (yp-y0)/dy;
	return  t>-tol & t-1<tol & b1;
}

__device__ __host__ int isPointOnEdgeTol(float *point, float * edge, float tol) {
	
	float x0 = edge[0];
	float y0 = edge[1];
	float dx = edge[2]-x0;
	float dy = edge[3]-y0;
	
	float xp = point[0];
	float yp = point[1];
	
	// test if point is located on supporting line
	int b1 = fabsf((xp-x0)*dy - (yp-y0)*dx)/(dx*dx+dy*dy)<tol;
	
	// check if point is located between edge bounds
	int ind = fabsf(dx)>fabsf(dy);
	float t;
	
	if (ind) 
		t = (xp-x0)/dx;
	else
		t = (yp-y0)/dy;
	return  t>-tol & t-1<tol & b1;
}

__device__ __host__ float distancePoints(float *p1, float *p2) {
	float dx = p1[0]-p2[0], dy = p1[1]-p2[1];
	return sqrtf(dx*dx + dy*dy);
}

__device__ __host__ void intersectLines(float *line1, float*line2, float *point) {
	float x1 = line1[0];
	float y1 = line1[1];
	float dx1 = line1[2];
	float dy1 = line1[3];
	
	float x2 = line2[0];
	float y2 = line2[1];
	float dx2 = line2[2];
	float dy2 = line2[3];
	
	// indices of parallel lines
	int par = fabsf(dx1*dy2-dx2*dy1)<1e-14;
	
	// indices of colinear lines
	int col = fabsf((x2-x1)*dy1-(y2-y1)*dx1)<1e-14 & par ;
		
	
	if (par && !col) {
#if !defined(__CUDA_ARCH__)
		point[0] = NAN;
		point[1] = NAN;
#else
		point[0] = CUDART_NAN_F;
		point[1] = CUDART_NAN_F;
#endif
		return;
	} else if (col) {
#if !defined(__CUDA_ARCH__)
		point[0] = INFINITY;
		point[1] = INFINITY;
#else
		point[0] = CUDART_INF_F;
		point[1] = CUDART_INF_F;
#endif
		return;
	}
	
	float x0,y0;

	x0 = ((y2-y1)*dx1*dx2 + x1*dy1*dx2 - x2*dy2*dx1) / (dx2*dy1-dx1*dy2) ;
	y0 = ((x2-x1)*dy1*dy2 + y1*dx1*dy2 - y2*dx2*dy1) / (dx1*dy2-dx2*dy1) ;
	

	  point[0] = x0;
	  point[1] = y0;
}

__device__ __host__ void intersectLineEdgeTol(float  line[], float edge[], float *inter, float tol ) {
	float x0 = line[0];
	float y0 = line[1];
	float dx1 = line[2];
	float dy1 = line[3];
	
	float x1 = edge[0];
	float y1 = edge[1];
	float x2 = edge[2];
	float y2 = edge[3];
	
	float dx2 = x2-x1;
	float dy2 = y2-y1;
	
	int par = fabsf(dx1*dy2-dx2*dy1)< 1e-14;
	int col = fabsf((x1-x0)*dy1-(y1-y0)*dx1)<1e-14 & par ;
	
	float xi,yi;
	

	if (par && !col) {
#if !defined(__CUDA_ARCH__)
		inter[0] = NAN;
		inter[1] = NAN;
#else
		inter[0] = CUDART_NAN_F;
		inter[1] = CUDART_NAN_F;
#endif
		return;
	} else if (col) {
#if !defined(__CUDA_ARCH__)
		inter[0] = INFINITY;
		inter[1] = INFINITY;
#else
		inter[0] = CUDART_INF_F;
		inter[1] = CUDART_INF_F;
#endif
		return;
	}
		
	xi = ((y1-y0)*dx1*dx2 + x0*dy1*dx2 - x1*dy2*dx1) / (dx2*dy1-dx1*dy2) ;
	yi = ((x1-x0)*dy1*dy2 + y0*dx1*dy2 - y1*dx2*dy1) / (dx1*dy2-dx2*dy1) ;
	
	float point[2];
	
	point[0] = xi;
	point[1] = yi;
	if (isPointOnEdgeTol(point, edge, tol)) {
		inter[0] = xi;
		inter[1] = yi;
		return;
	}

#if !defined(__CUDA_ARCH__)	
	inter[0] = NAN;
	inter[1] = NAN;
#else
	inter[0] = CUDART_NAN_F;
	inter[1] = CUDART_NAN_F;
#endif
}

__device__ __host__ float dotProd(float v1[], float v2[]) {
	return v1[0]*v2[0] + v1[1]*v2[1];
}

// dot prod with perpendicular v2
__device__ __host__ float dotPerp(float v1[], float v2[]) {
	return v1[0]*v2[1] + v1[1]*(-v2[0]);
}



__device__ __host__ void normalizeVec(float vec[]) {
	float mag = vecMag(vec);
	
	vec[0] /=mag;
	vec[1] /=mag;
}

__device__ __host__ void IntrLine2Segment(float *line, float *seg, float *inter ) {
	// The intersection of two lines is a solution to P0+s0*D0 = P1+s1*D1.
    // Rewrite this as s0*D0 - s1*D1 = P1 - P0 = Q.  If D0.Dot(Perp(D1)) = 0,
    // the lines are parallel.  Additionally, if Q.Dot(Perp(D1)) = 0, the
    // lines are the same.  If D0.Dot(Perp(D1)) is not zero, then
    //   s0 = Q.Dot(Perp(D1))/D0.Dot(Perp(D1))
    // produces the point of intersection.  Also,
    //   s1 = Q.Dot(Perp(D0))/D0.Dot(Perp(D1))
		
	// The segment is represented as (1-s)*P0+s*P1, where P0 and P1 are the
    // endpoints of the segment and 0 <= s <= 1.
    //
    // Some algorithms involving segments might prefer a centered
    // representation similar to how oriented bounding boxes are defined.
    // This representation is C+t*D, where C = (P0+P1)/2 is the center of
    // the segment, D = (P1-P0)/Length(P1-P0) is a unit-length direction
    // vector for the segment, and |t| <= e.  The value e = Length(P1-P0)/2
    // is the 'extent' (or radius or half-length) of the segment.
    	
	
	float segCenter[2];
	segCenter[0] = (seg[0] + seg[2])/2.0;
	segCenter[1] = (seg[1] + seg[3])/2.0;
	
	float originDiff[2];
	originDiff[0] = segCenter[0] - line[0];
	originDiff[1] = segCenter[1] - line[1];
	
	float lineDir[2];
	lineDir[0] = line[2];
	lineDir[1] = line[3];
	float segDir[2];
	segDir[0] = seg[2] - seg[0];
	segDir[1] = seg[3] - seg[1];
	float segMag = vecMag(segDir);
	
	
	float D0DotPerpD1 = dotPerp(lineDir, segDir);
	
	if (D0DotPerpD1 > 1e-14) {
		float invD0DotPerpD1 = 1.0/D0DotPerpD1;
		float diffDotPerpD0 = dotPerp(originDiff, lineDir);
		float diffDotPerpD1 = dotPerp(originDiff, segDir);
		
		float s0 = diffDotPerpD1*invD0DotPerpD1;
		float s1 = diffDotPerpD0*invD0DotPerpD1;
		
		float segExt = segMag/2;
		
		if (s1 <= segExt) {
			inter[0] = line[0] + s0*lineDir[0];
			inter[1] = line[1] + s0*lineDir[1];
		} else {
#if !defined(__CUDA_ARCH__)
			inter[0] = NAN;
			inter[1] = NAN;
#else
			inter[0] = CUDART_NAN_F;
			inter[1] = CUDART_NAN_F;
#endif
		}

		return;
	}
		
	// Lines are parallel
	normalizeVec(originDiff);
	
	float diffNDotPerpD1 = dotPerp(originDiff, segDir);
	
	if (diffNDotPerpD1 <= 1e-14) {
#if !defined(__CUDA_ARCH__)
		inter[0] = INFINITY;
		inter[1] = INFINITY;
#else
		inter[0] = CUDART_INF_F;
		inter[1] = CUDART_INF_F;
#endif
		return;
	}
#if !defined(__CUDA_ARCH__)	
	inter[0] = NAN;
	inter[1] = NAN;
#else
	inter[0] = CUDART_NAN_F;
	inter[1] = CUDART_NAN_F;
#endif
	return;
}

__device__ __host__ void IntrLine2Line(float *line0, float *line1, float *inter ) {
	// The intersection of two lines is a solution to P0+s0*D0 = P1+s1*D1.
    // Rewrite this as s0*D0 - s1*D1 = P1 - P0 = Q.  If D0.Dot(Perp(D1)) = 0,
    // the lines are parallel.  Additionally, if Q.Dot(Perp(D1)) = 0, the
    // lines are the same.  If D0.Dot(Perp(D1)) is not zero, then
    //   s0 = Q.Dot(Perp(D1))/D0.Dot(Perp(D1))
    // produces the point of intersection.  Also,
    //   s1 = Q.Dot(Perp(D0))/D0.Dot(Perp(D1))

	float originDiff[2];
	originDiff[0] = line1[0] - line0[0];
	originDiff[1] = line1[1] - line0[1];
	
	float lineDir0[2],lineDir1[2];
	lineDir0[0] = line0[2];
	lineDir0[1] = line0[3];
	lineDir1[0] = line1[2];
	lineDir1[1] = line1[3];
	
	float D0DotPerpD1 = dotPerp(lineDir0, lineDir1);
	
	if (D0DotPerpD1 > 1e-14) {
		float invD0DotPerpD1 = 1.0/D0DotPerpD1;
		// float diffDotPerpD0 = dotPerp(originDiff, lineDir0);
		float diffDotPerpD1 = dotPerp(originDiff, lineDir1);
		
		float s0 = diffDotPerpD1*invD0DotPerpD1;
		// float s1 = diffDotPerpD0*invD0DotPerpD1;
		
		inter[0] = line0[0] + s0*lineDir0[0];
		inter[1] = line0[1] + s0*lineDir0[1];

		return;
	}
	
	// Lines are parallel
	normalizeVec(originDiff);
	
	float diffNDotPerpD1 = dotPerp(originDiff, lineDir1);
	
	if (diffNDotPerpD1 <= 1e-14) {
#if !defined(__CUDA_ARCH__)

		inter[0] = INFINITY;
		inter[1] = INFINITY;
#else
		inter[0] = CUDART_INF_F;
		inter[1] = CUDART_INF_F;
#endif
		return;
	}
	
#if !defined(__CUDA_ARCH__)
	inter[0] = NAN;
	inter[1] = NAN;
#else
	inter[0] = CUDART_NAN_F;
	inter[1] = CUDART_NAN_F;
#endif
	return;

}

__device__ __host__ int isAnyNaN(float *t, int len) {
	int test = 0;
	
	for (int l=0; l<len; l++) {
		test += isnan(t[l]);
	}
	
	return test > 0;
}

__device__ __host__ int isAnyInf(float *t, int len) {
	int test = 0;
	
	for (int l=0; l<len; l++) {
		test += isinf(t[l]);
	}
	
	return test > 0;
}

__device__ __host__ void insertSortVectors(float ** list, int len, int vectorLen) {

	// Loop over list
	for (int index=0; index < len-1; index++) {
	   int indexOfSmallest = index;
	   
	   // Look over rest for smallest
	   for (int scan=index+1; scan<len; scan++) {
	   	int notLess = false;
	   	// Compare elements
	   	for (int t=0; t<vectorLen; t++)
	   	   if (list[indexOfSmallest][t]<list[scan][t])
	   	     break;
	   	   else if (list[indexOfSmallest][t]>list[scan][t]) {
	   	     notLess = true;
	   	     break;
	   	   }
	   	
	   	// Swap
	   	if (notLess)  {// smallest is not actually the smallest 
	   	   float *temp = list[scan];
	   	   list[scan] = list[indexOfSmallest];
	   	   list[indexOfSmallest] = temp;
	   	   indexOfSmallest = scan;
	   	}
	   }
	
	}

}

__host__ int selfitsComp ( const void * a, const void *b ) {
	float xa,ya,xb,yb;
	float **ap, **bp;
	
	ap = (float **)a;
	bp = (float **)b;
	
	xa = (*ap)[0];
	ya = (*ap)[1];
	xb = (*bp)[0];
	yb = (*bp)[1];
	
	if (xa < xb) 
		return -1;
	
	if (xa > xb)
		return 1;
	
	if (ya < yb)
		return -1;
	
	if (ya > yb)
		return 1;
	
	return 0;
}



__device__ __host__ int uniqueRowsCount(float **list, int len) {


//#if !defined(__CUDA_ARCH__)
//	qsort(list, 4, sizeof(float*), selfitsComp);
//#else

	insertSortVectors(list, len, 2);

//#endif
	int count=1; // The first is unique
	for (int l=0; l<len-1; l++) {
		
//#if !defined(__CUDA_ARCH__)
//	   if (selfitsComp(&list[l],&list[l+1]) != 0) 
//		count++;
//#else
	   int notEqual = false;	
	   for (int t=0; t<2; t++)
	     if (list[l][t] != list[l+1][t]) { // if any element is not equal
	   	 notEqual = true;
	   	 break;
	     } // floating point equal for exact match only
	     
	   if (notEqual) 
		count++;
//#endif
	}
	
	return count;
}

// Point Inclusion in Polygon Test
// Alogorithm from W. Randolph Franklin (WRF)
// 
__device__ __host__ int pnpoly(int nvert, float *vertx, float *verty, float testx, float testy)
{
	int i, j, c = 0;
	for (i = 0, j = nvert-1; i < nvert; j = i++) {
		if ( ((verty[i]>testy) != (verty[j]>testy)) &&
			(testx < (vertx[j]-vertx[i]) * (testy-verty[i]) / (verty[j]-verty[i]) + vertx[i]) )
			c = !c;
	}
	return c;
}

__device__ __host__ void pntsnpoly(float *intx, float *inty, float *xv, float *yv, int len, int *test, int tlen) {
	
	for (int l=0; l<tlen; l++) {
		test[l] = pnpoly(len,xv,yv,intx[l],inty[l]);
	}
}


__device__ __host__ void centroidConformalCoords(float *xv,float *yv, float *intx, float *inty, 
							 float **perpLines, int *inpolyTest, int len, FootPrint *newPoly, FootPrint*newTexcoords) {
	
	// Compute texture coords for geometric coords and remove duplicates
	
	float	lineEdgeTol = .0001;
	float tempVec[4];
	float aInt[2];
	float bInt[2];
	
	
	int ii = 1;
	
	for (int i=0; i<4; i++) {
		if (inpolyTest[i]) {
			newPoly->x[newPoly->numPoints] = intx[i];
			newPoly->y[newPoly->numPoints] = inty[i];

			newPoly->numPoints++;

			ii +=2; // skip
		} else {
			//
			// perp lines do not intersect inside quadrilateral so check
			// intersections with a quadrilateral edge between neighbors
				
			// Check intersect of two consecutive perp lines to shared edge
				
			// next edge intersection
			
			
			// left
			tempVec[0] = xv[i%4];
			tempVec[1] = yv[i%4];
			tempVec[2] = xv[(i+1)%4];
			tempVec[3] = yv[(i+1)%4];
			
			
			intersectLineEdgeTol(perpLines[i%4], tempVec, aInt, lineEdgeTol);
			
			if (isAnyNaN(aInt,2)) { // over shot the edge try the next one
				tempVec[0] = xv[(i+1)%4];
				tempVec[1] = yv[(i+1)%4];
				tempVec[2] = xv[(i+2)%4];
				tempVec[3] = yv[(i+2)%4];
			
				intersectLineEdgeTol(perpLines[i%4], tempVec, aInt, lineEdgeTol);
			}
			
			// right, looking backwards
			tempVec[0] = xv[i%4];
			tempVec[1] = yv[i%4];
			tempVec[2] = xv[(i+1)%4];
			tempVec[3] = yv[(i+1)%4];
			
			
			intersectLineEdgeTol(perpLines[(i+1)%4], tempVec, bInt, lineEdgeTol);

			
			if (isAnyNaN(bInt, 2)) {
				tempVec[0] = xv[(i+4-1)%4];
				tempVec[1] = yv[(i+4-1)%4];
				tempVec[2] = xv[i%4];
				tempVec[3] = yv[i%4];
				
				intersectLineEdgeTol(perpLines[(i+1)%4], tempVec, bInt, lineEdgeTol);
			}
			
			
			// detect and remove redundant points
			int dupprevbIntaInt = 0; // new a same as prev b
			int dupfirstaIntbInt = 0; // new b same as first a (cycle)
			
			// check for and remove duplicates
			if (newPoly->numPoints>0) { 
				dupprevbIntaInt = aInt[0]==newPoly->x[newPoly->numPoints-1] && aInt[1]==newPoly->y[newPoly->numPoints-1];
				dupfirstaIntbInt = bInt[0]==newPoly->x[0] && bInt[1]==newPoly->y[0];
			}
			
			
			if (!dupprevbIntaInt) {
				// no duplicate of new a so append to poly
				newPoly->x[newPoly->numPoints] = aInt[0];
				newPoly->y[newPoly->numPoints] = aInt[1];

				newPoly->numPoints++;
			}
			
			if (!dupfirstaIntbInt) {
				newPoly->x[newPoly->numPoints] = bInt[0];
				newPoly->y[newPoly->numPoints] = bInt[1];
				
				newPoly->numPoints++;
			}
			
			ii = ii + 2;
		}
		
	}
	
	
	if (isAnyNaN(&newPoly->x[0], newPoly->numPoints) || 
		isAnyNaN(&newPoly->y[0], newPoly->numPoints)) {
#if !defined(__CUDA_ARCH__)	
		printf("Mis-clip. Possible calculation problem\n");
#endif		
		newPoly->numPoints = 0;
		newTexcoords->numPoints = 0;
		return;   
	}
	
	float testVerts[8][2];
	float *testPtrs[8];
	
	// Copy clipped points to buffer for unique tests
	for (int g=0; g<newPoly->numPoints; g++) {
		testVerts[g][0] = newPoly->x[g];
		testVerts[g][1]	= newPoly->y[g];
		testPtrs[g]=&testVerts[g][0];// Used for qsort
	}
	
	if (uniqueRowsCount(&testPtrs[0], newPoly->numPoints)<4) {
#if !defined(__CUDA_ARCH__)	
		printf("Overclipped\n");
#endif
		newPoly->numPoints = 0;
		newTexcoords->numPoints = 0;
		return; 
	}
	
	// With polys we can find the centroid  and define texture mapping    
	float Cr = 0.0, Cc = 0.0;
	
	for (int g=0; g<newPoly->numPoints; g++) {
		Cr += newPoly->y[g];
		Cc += newPoly->x[g];
	}
	
	Cr *= 1.0/(float)(newPoly->numPoints);
	Cc *= 1.0/(float)(newPoly->numPoints);
	
	// Should not be needed
	if (isnan(Cc) || isnan(Cr)) {
#if !defined(__CUDA_ARCH__)			
		printf("Can't find a centroid\n");
#endif
		newPoly->numPoints = 0;
		newTexcoords->numPoints = 0;
		return;
	}
	
	for (int i=0; i<newPoly->numPoints; i++) {
		float angle = fmod(atan2(newPoly->y[i]-Cr, newPoly->x[i]-Cc) + 2*M_PI,2*M_PI);
		
		newTexcoords->x[i] = 0.5 + .5*cos(angle);
		newTexcoords->y[i] = 0.5 + 0.5*sin(angle);
	}

	newTexcoords->numPoints = newPoly->numPoints;
	
}
}

//  puts texRef into Cuda's reference name space
// texture<float,3,cudaReadModeElementType> texRefFilter;
// texture<unsigned char,3,cudaReadModeNormalizedFloat> texRefFilter;
 texture<unsigned char,3,cudaReadModeNormalizedFloat> texRefFilter;
 texture<unsigned short,2,cudaReadModeElementType> texRefIx;
 texture<unsigned short,2,cudaReadModeElementType> texRefIy;


__device__  static void CopyBack(cudaArray *boxFilter_Src, int *d_sensorX, int *d_sensorY, int *d_plane,
	clippedGeom *d_resultsGeom, int *d_resultsGeomLen, int maxSensors) {
	
	int i =  blockDim.x*blockIdx.x + threadIdx.x;

	
	if (i>=maxSensors) {
	   ++d_resultsGeomLen[0];
	return;
	}
	
	i = threadIdx.x;
	
	d_resultsGeom[i].x[0] = (float)d_sensorX[i];
	d_resultsGeom[i].y[0] = (float)d_sensorY[i];
	d_resultsGeomLen[i]=1;
	++d_resultsGeomLen[1];

}                           

__device__ __host__ float magIntVec(int x, int y ) {
	return  sqrtf((float)x*x + (float)y*y);
}



__device__ int midRangeFilterReponse(int col, int row, int pOffSet, int numExposures ) {

	int found = 0;
	float imgtest = 0.0;
	int expIndex = 0;
	
	int j;
	
	for (j=0; j<numExposures; j++) {
	  imgtest = tex3D(texRefFilter, (float)col, (float)row, (float) (pOffSet+j));
	  
	  if (imgtest < 0.9 && imgtest > 0.3) {
	    expIndex = j;
	    found = 1;
	  }
	}
	
	if (!found) return numExposures - 1;
	
	return expIndex;
}


__global__ void computeTriClipGPU(
	int filterFrame, int numExposures,int spatialImgWidth, int spatialImgHeight, int *d_sensorX, int *d_sensorY, int *d_plane,
	clippedGeom *d_resultsGeom, clippedGeom *d_resultsTex, int *d_resultsGeomLen, int numSensors, experimentData *debug) {
	
	float dudx, dudy, dvdx, dvdy;
	float ndudx, ndudy, ndvdx, ndvdy;
	float magdx, magdy;

	int sIndex =  blockDim.x*blockIdx.x + threadIdx.x;
	
	
	if (sIndex >= numSensors) return;
	
	int pIndex = d_plane[sIndex];
	
	int x = d_sensorX[sIndex]-1; // Matlab shift to C indexing
	int y = d_sensorY[sIndex]-1;
	
	dudx = (float)tex2D(texRefIx, (float)(x+1), (float)y) - (float)tex2D(texRefIx, (float)(x-1), (float)y);
	dudy = (float)tex2D(texRefIx, (float)x, (float)(y+1)) - (float)tex2D(texRefIx, (float)x, (float)(y-1));
	dvdx = (float)tex2D(texRefIy, (float)(x+1), (float)y) - (float)tex2D(texRefIy, (float)(x-1), (float)y);
	dvdy = (float)tex2D(texRefIy, (float)x, (float)(y+1)) - (float)tex2D(texRefIy, (float)x, (float)(y-1));
	
	magdx = magIntVec(dudx, dvdx);
	magdy = magIntVec(dudy, dvdy);
	
	ndudx = dudx/magdx;
	ndvdx = dvdx/magdx;
	ndudy = dudy/magdy;
	ndvdy = dvdy/magdy;
		
	
	// may be parallelized in CUDA


	int n1r,n1c,n2r,n2c,n3r,n3c,n4c,n4r; // sensor neighborhood coordinates
	int Cr, Cc; // sensor coordinates
	int n5r,n5c,n6r,n6c,n7r,n7c,n8r,n8c; // corner sensors
	float centroid_r,centroid_c;

	Cc = x;
	Cr = y;			
	
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
	
	// Find the best linear response  pixel
	int Iexp = midRangeFilterReponse(Cc, Cr, pIndex*numExposures, numExposures);


	// Sample image at linear exposure
	float rIi1,rIi2,rIi3,rIi4,rIi5,rIi6,rIi7,rIi8,rIic;
	
	int expIndex = pIndex*numExposures + Iexp;

	rIi1 = tex3D(texRefFilter, (float)n1c, (float)n1r, (float)expIndex);
	rIi2 = tex3D(texRefFilter, (float)n2c, (float)n2r, (float)expIndex);
	rIi3 = tex3D(texRefFilter, (float)n3c, (float)n3r, (float)expIndex);
	rIi4 = tex3D(texRefFilter, (float)n4c, (float)n4r, (float)expIndex);
	rIi5 = tex3D(texRefFilter, (float)n5c, (float)n5r, (float)expIndex);
	rIi6 = tex3D(texRefFilter, (float)n6c, (float)n6r, (float)expIndex);
	rIi7 = tex3D(texRefFilter, (float)n7c, (float)n7r, (float)expIndex);
	rIi8 = tex3D(texRefFilter, (float)n8c, (float)n8r, (float)expIndex);
	rIic = tex3D(texRefFilter, (float)Cc, (float)Cr, (float)expIndex);
	
	
	// Clipping computations
	float Rtotal, areaL, kIinv, uneighbors[4], vneighbors[4];
	
	Rtotal = rIi1 + rIi2 + rIic + rIi3 + rIi4;
	Rtotal = Rtotal + rIi5 + rIi6 + rIi7 + rIi8;
	
	
	uneighbors[0] = tex2D(texRefIx, (float)n1c, (float)n1r);//Ix[i1];
	vneighbors[0] = tex2D(texRefIy, (float)n1c, (float)n1r);//Iy[i1];
	uneighbors[1] = tex2D(texRefIx, (float)n2c, (float)n2r);//Ix[i2];
	vneighbors[1] = tex2D(texRefIy, (float)n2c, (float)n2r);//Iy[i2];
	uneighbors[2] = tex2D(texRefIx, (float)n3c, (float)n3r);//Ix[i3];
	vneighbors[2] = tex2D(texRefIy, (float)n3c, (float)n3r);//Iy[i3];
	uneighbors[3] = tex2D(texRefIx, (float)n4c, (float)n4r);//Ix[i4];
	vneighbors[3] = tex2D(texRefIy, (float)n4c, (float)n4r);//Iy[i4];			
	
	float point1[2], point2[2], point3[2], point4[2];
	
	point1[0] = uneighbors[0];
	point1[1] = vneighbors[0];
	point2[0] = uneighbors[1];
	point2[1] = vneighbors[1];
	point3[0] = uneighbors[2];
	point3[1] = vneighbors[2];
	point4[0] = uneighbors[3];
	point4[1] = vneighbors[3];
	
	// Warning, we assume the spatial image has the same dimensions as the filter response images
	centroid_c = 0.25*(point1[0] + point2[0] + point3[0] + point4[0]);
	centroid_r = 0.25*(point1[1] + point2[1] + point3[1] + point4[1]);

	float tempF = polygonArea(uneighbors, vneighbors, 4);
	
	areaL = fabs(tempF);
	
	kIinv = areaL/Rtotal;
		
	float diagvec1[4], diagvec2[4], diagvec3[4], diagvec4[4];
	
	diagvec1[0] = point1[0];
	diagvec1[1] = point1[1];
	diagvec1[2] = ndudy;
	diagvec1[3] = ndvdy;
	
	diagvec2[0] = point2[0];
	diagvec2[1] = point2[1];
	diagvec2[2] = ndudx;
	diagvec2[3] = ndvdx;
	
	diagvec3[0] = point3[0];
	diagvec3[1] = point3[1];
	diagvec3[2] = -ndudy;
	diagvec3[3] = -ndvdy;
	                                                     
	diagvec4[0] = point4[0];
	diagvec4[1] = point4[1];
	diagvec4[2] = -ndudx;
	diagvec4[3] = -ndvdx;
			
	float projLeft1[2],projRight1[2],projDist1,projLeft2[2],projRight2[2],projDist2;
	float projLeft3[2],projRight3[2],projDist3,projLeft4[2],projRight4[2],projDist4;
		
	
	// Centroid based hclip
	
	// Make a line for the base of a triangle,
	// perpendicular to diagonal vector (vertex to centroid) or
	// reference triangle altitiude
		
	projPointOnLine(point4, diagvec1, projLeft1);
	projPointOnLine(point2, diagvec1, projRight1);

	float tempVecA[2], tempVecB[2];

	tempVecA[0] = point1[0] - projLeft1[0]; tempVecA[1] = point1[1] - projLeft1[1];
	tempVecB[0] = point1[0] - projRight1[0]; tempVecB[1] = point1[1] - projRight1[1];
	projDist1 = fminf(vectorNorm(tempVecA), vectorNorm(tempVecB));

	projPointOnLine(point1, diagvec2, projLeft2);
	projPointOnLine(point3, diagvec2, projRight2);
	tempVecA[0] = point2[0] - projLeft2[0]; tempVecA[1] = point2[1] - projLeft2[1];
	tempVecB[0] = point2[0] - projRight2[0]; tempVecB[1] = point2[1] - projRight2[1];
	projDist2 = fminf(vectorNorm(tempVecA), vectorNorm(tempVecB));

	projPointOnLine(point2, diagvec3, projLeft3);
	projPointOnLine(point4, diagvec3, projRight3);
	tempVecA[0] = point3[0] - projLeft3[0]; tempVecA[1] = point3[1] - projLeft3[1];
	tempVecB[0] = point3[0] - projRight3[0]; tempVecB[1] = point3[1] - projRight3[1];
	projDist3 = fminf(vectorNorm(tempVecA), vectorNorm(tempVecB));

	projPointOnLine(point3, diagvec4, projLeft4);
	projPointOnLine(point1, diagvec4, projRight4);
	tempVecA[0] = point4[0] - projLeft4[0]; tempVecA[1] = point4[1] - projLeft4[1];
	tempVecB[0] = point4[0] - projRight4[0]; tempVecB[1] = point4[1] - projRight4[1];
	projDist4 = fminf(vectorNorm(tempVecA), vectorNorm(tempVecB));
	
	// projDist is the altitude length and 
	// convex_pln is the perpendicular base line equation of the
	// reference triangle
	//
		
	float convex_pln_1[4], convex_pln_2[4], convex_pln_3[4], convex_pln_4[4];
	
	convex_pln_1[0] = point1[0]+projDist1*ndudy; 
	convex_pln_1[1] = point1[1]+projDist1*ndvdy; 
	convex_pln_1[2] = -ndvdy;
	convex_pln_1[3] = ndudy;
	
	convex_pln_2[0] = point2[0]+projDist2*ndudx; 
	convex_pln_2[1] = point2[1]+projDist2*ndvdx; 
	convex_pln_2[2] = -ndvdx;
	convex_pln_2[3] = ndudx;
	
	convex_pln_3[0] = point3[0]-projDist3*ndudy; 
	convex_pln_3[1] = point3[1]-projDist3*ndvdy;
	convex_pln_3[2] = -ndvdy; 
	convex_pln_3[3] = ndudy;
	
	convex_pln_4[0] = point4[0]-projDist4*ndudx; 
	convex_pln_4[1] = point4[1]-projDist4*ndvdx;
	convex_pln_4[2] = -ndvdx; 
	convex_pln_4[3] = ndudx;
	
	
	// Compute intersections between base line and edges 
	// about a  vertex of the quadrilateral
	// a Nan may result if outside edge end points
	float lineEdgeTol = .0001;
	
	float intLeft1[2], intRight1[2], intLeft2[2], intRight2[2], intLeft3[2], intRight3[2], intLeft4[2], intRight4[2];
	
	float edge41[4], edge12[4], edge23[4], edge34[4];
	
	edge41[0] = point4[0]; edge41[1] = point4[1]; edge41[2] = point1[0]; edge41[3] = point1[1];
	edge12[0] = point1[0]; edge12[1] = point1[1]; edge12[2] = point2[0]; edge12[3] = point2[1];
	edge23[0] = point2[0]; edge23[1] = point2[1]; edge23[2] = point3[0]; edge23[3] = point3[1];
	edge34[0] = point3[0]; edge34[1] = point3[1]; edge34[2] = point4[0]; edge34[3] = point4[1];

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
	
	// Debug
	debug[sIndex].hclip[0] = hclip1;
	debug[sIndex].hclip[1] = hclip2;
	debug[sIndex].hclip[2] = hclip3;
	debug[sIndex].hclip[3] = hclip4;
	
	debug[sIndex].magdx = magdx;
	debug[sIndex].magdy = magdy;
	
	float pln_1[4], pln_2[4], pln_3[4], pln_4[4];
	
	pln_1[0] = point1[0] + hclip1*ndudy;
	pln_1[1] = point1[1] + hclip1*ndvdy;
	pln_1[2] = -ndvdy;
	pln_1[3] = ndudy;
	
	pln_2[0] = point2[0] + hclip2*ndudx;
	pln_2[1] = point2[1] + hclip2*ndvdx;
	pln_2[2] = -ndvdx;
	pln_2[3] = ndudx;

	pln_3[0] = point3[0] - hclip3*ndudy;
	pln_3[1] = point3[1] - hclip3*ndvdy;
	pln_3[2] = -ndvdy;
	pln_3[3] = ndudy;
	
	pln_4[0] = point4[0] - hclip4*ndudx;
	pln_4[1] = point4[1] - hclip4*ndvdx;
	pln_4[2] = -ndvdx;
	pln_4[3] = ndudx;
	

	
	/* Check 
		   no response: rIic(s) = 0
		   or 
		   nonconvex quad (area == NaN) or 
		   or
		   response profile is off	
	*/
	if (rIic < .1 || isnan(areaL)) {
		d_resultsGeomLen[sIndex] = -1;
	} else {
		float xv[4], yv[4];
		
		xv[0]=point1[0]; xv[1]=point2[0]; xv[2]=point3[0]; xv[3]=point4[0];
		yv[0]=point1[1]; yv[1]=point2[1]; yv[2]=point3[1]; yv[3]=point4[1];
		
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
				
		float *perpLines[4];
			
		perpLines[0] = &newpln_1[0];
		perpLines[1] = &newpln_2[0]; 
		perpLines[2] = &newpln_3[0];
		perpLines[3] = &newpln_4[0]; 
		
		for (int t=0; t<4; t++) {
		  debug[sIndex].newpln_1[t] = newpln_1[t];
		  debug[sIndex].newpln_2[t] = newpln_2[t];
		  debug[sIndex].newpln_3[t] = newpln_3[t];
		  debug[sIndex].newpln_4[t] = newpln_4[t];
		  debug[sIndex].xv[t] = xv[t];
		  debug[sIndex].yv[t] = yv[t];
		}
		
		//Debug
		for (int t=0; t<4;t++)
		  for(int q=0; q<4; q++)
		     debug[sIndex].perpLines[t][q] = perpLines[t][q];
		     
		     
		float lineEdges[8][2];
		
		float *edges[4];
			
		edges[0] = &edge12[0];
		edges[1] = &edge23[0];
		edges[2] = &edge34[0];
		edges[3] = &edge41[0];
		
		intersectLineEdgeTol(newpln_1, edge12, &lineEdges[0][0], lineEdgeTol);
		intersectLineEdgeTol(newpln_2, edge12, &lineEdges[1][0], lineEdgeTol);
		intersectLineEdgeTol(newpln_2, edge23, &lineEdges[2][0], lineEdgeTol);
		intersectLineEdgeTol(newpln_3, edge23, &lineEdges[3][0], lineEdgeTol);
		intersectLineEdgeTol(newpln_3, edge34, &lineEdges[4][0], lineEdgeTol);
		intersectLineEdgeTol(newpln_4, edge34, &lineEdges[5][0], lineEdgeTol);
		intersectLineEdgeTol(newpln_4, edge41, &lineEdges[6][0], lineEdgeTol);
		intersectLineEdgeTol(newpln_1, edge41, &lineEdges[7][0], lineEdgeTol);
		
		
		// Debug
		debug[sIndex].lineEdge[0][0] = lineEdges[0][0];
		debug[sIndex].lineEdge[0][1] = lineEdges[0][1];
		debug[sIndex].lineEdge[1][0] = lineEdges[1][0];
		debug[sIndex].lineEdge[1][1] = lineEdges[1][1];
		debug[sIndex].lineEdge[2][0] = lineEdges[2][0];
		debug[sIndex].lineEdge[2][1] = lineEdges[2][1];
		debug[sIndex].lineEdge[3][0] = lineEdges[3][0];
		debug[sIndex].lineEdge[3][1] = lineEdges[3][1];
		debug[sIndex].lineEdge[4][0] = lineEdges[4][0];
		debug[sIndex].lineEdge[4][1] = lineEdges[4][1];
		debug[sIndex].lineEdge[5][0] = lineEdges[5][0];
		debug[sIndex].lineEdge[5][1] = lineEdges[5][1];
		debug[sIndex].lineEdge[6][0] = lineEdges[6][0];
		debug[sIndex].lineEdge[6][1] = lineEdges[6][1];
		debug[sIndex].lineEdge[7][0] = lineEdges[7][0];
		debug[sIndex].lineEdge[7][1] = lineEdges[7][1];

		
		

		float xple[8], yple[8];
		
		xple[0]=lineEdges[0][0];xple[1]=lineEdges[1][0];xple[2]=lineEdges[2][0];xple[3]=lineEdges[3][0];
		xple[4]=lineEdges[4][0];xple[5]=lineEdges[5][0];xple[6]=lineEdges[6][0];xple[7]=lineEdges[7][0];
		yple[0]=lineEdges[0][1];yple[1]=lineEdges[1][1];yple[2]=lineEdges[2][1];yple[3]=lineEdges[3][1];
		yple[4]=lineEdges[4][1];yple[5]=lineEdges[5][1];yple[6]=lineEdges[6][1];yple[7]=lineEdges[7][1];

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
				
		// Debug
		debug[sIndex].selfitsnx[0] = intx[0];
		debug[sIndex].selfitsnx[1] = intx[1];
		debug[sIndex].selfitsnx[2] = intx[2];
		debug[sIndex].selfitsnx[3] = intx[3];
		debug[sIndex].selfitsny[0] = inty[0];
		debug[sIndex].selfitsny[1] = inty[1];
		debug[sIndex].selfitsny[2] = inty[2];
		debug[sIndex].selfitsny[3] = inty[3];	
		
		float *sorteditsnlist[4];
		
		for (int l=0; l<4; l++) {
			sorteditsnlist[l] = &selfitsnlist[l][0]; // Do not use for anything other than uniqueRowsCount
		}
		
		
		int uniqRows = uniqueRowsCount(sorteditsnlist, 4);
		
		debug[sIndex].uniqueSelfItsn = uniqRows;

		
		// Debug
		debug[sIndex].sortedselfitsnx[0] = sorteditsnlist[0][0];
		debug[sIndex].sortedselfitsnx[1] = sorteditsnlist[1][0];
		debug[sIndex].sortedselfitsnx[2] = sorteditsnlist[2][0];
		debug[sIndex].sortedselfitsnx[3] = sorteditsnlist[3][0];

		debug[sIndex].sortedselfitsny[0] = sorteditsnlist[0][1];
		debug[sIndex].sortedselfitsny[1] = sorteditsnlist[1][1];
		debug[sIndex].sortedselfitsny[2] = sorteditsnlist[2][1];
		debug[sIndex].sortedselfitsny[3] = sorteditsnlist[3][1];
			
		if (isAnyNaN(intx, 4) || isAnyInf(intx, 4) || isAnyNaN(inty, 4) || isAnyInf(inty, 4)) {
			//printf("Exception: CUDART_NAN_F poly detected. line o=%d, sensor=%d\n",o,s);
			d_resultsGeomLen[sIndex] = -2;
		} else if (uniqRows<4){
			//printf("Exception: Degenerate poly detected. line o=%d, sensor=%d\n",o,s);
			d_resultsGeomLen[sIndex] = -3;
		} else {
			// Determine footprint polygon coords and texture map coords
			// check if perp lines self-intersect within the polygon, if they do then use
			// them as footprint coords instead of line-quad intersections
			

			int inpolyTest[4]={0};
		
			pntsnpoly(intx, inty, xv, yv, 4, inpolyTest, 4);


			// Debug
			debug[sIndex].inpolyTest[0] = inpolyTest[0];
			debug[sIndex].inpolyTest[1] = inpolyTest[1];
			debug[sIndex].inpolyTest[2] = inpolyTest[2];			
			debug[sIndex].inpolyTest[3] = inpolyTest[3];
			
			
			// At this point we have perpLines that intersect edges within valid range


			FootPrint newPoly, newTexcoords;
			newPoly.numPoints = 0;
	
			
			for (int e=0;e<4; e++) {
				if (inpolyTest[e]) {
					newPoly.x[newPoly.numPoints] = intx[e];
					newPoly.y[newPoly.numPoints] = inty[e];
					newPoly.numPoints++;				
				} else {
				
					float aInt[2];
					float bInt[2];
	
					// Test the 2 line-edge coords
					if (!isnan(lineEdges[2*e][0]) &&   !isnan(lineEdges[2*e][1])) {
						aInt[0] = lineEdges[2*e][0];
						aInt[1] = lineEdges[2*e][1];
					} else {
					// over shot the edge try the next one
						intersectLineEdgeTol(perpLines[e], edges[(e+1)%4], aInt, lineEdgeTol);				
					}
					
					if (!isnan(lineEdges[2*e+1][0]) && !isnan(lineEdges[2*e+1][1])) {
						bInt[0] = lineEdges[2*e+1][0];
						bInt[1] = lineEdges[2*e+1][1];
					} else {
						// over shot the edge try the next one
						intersectLineEdgeTol(perpLines[e], edges[(e+2)%4], bInt, lineEdgeTol);			
					}
					
					// Add these new points as polygon vertices
					
					if (newPoly.numPoints>0) {
				
						// detect and remove redundant points
						int dupprevbIntaInt = 0; // new a same as prev b
						int dupfirstaIntbInt = 0; // new b same as first a (cycle)
						
						dupprevbIntaInt = aInt[0]==newPoly.x[newPoly.numPoints-1] && 
						     aInt[1]==newPoly.y[newPoly.numPoints-1];
						    
						dupfirstaIntbInt = bInt[0]==newPoly.x[0] && bInt[1]==newPoly.y[0];
					
						if (!dupprevbIntaInt) {
							// no duplicate of new a so append to poly
							newPoly.x[newPoly.numPoints] = aInt[0];
							newPoly.y[newPoly.numPoints] = aInt[1];
							newPoly.numPoints++;
						}
				
						if (!dupfirstaIntbInt) {
							newPoly.x[newPoly.numPoints] = bInt[0];
							newPoly.y[newPoly.numPoints] = bInt[1];
							newPoly.numPoints++;
						}
						
					} else { // First points
				
						newPoly.x[newPoly.numPoints] = aInt[0];
						newPoly.y[newPoly.numPoints] = aInt[1];
						newPoly.numPoints++;
						
						newPoly.x[newPoly.numPoints] = bInt[0];
						newPoly.y[newPoly.numPoints] = bInt[1];
						newPoly.numPoints++;		
					
					}	
				}
			}

			
			if (isAnyNaN(&newPoly.x[0], newPoly.numPoints) || 
				isAnyNaN(&newPoly.y[0], newPoly.numPoints)) {
#if !defined(__CUDA_ARCH__)	
				printf("Mis-clip. Possible calculation problem\n");
#endif		
				newPoly.numPoints = 0;
				newTexcoords.numPoints = 0;
			}
			
			float testVerts[8][2];
			float *testPtrs[8];
			
			// Copy clipped points to buffer for unique tests
			for (int g=0; g<newPoly.numPoints; g++) {
				testVerts[g][0] = newPoly.x[g];
				testVerts[g][1]	= newPoly.y[g];
				testPtrs[g]=&testVerts[g][0];// Used for qsort
			}
			
			if (uniqueRowsCount(&testPtrs[0], newPoly.numPoints)<4) {
			
#if !defined(__CUDA_ARCH__)	
				printf("Overclipped\n");
#endif
				newPoly.numPoints = 0;
				newTexcoords.numPoints = 0;
			}
			
			// With polys we can find the centroid  and define texture mapping    
			float Cr = 0.0, Cc = 0.0;
			
			for (int g=0; g<newPoly.numPoints; g++) {
				Cr += newPoly.y[g];
				Cc += newPoly.x[g];
			}
			
			if (newPoly.numPoints>0) {
			   Cr *= 1.0/(float)(newPoly.numPoints);
			   Cc *= 1.0/(float)(newPoly.numPoints);
			}
			
			// Should not be needed
			if (isnan(Cc) || isnan(Cr)) {
			
#if !defined(__CUDA_ARCH__)			
				printf("Can't find a centroid\n");	
#endif
				newPoly.numPoints = 0;
				newTexcoords.numPoints = 0;
			}
			
			for (int i=0; i<newPoly.numPoints; i++) {
				float angle = fmod(atan2(newPoly.y[i]-Cr, newPoly.x[i]-Cc) + 2*M_PI,2*M_PI);
				
				newTexcoords.x[i] = 0.5 + .5*cos(angle);
				newTexcoords.y[i] = 0.5 + 0.5*sin(angle);
			}
		
			newTexcoords.numPoints = newPoly.numPoints;
			

			//newPoly.numPoints = 0;
			// Compute clipped polygon coords and texture coords
			//centroidConformalCoords(xv,yv, intx, inty, perpLines, inpolyTest, 4, &newPoly, &newTexcoords);
			
			// Debug
			/*
			for (int m=0; m<newPoly.numPoints;m++) {
			   debug[sIndex].newPoly.x[m] = newPoly.x[m];
			   debug[sIndex].newPoly.y[m] = newPoly.y[m];
			   debug[sIndex].newTex.x[m] = newTexcoords.x[m];
			   debug[sIndex].newTex.y[m] = newTexcoords.y[m];
			}
			*/
			
			memcpy(&(debug[sIndex].newPoly.x[0]), &(newPoly.x[0]), sizeof(float)*newPoly.numPoints);
			memcpy(&(debug[sIndex].newPoly.y[0]), &(newPoly.y[0]), sizeof(float)*newPoly.numPoints);
			debug[sIndex].newPoly.numPoints = newPoly.numPoints;
			
			
			memcpy(&(debug[sIndex].newTex.x[0]), &(newTexcoords.x[0]), sizeof(float)*newPoly.numPoints);
			memcpy(&(debug[sIndex].newTex.y[0]), &(newTexcoords.y[0]), sizeof(float)*newPoly.numPoints);
			
			debug[sIndex].newPoly.numPoints = newPoly.numPoints;
			debug[sIndex].newTex.numPoints = newTexcoords.numPoints;
		
			if (newPoly.numPoints<=0) {
				//printf("Empty new poly detected. line o=%d,  sensor=%d \n", o,s);             
			   d_resultsGeomLen[sIndex] = newPoly.numPoints-10;
			} else if (newPoly.numPoints >= 4) {
				d_resultsGeomLen[sIndex] = newPoly.numPoints;
				memcpy(&(d_resultsGeom[sIndex].x[0]), &(newPoly.x), sizeof(float)*newPoly.numPoints);
				memcpy(&(d_resultsGeom[sIndex].y[0]), &(newPoly.y), sizeof(float)*newPoly.numPoints);
				memcpy(&(d_resultsTex[sIndex].x[0]), &(newTexcoords.x), sizeof(float)*newPoly.numPoints);
				memcpy(&(d_resultsTex[sIndex].y[0]), &(newTexcoords.y), sizeof(float)*newPoly.numPoints);

			} else {
				//printf("Lower orer new poly detected. line o=%d, sensor=%d \n",o,s);
				d_resultsGeomLen[sIndex] = -11;	
			}				
			
		}
			
	}
	
	
}



__global__ void testTriClip(int filterFrame, int numExposures,int spatialImgWidth, int spatialImgHeight, int *d_sensorX, int *d_sensorY, int *d_plane,
	clippedGeom *d_resultsGeom, clippedGeom *d_resultsTex, int *d_resultsGeomLen, int numSensors) {

	int sIndex =  blockDim.x*blockIdx.x + threadIdx.x;
	
	if (sIndex>=numSensors) return;
	
	int pIndex = d_plane[sIndex];

	int x = d_sensorX[sIndex]-1; //Matlab shift to C indexing
	int y = d_sensorY[sIndex]-1;
	
	
	int n1r,n1c,n2r,n2c,n3r,n3c,n4c,n4r; // sensor neighborhood coordinates
	int Cr, Cc; // sensor coordinates
	float centroid_r,centroid_c;

	Cc = x;
	Cr = y;			
	
	// sensor neighborhood coordinates
	n1r = Cr-1;
	n1c = Cc;
	n2r = Cr;
	n2c = Cc-1;
	n3r = Cr+1;
	n3c = Cc;
	n4r = Cr;
	n4c = Cc+1;
	

	
	d_resultsGeom[sIndex].x[0] = (float) tex2D(texRefIx, (float)Cc, (float)Cr);
	d_resultsGeom[sIndex].y[0] = (float) tex2D(texRefIy, (float)Cc, (float)Cr);

	d_resultsGeom[sIndex].x[1] = (float) tex2D(texRefIx, (float)n1c, (float)n1r);
	d_resultsGeom[sIndex].y[1] = (float) tex2D(texRefIy, (float)n1c, (float)n1r);

	d_resultsGeom[sIndex].x[2] = (float) tex2D(texRefIx, (float)n2c, (float)n2r);
	d_resultsGeom[sIndex].y[2] = (float) tex2D(texRefIy, (float)n2c, (float)n2r);

	d_resultsGeom[sIndex].x[3] = (float) tex2D(texRefIx, (float)n3c, (float)n3r);
	d_resultsGeom[sIndex].y[3] = (float) tex2D(texRefIy, (float)n3c, (float)n3r);
	
	d_resultsGeom[sIndex].x[4] = (float) tex2D(texRefIx, (float)n4c, (float)n4r);
	d_resultsGeom[sIndex].y[4] = (float) tex2D(texRefIy, (float)n4c, (float)n4r);

	d_resultsTex[sIndex].x[0] = (float) tex3D(texRefFilter, (float)Cc, (float)Cr, pIndex*3+0.0);
	d_resultsTex[sIndex].x[1] = (float) tex3D(texRefFilter, (float)n1c, (float)n1r, pIndex*3+0.0);
	d_resultsTex[sIndex].x[2] = (float) tex3D(texRefFilter, (float)n2c, (float)n2r, pIndex*3+0.0);
	d_resultsTex[sIndex].x[3] = (float) tex3D(texRefFilter, (float)n3c, (float)n3r, pIndex*3+0.0);
	d_resultsTex[sIndex].x[4] = (float) tex3D(texRefFilter, (float)n4c, (float)n4r, pIndex*3+0.0);

	d_resultsTex[sIndex].y[0] = (float) tex3D(texRefFilter, (float)Cc, (float)Cr, pIndex*3+1.0);
	d_resultsTex[sIndex].y[1] = (float) tex3D(texRefFilter, (float)n1c, (float)n1r, pIndex*3+1.0);
	d_resultsTex[sIndex].y[2] = (float) tex3D(texRefFilter, (float)n2c, (float)n2r, pIndex*3+1.0);
	d_resultsTex[sIndex].y[3] = (float) tex3D(texRefFilter, (float)n3c, (float)n3r, pIndex*3+1.0);
	d_resultsTex[sIndex].y[4] = (float) tex3D(texRefFilter, (float)n4c, (float)n4r, pIndex*3+1.0);
	
	d_resultsTex[sIndex].x[5] = (float) tex3D(texRefFilter, (float)Cc, (float)Cr, pIndex*3+2.0);
	d_resultsTex[sIndex].x[6] = (float) tex3D(texRefFilter, (float)n1c, (float)n1r, pIndex*3+2.0);
	d_resultsTex[sIndex].x[7] = (float) tex3D(texRefFilter, (float)n2c, (float)n2r, pIndex*3+2.0);

	d_resultsTex[sIndex].y[5] = (float) tex3D(texRefFilter, (float)n3c, (float)n3r, pIndex*3+2.0);
	d_resultsTex[sIndex].y[6] = (float) tex3D(texRefFilter, (float)n4c, (float)n4r, pIndex*3+2.0);
	d_resultsTex[sIndex].y[7] = -1.0;

}

void callKernel( int filterFrames, int numExposures,int spatialImgWidth, int spatialImgHeight,int *d_sensorX, int *d_sensorY, int* d_plane,
		clippedGeom *d_resultsGeom,clippedGeom *d_resultsTex, int *d_resultsGeomLen, int maxSensors, experimentData *debug) {
	
	dim3 threadsPerBlock(256,1);
	dim3 numBlocks(256,1);
 
	computeTriClipGPU<<<numBlocks, threadsPerBlock>>>(filterFrames, numExposures, spatialImgWidth, spatialImgHeight, 
		d_sensorX, d_sensorY, d_plane, d_resultsGeom, d_resultsTex, d_resultsGeomLen, maxSensors, debug);
	
	
//	CopyBack<<<numBlocks,threadsPerBlock>>>(boxFilter_Src,d_sensorX, d_sensorY, d_plane, d_resultsGeom, d_resultsGeomLen, maxSensors);
     cudaError_t ce = cudaGetLastError(); 
        cutilSafeCall( cudaThreadSynchronize() );

if (ce != cudaSuccess)  
fprintf( stderr, "%s: %s\n", "My kernel execution", cudaGetErrorString(ce) );
/*
	CopyBack<<<32,32>>>(boxFilter_Src,d_sensorX, d_sensorY, d_plane, d_resultsGeom, d_resultsGeomLen, maxSensors);
  ce = cudaGetLastError(); 

if (ce != cudaSuccess)  
fprintf( stderr, "%s: %s\n", "My kernel execution", cudaGetErrorString(ce) );
  */  	
}

