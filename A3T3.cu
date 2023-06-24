#include <stdio.h>
#include <iostream>
#include "wb.h"

using namespace std;

#define BLOCK_WIDTH 2
#define SECTION_WIDTH 2
#define SECTION_HEIGHT 2
#define GRANULARITY 1
#define BLOCK_SIZE BLOCK_WIDTH*GRANULARITY
//tweak index calculation here to allow summing elements of the same row only





void print2d(float *A, int height, int width) {
	for(int i = 0; i < height; i++) {
		for(int j = 0; j < width; j++)
			cout << A[i*width+j] << "\t";
		cout << endl;
	}
}

//Phase 1__syncthreads();
__global__ void seqKernel(float *X, float *Y, int width, int height){
	__shared__ float XY[SECTION_HEIGHT][BLOCK_SIZE];
	int i = GRANULARITY*(blockIdx.x * blockDim.x + threadIdx.x);
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (j < height){
		//loading from X
		for (int k = 0; k < GRANULARITY; k++){
			if (i+k < width) XY[threadIdx.y][threadIdx.x*GRANULARITY+k] = X[j*width+i+k];
			else XY[threadIdx.y][threadIdx.x*GRANULARITY+k] =0;
		}

		//Sequential Scan Phase
		for (int k = 1; k < GRANULARITY; k++){
			 XY[threadIdx.y][threadIdx.x*GRANULARITY+k] += XY[threadIdx.y][threadIdx.x*GRANULARITY+k-1];
		}
		__syncthreads();



		//Kogge-Stone Phase
		for (int stride = 1; stride < blockDim.x; stride *= 2){
			if(threadIdx.x >= stride) {
				__syncthreads();
				float temp = XY[threadIdx.y][(threadIdx.x+1)*GRANULARITY-1] + XY[threadIdx.y][(threadIdx.x+1-stride)*GRANULARITY-1];
				__syncthreads();
				XY[threadIdx.y][(threadIdx.x+1)*GRANULARITY-1] = temp;
			}
		}
		__syncthreads();


		//Distribution Phase
		if(threadIdx.x != 0)
			for (int k = 0; k < GRANULARITY-1; k++){
				XY[threadIdx.y][threadIdx.x*GRANULARITY+k] += XY[threadIdx.y][threadIdx.x*GRANULARITY+k-1];
			}

			//loading into Y
			for (int k = 0; k < GRANULARITY; k++){
				if (i+k < width)	Y[j*width+i+k]= XY[threadIdx.y][threadIdx.x*GRANULARITY+k];
			}
	}
}

//Phase 2
__global__ void KoggestoneKernel(float *X, float *Y, int width, int height){
	__shared__ float XY[SECTION_HEIGHT][SECTION_WIDTH];
	int i = (blockIdx.x * blockDim.x + threadIdx.x+1) *BLOCK_SIZE-1;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if(j < height){
		if(i<width) {
			XY[threadIdx.y][threadIdx.x] = X[j*width+i];
		}
		//the code below performs iterative scan on T
		for (int stride = 1; stride < blockDim.x; stride *= 2){
			if(threadIdx.x >= stride) {
				__syncthreads();
				float temp = XY[threadIdx.y][threadIdx.x] + XY[threadIdx.y][threadIdx.x - stride];
				__syncthreads();
				XY[threadIdx.y][threadIdx.x] = temp;
			}
		}
		Y[j*width+i] = XY[threadIdx.y][threadIdx.x];
	}
}



//Phase 2
__global__ void brentKungKernel(float *X, float *Y, int width, int height){

	__shared__ float XY[SECTION_HEIGHT][SECTION_WIDTH];

	int i = (2*blockIdx.x * blockDim.x + threadIdx.x+1)* BLOCK_SIZE-1;
	int j = blockIdx.y * blockDim.y + threadIdx.y;


	if(j < height){
		if (i < width) XY[threadIdx.y][threadIdx.x] = X[j*width+i];
		if (i+blockDim.x < width) XY[threadIdx.y][threadIdx.x+blockDim.x] = X[j*width+i+blockDim.x];



	for(unsigned int stride = 1; stride <= blockDim.x; stride *= 2){
		__syncthreads();
		int index = (threadIdx.x + 1) * 2 * stride -1;
		if(index < SECTION_WIDTH){
			XY[threadIdx.y][index] += XY[threadIdx.y][index-stride];
		}
	}

	for(int stride = SECTION_WIDTH/4; stride > 0; stride /=2){
		__syncthreads();
		int index = (threadIdx.x + 1) * stride * 2 -1;
		if(index + stride < SECTION_WIDTH){
			XY[threadIdx.y][index + stride] += XY[threadIdx.y][index];
		}
	}

	__syncthreads();
	if(i < width) Y[j*width+i] = XY[threadIdx.y][threadIdx.x];
	if(i + blockDim.x < width) Y[j*width+i+blockDim.x] = XY[threadIdx.y][threadIdx.x+blockDim.x];
	}
}

//Phase 3
__global__ void distKernel(float *X, float *Y, int width, int height){
	__shared__ float XY[SECTION_HEIGHT][BLOCK_SIZE];
	__shared__ float Prev[SECTION_HEIGHT];
	if(blockIdx.x != 0){
		int i = GRANULARITY*(blockIdx.x * blockDim.x + threadIdx.x);
		int j = blockIdx.y * blockDim.y + threadIdx.y;

		if (j < height){
			if(threadIdx.x == 0){
				Prev[threadIdx.y] = X[j*width+i-1];
			}
			__syncthreads();
			//loading from X
				for (int k = 0; k < GRANULARITY; k++){
					if (i+k < width) XY[threadIdx.y][threadIdx.x*GRANULARITY+k] = X[j*width+i+k];
					else XY[threadIdx.y][threadIdx.x*GRANULARITY+k] =0;
				}

			__syncthreads();
			//Sequential Scan Phase
			if(threadIdx.x < blockDim.x-1)
				for (int k = 0; k < GRANULARITY; k++){
					XY[threadIdx.y][threadIdx.x*GRANULARITY+k] += Prev[threadIdx.y];
				}
			else
				for (int k = 0; k < GRANULARITY-1; k++){
					XY[threadIdx.y][threadIdx.x*GRANULARITY+k] += Prev[threadIdx.y];
				}
			__syncthreads();


			//loading into Y
			for (int k = 0; k < GRANULARITY; k++){
				if (i+k < width)	Y[j*width+i+k]= XY[threadIdx.y][threadIdx.x*GRANULARITY+k];
			}
		}
	}
}

void kernel_wrapper(float *I, float *S, int width, int height){
	int size = width*height*sizeof(float);
	float *dI, *dS;
	cudaMalloc((void**)(&dI), size);
	cudaMemcpy(dI, I, size, cudaMemcpyHostToDevice);
	cudaMalloc((void**)(&dS), size);

	dim3 dimBlock(BLOCK_WIDTH,SECTION_HEIGHT, 1);
	dim3 dimGrid( ceil(width/float(BLOCK_SIZE)), ceil(height/float(SECTION_HEIGHT)),1);

	dim3 dimBlock2( SECTION_WIDTH,SECTION_HEIGHT, 1);
	dim3 dimGrid2( ceil(ceil(width/float(BLOCK_SIZE))/(float(SECTION_WIDTH))), ceil(height/float(SECTION_HEIGHT)),1);
	cout << "Phase 1\n";
	seqKernel<<<dimGrid, dimBlock>>>(dI, dS, width, height);
	cudaDeviceSynchronize();
	cudaMemcpy(S, dS, size, cudaMemcpyDeviceToHost);
	print2d(S, 4, 4);
	cout << "Phase 2\n";
	KoggestoneKernel<<<dimGrid2, dimBlock2>>>(dS, dS, width, height);
	cudaDeviceSynchronize();
	cudaMemcpy(S, dS, size, cudaMemcpyDeviceToHost);
	print2d(S, 4, 4);

	cout << "Phase 3\n";
	distKernel<<<dimGrid, dimBlock>>>(dS, dS, width, height);
	cudaDeviceSynchronize();
	cout << "Done!\n";
	cudaMemcpy(S, dS, size, cudaMemcpyDeviceToHost);
	cout << "Copied Done!\n";
	cudaFree(dI);
	cudaFree(dS);
	cout << "I'm free!\n";
	return;
}

void sum_table_seq(float*I, float*S, int width, int height){
	for (int i = 0; i < height; i++){
		for (int j = 0; j < width; j++){
			S[i*width+j] = (j-1<0)?I[i*width+j]: I[i*width+j]+ S[i*width+j-1];
		}
	}

	for (int j = 0; j < width; j++){
		for (int i = 0; i < height; i++){
			S[i*width+j] = (i-1<0)?S[i*width+j]: S[i*width+j]+ S[(i-1)*width+j];
		}
	}
}

//void wrapper()

void transpose(float*I, float*S, int width, int height){
	for(int i =0;i < height;i++){
		for(int j =0; j < width; j++){
			S[j*height+i] = I[i*width+j];
		}
	}
}


int main(){
  float A[16] = {10, 50, 100, 0, 50, 255, 35, 0, 0, 16, 10, 80, 10, 20, 200, 150};
  float B[16];
  float C[16];
  float D[16];
  float E[16];
  float F[16];
  sum_table_seq(A, B, 4, 4);
  print2d(A, 4, 4);
  cout << endl << endl;
  print2d(B, 4, 4);
  cout << "Hi world\n\n\n";
  kernel_wrapper(A, C, 4, 4);
  transpose(C,D,4,4);
  kernel_wrapper(D, E, 4, 4);
  transpose(E,F,4,4);
  cout << "Bye World!\n\n\n";
  print2d(A, 4, 4);
  cout << endl << endl;
  print2d(F, 4, 4);

  //for (int i = 0; i < 16; i++)
	//  cout << B[i] << " ";
  /*wbImage_t myPic;

  myPic = wbImport((const char *)"/home/ahmed/Desktop/image.ppm");

  int width = wbImage_getWidth(myPic);
  int height = wbImage_getHeight(myPic);
  int channel = wbImage_getChannels(myPic);


  wbExport((const char *)"/home/ahmed/Desktop/imageX.ppm", myPic);

  float *data = wbImage_getData(myPic);

  printf("%d %d %d \n", width, height, channel);

  for (int i = 0; i < 1; i++){
	  for (int j = 0; j < width; j++){
		  for (int k = 0; k < 3; k++){
			  cout << data[i + width * (j + channel * k)] << " ";
		  }
	  cout << endl;
	  }
  }

//data[i][j][k] = data[i + width * (j + depth * k)]
*/
}
