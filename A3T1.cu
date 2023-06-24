#include "wb.h"
#include <iostream>

using namespace std;

#define MASK_WIDTH 3
#define TILE_SIZE 8

float blur[MASK_WIDTH][MASK_WIDTH] = {{0.0625, 0.125, 0.0625},{0.125, 0.25, 0.125},{0.0625, 0.125, 0.0625}};
float emboss[MASK_WIDTH][MASK_WIDTH]= {{-2, -1, 0},{-1, 1, 1},{0, 1, 2}};
float outline[MASK_WIDTH][MASK_WIDTH] = {{-1, -1, -1},{-1, 8, -1},{-1, -1, -1}};
float sharpen[MASK_WIDTH][MASK_WIDTH] = {{0, -1, 0}, {-1, 5, -1}, {0, -1, 0}};
float left_sobel[MASK_WIDTH][MASK_WIDTH] = {{1, 0, -1}, {2, 0, -2}, {1, 0, -1}};
float right_sobel[MASK_WIDTH][MASK_WIDTH] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
float top_sobel[MASK_WIDTH][MASK_WIDTH] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};
float bottom_sobel[MASK_WIDTH][MASK_WIDTH] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

__constant__ float Mc[MASK_WIDTH*MASK_WIDTH];

void conv2d_seq(float* A, float *B, float M[MASK_WIDTH][MASK_WIDTH], int height, int width){
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++){
			B[i*width+j] = 0.0f;
			for (int k = -MASK_WIDTH/2; k <= MASK_WIDTH/2; k++)
				for(int t = -MASK_WIDTH/2; t <= MASK_WIDTH/2; t++)
					B[i*width+j] += A[(((i+k)<0)? 0:((i+k)>=height)? (height-1):(i+k))*width + (((j+t)<0)? 0:((j+t)>=width)? (width-1):(j+t))] * M[k+MASK_WIDTH/2][t+MASK_WIDTH/2];
		}
}

float* linearizeKernel(float m[MASK_WIDTH][MASK_WIDTH]){
	float *p = new float[MASK_WIDTH*MASK_WIDTH];
	cout << "I'm working\n";
	for(int i =0;i < MASK_WIDTH;i++)
		for(int j =0;j < MASK_WIDTH;j++)
			p[i*MASK_WIDTH+j] = m[i][j];
	return p;
}

__global__ void conv2dKernel(float *A, float* B, int height, int width) {

	__shared__ float N_ds[TILE_SIZE][TILE_SIZE];
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int row_o = blockIdx.y * TILE_SIZE + ty;
	int col_o = blockIdx.x * TILE_SIZE + tx;
	int row_i = row_o - MASK_WIDTH/2;
	int col_i = col_o - MASK_WIDTH/2;

	if((row_o >= 0) && (row_o < height) && (col_o >= 0) && (col_o < width) )
		N_ds[ty][tx] = A[row_o*width + col_o];

	float output = 0.0f;
	//if(ty < TILE_WIDTH && tx < TILE_WIDTH){
		for(int i = 0; i < MASK_WIDTH; i++) {
			for(int j = 0; j < MASK_WIDTH; j++) {
				//float var = 0.0f;
				//if(row_o%TILE_WIDTH == 0){
				//	if()
				//}
				if(row_i + i > 0 && row_i + i < height && col_i + j > 0 && col_i + j < width)
					if(row_i+i > blockIdx.y*blockDim.y && row_i+i < (blockIdx.y+1)*blockDim.y && col_i+j > blockIdx.x*blockDim.x && col_i+j<(blockIdx.x+1)*blockDim.x)
						output += Mc[i*MASK_WIDTH+j] * N_ds[i+ty-MASK_WIDTH/2][j+tx-MASK_WIDTH/2];
						//output += Mc[i*MASK_WIDTH+j] * (i+ty < row_o)? N_ds[i- Mask_Width/2+ty][j- Mask_Width/2+tx];
					else
						output += Mc[i*MASK_WIDTH+j] * A[(row_i+i)*width+(col_i+j)];
				else
					output+=Mc[i*MASK_WIDTH+j] * A[(((row_i+i)<0)? 0:((row_i+i)>=height)? (height-1):(row_i+i))*width+(((col_i+j)<0)? 0:((col_i+j)>=width)? (width-1):(col_i+j))];
			}
		}

		if(row_o < height && col_o < width){
			B[row_o*width + col_o] = output;
		}
	}


void conv2d_par(float *A, float* B, float h_M[MASK_WIDTH][MASK_WIDTH], int height, int width) {
	int size = height*width*sizeof(float);
	float* mask = linearizeKernel(h_M);

	float *dA, *dB;
	cout << "Size is " << size << endl;
	cout << "All is well till cudaMalloc mem.\n" ;

	cudaMalloc ((void **)&dA, size);

	cout << "All is well till second cudaMalloc mem.\n";

	cudaMalloc ((void **)&dB, size);
	cout << "All is well till cudaMemcpy mem.\n";
	cudaMemcpy(dA, A, size, cudaMemcpyHostToDevice);

	dim3 dimBlock(TILE_SIZE, TILE_SIZE, 1);
	dim3 dimGrid(ceil(width/float(dimBlock.x)),ceil(height/float(dimBlock.y)),1);

	cout << "All is well till const mem.\n";
	cudaMemcpyToSymbol(Mc, mask, MASK_WIDTH*MASK_WIDTH*sizeof(float));

	clock_t start= clock();
	conv2dKernel<<<dimGrid, dimBlock>>>(dA,dB,height,width);

	cudaDeviceSynchronize();
	clock_t stop= clock();
	double time_spent = (double)(stop-start) / CLOCKS_PER_SEC;

	printf("Block %dx%d: kernel takes: %lf\n", dimBlock.x, dimBlock.y, time_spent);


	cudaMemcpy(B, dB, size, cudaMemcpyDeviceToHost);

	cudaFree(dA); 	cudaFree(dB);
}


void print2d(float *A, int height, int width) {
	for(int i = 0; i < height; i++) {
		for(int j = 0; j < width; j++)
			cout << A[i*width+j] << "\t";
		cout << endl;
	}
}


int main(){

	char* directory = (char *)"/home/ahmed531998/cuda-workspace/A3T1/img00.ppm";
	wbImage_t myImage = wbImport(directory);
	wbImage_t myImage_out = wbImport(directory);

//***for debugging***
	float A[] = {120, 60, 120, 80, 70, 90, 100, 50, 40, 90, 60, 30, 20, 100, 120, 90, 100, 50, 40, 90,120, 60, 120, 80, 70, 90, 100, 50, 40, 90, 60, 30, 20, 100, 120, 90, 100, 50, 40, 90,120, 60, 120, 80, 70, 90, 100, 50, 40, 90, 60, 30, 20, 100, 120, 90, 100, 50, 40, 90,120, 60, 120, 80, 70, 90, 100, 50, 40, 90, 60, 30, 20, 100, 120, 90, 100, 50, 40, 90};
	float *B = new float[80];
	conv2d_seq(A, B, emboss, 8, 10);
	print2d(A,8,10);
	cout << endl;
	print2d(B,8,10);
	cout << endl;cout << endl;
	conv2d_par(A, B, emboss, 8, 10);
	print2d(A,8,10);
	cout << endl;
	print2d(B,8,10);
//*/
	cout<< myImage->channels << endl;
	conv2d_seq(myImage->data, myImage_out->data, left_sobel, myImage->height, myImage->width);


	conv2d_par(myImage->data, myImage_out->data,left_sobel , myImage->height, myImage->width);
	wbExport("./out_img.ppm", myImage_out);
}
