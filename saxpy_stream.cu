#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include "CycleTimer.h"
#include "saxpy.h"
__global__ void
saxpy_kernel(int N, float alpha, float* x, float* y, float* result) {
	// compute overall index from position of thread in current block,
	// and given the block we are in
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < N)
		result[index] = alpha * x[index] + y[index];
}
static inline
int getBlocks(long working_set_size, int threadsPerBlock) {
	// TODO: implement and use this interface if necessary  
}
void 
getArrays(int size, float **xarray, float **yarray, float **resultarray) {
	// TODO: implement and use this interface if necessary  
}
void 
freeArrays(float *xarray, float *yarray, float *resultarray) {
	// TODO: implement and use this interface if necessary  
}
void
saxpyCuda(long total_elems, float alpha, float* xarray, float* yarray, float* resultarray, int partitions) {
	const int threadsPerBlock = 512; // change this if necessary
	// printf("partitions is %d\n ", partitions);
	float *device_x;
	float *device_y;
	float *device_result;
	//
	// TODO: allocate device memory buffers on the GPU using
	// cudaMalloc.  The started code issues warnings on build because
	// these buffers are used in the call to saxpy_kernel below
	// without being initialized.
	//
	cudaMalloc(&device_x,total_elems*sizeof(float));
	cudaMalloc(&device_y,total_elems*sizeof(float));
	cudaMalloc(&device_result,total_elems*sizeof(float));
	cudaStream_t Stream[partitions];
	cudaError_t Result[partitions];
	int streamSize=total_elems/partitions;
	int numBlocks=(total_elems/partitions+threadsPerBlock-1)/threadsPerBlock;
	int offset;
	// start timing after allocation of device memory.
	double startTime = CycleTimer::currentSeconds();
	//double startH2D,endH2D,startD2H,endD2H,timeKernel;
	for (int i=0; i<partitions; i++) {   
		offset=i*streamSize;
		//
		// TODO: copy input arrays to the GPU using cudaMemcpy
		//
		//startH2D=CycleTimer::currentSeconds();
		Result[i]=cudaStreamCreate(&Stream[i]);
		cudaMemcpyAsync(&(device_x[offset]),&(xarray[offset]),streamSize*sizeof(float),cudaMemcpyHostToDevice,Stream[i]);
		cudaMemcpyAsync(&(device_y[offset]),&(yarray[offset]),streamSize*sizeof(float),cudaMemcpyHostToDevice,Stream[i]);
		//endH2D=CycleTimer::currentSeconds();
		//
		// TODO: insert time here to begin timing only the kernel
		//
		//double startGPUtime=CycleTimer::currentSeconds();
		// compute number of blocks and threads per block
		// run saxpy_kernel on the GPU
		saxpy_kernel<<<numBlocks, threadsPerBlock,0,Stream[i]>>>(streamSize,alpha,device_x+offset,device_y+offset,device_result+offset);
		//
		// TODO: insert timer here to time only the kernel.  Since the
		// kernel will run asynchronously with the calling CPU thread, you
		// need to call cudaDeviceSynchronize() before your timer to
		// ensure the kernel running on the GPU has completed.  (Otherwise
		// you will incorrectly observe that almost no time elapses!)
		//
		//cudaDeviceSynchronize();
		//double endGPUtime=CycleTimer::currentSeconds();
		//timeKernel=endGPUtime-startGPUtime;
		//printf("kernel %d time is %f\n",i,timeKernel);        
		cudaError_t errCode = cudaPeekAtLastError();
		if (errCode != cudaSuccess) {
			fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
		}

		//
		// TODO: copy result from GPU using cudaMemcpy
		//
		// startD2H=CycleTimer::currentSeconds();
		cudaMemcpyAsync(&(resultarray[offset]),&(device_result[offset]),streamSize*sizeof(float),cudaMemcpyDeviceToHost,Stream[i]);
		//endD2H=CycleTimer::currentSeconds(); 
		Result[i]=cudaStreamDestroy(Stream[i]);  
	}
	cudaDeviceSynchronize();
	// end timing after result has been copied back into host memory.
	// The time elapsed between startTime and endTime is the total
	// time to copy data to the GPU, run the kernel, and copy the
	// result back to the CPU

	double endTime = CycleTimer::currentSeconds();
	double overallDuration = endTime - startTime;
	totalTimeAvg+=overallDuration;
	//timeKernelAvg+=timeKernel;
	//timeCopyH2DAvg+=(endH2D-startH2D);
	//timeCopyD2HAvg+=(endD2H-startD2H);

	//
	// TODO free memory buffers on the GPU
	//
	cudaFree(device_x);
	cudaFree(device_y);
	cudaFree(device_result);
}
void
printCudaInfo() {
	// for fun, just print out some stats on the machine
	int deviceCount = 0;
	cudaError_t err = cudaGetDeviceCount(&deviceCount);
	printf("---------------------------------------------------------\n");
	printf("Found %d CUDA devices\n", deviceCount);
	for (int i=0; i<deviceCount; i++) {
		cudaDeviceProp deviceProps;
		cudaGetDeviceProperties(&deviceProps, i);
		printf("Device %d: %s\n", i, deviceProps.name);
		printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
		printf("   Global mem: %.0f MB\n",
				static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
		printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
	}
	printf("---------------------------------------------------------\n");
}

