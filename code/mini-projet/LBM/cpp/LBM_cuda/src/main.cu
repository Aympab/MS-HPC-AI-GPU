#include <cstdlib>
#include <string>
#include <iostream>

// =========================
// CUDA imports 
// =========================
#include <cuda_runtime.h>


#include "lbm/LBMSolver.h" 
#include "utils/monitoring/CudaTimer.h"

int main(int argc, char* argv[])
{

  // read parameter file and initialize parameter
  // parse parameters from input file
  std::string input_file = argc>1 ? std::string(argv[1]) : "flowAroundCylinder.ini";

  ConfigMap configMap(input_file);

  // test: create a LBMParams object
  LBMParams params = LBMParams();
  params.setup(configMap);

  // print parameters on screen
  params.print();

  const int nx = params.nx;
  const int ny = params.ny;

  /*N is the total number of elements, we have : 
        - 3 arrays of size nxny*npop (double or float)
        - 3 arrays of size nxny (double or float)
        - 1 array of size nxny (int)
  */
  unsigned long N = nx * ny * LBMParams::npop * 3 \
                  + nx * ny * 3; 

  unsigned long numBytes = 2.0*(N*sizeof(real_t) \
                               + nx*ny*sizeof(int));


  /*64 is the number of CUDA cores in each SM of our GPU
  we are going to divide nx by 64 to get how many block we want in our
  2D grid for the x axis. y axis will be ny, so our grid will be of
  size (nx/64, ny), each block will have 64 threads, so one SM will be filled
  with 64 threads, and these 64 threads will each one have values that are
  continuous in memory. this memory storage optimizes memory coalescing
  */
  if(params.nx % 64 != 0 || params.ny % 64 != 0)
    throw std::invalid_argument("nx and ny must be divisible by 64");

  LBMSolver* solver = new LBMSolver(params);

  CudaTimer gpuTimer;
  gpuTimer.start();

  solver->run();

  gpuTimer.stop();

  printf("GPU CODE (CUDA): %ld elements, %10.6f (s) in total, %10.6f (s) by iteration, total %f GB/s\n",
         N,
         gpuTimer.elapsed(),
         gpuTimer.elapsed()/params.maxIter,
         1e-9*numBytes*params.maxIter/gpuTimer.elapsed());


  // print bandwidth:
  {
    printf("\nbandwidth is %f GBytes (%f)/s\n",
	   1e-9*numBytes/gpuTimer.elapsed(),
	   gpuTimer.elapsed() );
  }

  // print peak bandwidth
  {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    printf("  Peak Memory Bandwidth (GB/s): %f\n",
	   2.0*deviceProp.memoryClockRate*(deviceProp.memoryBusWidth/8)/1.0e6);    
  }
  
  delete solver;
  return EXIT_SUCCESS;
}
