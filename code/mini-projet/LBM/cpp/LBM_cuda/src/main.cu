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
  const int maxIter = params.maxIter;

  /*Here is the total number of elements, we have : 
        - 3 arrays of size nxny*npop -> size_f (double or float)
        - 3 arrays of size nxny      -> size_n (double or float)
        - 1 array of size nxny       -> size_n (int)
  */
  unsigned long size_f = nx * ny * LBMParams::npop;
  unsigned long size_n = nx * ny;

  unsigned long numBytes = \
                sizeof(real_t)*(3*size_f + 3*size_n) \
              + sizeof(int)*size_n;
  numBytes *= 2.0; //for read and write
  numBytes *= maxIter;

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

  {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    auto theorical_bandwith = 2.0*deviceProp.memoryClockRate*(deviceProp.memoryBusWidth/8)/1.0e6;
    printf("\nPeak Memory Bandwidth (GB/s): %f\n", theorical_bandwith);


    /*
      for 1 iteration, these are the number of time we use each array :
          - 7 : fin
          - 3 : rho, ux, uy, fout, feq
          - 1 : obstacle
    */
    unsigned long numOpe = \
                  (7*size_f + 3*(2*size_f) + 3*(3*size_n)) \
                + (size_n);
    numOpe *= 2.0; //for read and write
    numOpe *= maxIter-1; //to get the total number of flop

    float time = gpuTimer.elapsed();
    float gbs = 1e-9*numBytes/time;
    float proportion = 100*gbs/theorical_bandwith;


    printf("GPU CODE (CUDA): %d elements, %10.6f (s) in total, %10.6f (s) by iteration, total %f GB/s (%3.2f %% of total)\n",
          nx*ny,
          time,
          time/maxIter,
          gbs,
          proportion);

    float gflops = numOpe/(time*1e9);
    std::cout << gflops << " GFLOP/s (approx)" << std::endl;
    /*for parsing purpose to plot on python :
      to_parse;size;nbIte;time;bandwith;prop;gflop
    */
    printf("to_parse;%d;%d;%f;%f;%3.5f;%f\n", nx*ny, maxIter,time,gbs,proportion,gflops);
    // print peak bandwidth

  }

  delete solver;
  return EXIT_SUCCESS;
}
