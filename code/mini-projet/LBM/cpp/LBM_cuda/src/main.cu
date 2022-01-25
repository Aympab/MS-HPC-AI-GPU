#include <cstdlib>
#include <string>
#include <iostream>

// =========================
// CUDA imports 
// =========================
#include <cuda_runtime.h>


#include "lbm/LBMSolver.h" 
#include "utils/monitoring/SimpleTimer.h"

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

  LBMSolver* solver = new LBMSolver(params);

  /*64 is the number of CUDA cores in each SM of our GPU
  we are going to divide nx by 64 to get how many block we want in our
  2D grid for the x axis. y axis will be ny, so our grid will be of
  size (nx/64, ny), each block will have 64 threads, so one SM will be filled
  with 64 threads, and these 64 threads will each one have values that are
  continuous in memory. this memory storage optimizes memory coalescing
  */
  if(params.nx % 64 != 0 || params.ny % 64 != 0)
    throw std::invalid_argument("nx and ny must be divisible by 64");

  SimpleTimer* timer = new SimpleTimer;
  solver->run();
  timer->stop();

  std::cout << "Duration (seconds) : " <<timer->elapsed() << std::endl;
  
  delete timer;
  delete solver;
  return EXIT_SUCCESS;
}
