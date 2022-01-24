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

  SimpleTimer* timer = new SimpleTimer;
  solver->run();
  timer->stop();

  std::cout << timer->elapsed() << std::endl;
  delete timer;

  delete solver;
  return EXIT_SUCCESS;
}
