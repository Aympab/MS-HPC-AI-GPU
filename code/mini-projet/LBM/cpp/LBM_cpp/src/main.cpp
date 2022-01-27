#include <cstdlib>
#include <string>
#include <iostream>

#include "lbm/LBMSolver.h"
#include "utils/monitoring/SimpleTimer.h"

// TODO : uncomment when building with OpenACC
//#include "utils/openacc_utils.h"

int main(int argc, char* argv[])
{

  // read parameter file and initialize parameter
  // parse parameters from input file
  std::string input_file = argc>1 ? std::string(argv[1]) : "flowAroundCylinder.ini";

  // TODO : uncomment the last two lines when activating OpenACC
  // print OpenACC version / info
  // print_openacc_version();
  //init_openacc();

  ConfigMap configMap(input_file);

  // test: create a LBMParams object
  LBMParams params = LBMParams();
  params.setup(configMap);

  // print parameters on screen
  params.print();

  // SimpleTimer *timer = new SimpleTimer;
  SimpleTimer *timer = new SimpleTimer;
  // timer.start(); 

  LBMSolver* solver = new LBMSolver(params);

  solver->run();

  delete solver;

  timer -> stop();
  std::cout << "Duration (seconds) : " << timer->elapsed() << std::endl;
  

  {
    unsigned long size_f = params.nx * params.ny * LBMParams::npop;
    unsigned long size_n = params.nx * params.ny;
    unsigned long numBytes = \
                sizeof(real_t)*(3*size_f + 3*size_n) \
              + sizeof(int)*size_n;
    numBytes *= 2.0; //for read and write
    numBytes *= params.maxIter;
    float gbs = 1e-9*numBytes/timer->elapsed();
    printf("to_parse;%d;%d;%f;%f\n", params.nx*params.ny, params.maxIter,timer->elapsed(),gbs);
  }

  return EXIT_SUCCESS;
}
