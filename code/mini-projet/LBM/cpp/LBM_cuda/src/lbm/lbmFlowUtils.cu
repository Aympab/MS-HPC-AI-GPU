#include <math.h> // for M_PI = 3.1415....

#include "lbmFlowUtils.h"

#include "lbmFlowUtils_kernels.h"
#include "cuda_error.h"

// ======================================================
// ======================================================
void macroscopic(const LBMParams& params, 
                 const velocity_array_t v,
                 const real_t* fin_d,
                 real_t* rho_d,
                 real_t* ux_d,
                 real_t* uy_d)
{

  const int nx = params.nx;
  const int ny = params.ny;

  dim3 gridSize(nx/64,ny); 
  dim3 blockSize(64);

  macroscopic_kernel<<<gridSize, blockSize>>>(params,
                                              v,
                                              fin_d,
                                              rho_d,
                                              ux_d,
                                              uy_d);

  cudaDeviceSynchronize();

} // macroscopic

// ======================================================
// ======================================================
void equilibrium(const LBMParams& params, 
                 const velocity_array_t v,
                 const weights_t t,
                 const real_t* rho_d,
                 const real_t* ux_d,
                 const real_t* uy_d,
                 real_t* feq_d)
{

  const int nx = params.nx;
  const int ny = params.ny;

  /*We want nx/64 * ny blocks, each with 64 threads
  each thread will do the loop on npop = 9
  */
  dim3 gridSize(nx/64,ny); 
  dim3 blockSize(64);

  // launch the kernel
  equilibrium_kernel<<<gridSize, blockSize>>>(params, v, t,
                                              rho_d,
                                              ux_d,
                                              uy_d,
                                              feq_d);

  cudaDeviceSynchronize();
} // equilibrium

// ======================================================
// ======================================================
void init_obstacle_mask(const LBMParams& params, 
                        int* obstacle, 
                        int* obstacle_d)
{

  const int nx = params.nx;
  const int ny = params.ny;

  const real_t cx = params.cx;
  const real_t cy = params.cy;

  const real_t r = params.r;

  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {

      int index = i + nx * j;

      real_t x = 1.0*i;
      real_t y = 1.0*j;

      obstacle[index] = (x-cx)*(x-cx) + (y-cy)*(y-cy) < r*r ? 1 : 0;

    } // end for i
  } // end for j

  // copy host to device
  CUDA_API_CHECK( cudaMemcpy( obstacle_d, obstacle, nx*ny * sizeof(int),
                            cudaMemcpyHostToDevice ) );

  cudaDeviceSynchronize();
} // init_obstacle_mask

// ======================================================
// ======================================================
__host__ __device__
real_t compute_vel(int dir, int i, int j, real_t uLB, real_t ly)
{

  // flow is along X axis
  // X component is non-zero
  // Y component is always zero

  return (1-dir) * uLB * (1 + 1e-4 * sin(j/ly*2*M_PI));

} // compute_vel

// ======================================================
// ======================================================
void initialize_macroscopic_variables(const LBMParams& params, 
                                      real_t* rho, real_t* rho_d,
                                      real_t* ux, real_t* ux_d,
                                      real_t* uy, real_t* uy_d)
{

  const int nx = params.nx;
  const int ny = params.ny;

  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {

      int index = i + nx * j;

      rho[index] = 1.0;
      ux[index]  = compute_vel(0, i, j, params.uLB, params.ly);
      uy[index]  = compute_vel(1, i, j, params.uLB, params.ly);

    } // end for i
  } // end for j

  // copy host to device
  CUDA_API_CHECK( cudaMemcpy( rho_d, rho, nx*ny * sizeof(real_t),
                            cudaMemcpyHostToDevice ) );

  CUDA_API_CHECK( cudaMemcpy( ux_d, ux, nx*ny * sizeof(real_t),
                            cudaMemcpyHostToDevice ) );

  CUDA_API_CHECK( cudaMemcpy( uy_d, uy, nx*ny * sizeof(real_t),
                            cudaMemcpyHostToDevice ) );

  cudaDeviceSynchronize();
} // initialize_macroscopic_variables

// ======================================================
// ======================================================
void border_outflow(const LBMParams& params, real_t* fin_d)
{ 
  // const int nx = params.nx;
  const int ny = params.ny;

  /*Here we give 64 rows to 64 threads in 1 grid, they will all
  update one row
    fin[index1 + 6*nxny] = fin[index2 + 6*nxny];
    fin[index1 + 7*nxny] = fin[index2 + 7*nxny];
    fin[index1 + 8*nxny] = fin[index2 + 8*nxny];
  
  TODO : optimize this one by splitting both my row and cols
  */
  dim3 gridSize(ny/64); 
  dim3 blockSize(64);

  border_outflow_kernel<<<gridSize, blockSize>>>(params, fin_d);
  
  CUDA_KERNEL_CHECK("border_outflow_kernel");
  cudaDeviceSynchronize();

} // border_outflow

// ======================================================
// ======================================================
void border_inflow(const LBMParams& params, const real_t* fin_d, 
                   real_t* rho_d, real_t* ux_d, real_t* uy_d)
{

  // TODO : call kernel
} // border_inflow

// ======================================================
// ======================================================
void update_fin_inflow(const LBMParams& params, const real_t* feq_d, 
                       real_t* fin_d)
{

  // TODO : call kernel

} // update_fin_inflow
  
// ======================================================
// ======================================================
void compute_collision(const LBMParams& params, 
                       const real_t* fin_d,
                       const real_t* feq_d,
                       real_t* fout_d)
{

  // const int nx = params.nx;
  // const int ny = params.ny;

  // TODO : call kernel

} // compute_collision

// ======================================================
// ======================================================
void update_obstacle(const LBMParams &params, 
                     const real_t* fin_d,
                     const int* obstacle_d, 
                     real_t* fout_d)
{

  // const int nx = params.nx;
  // const int ny = params.ny;

  // TODO : call kernel

} // update_obstacle

// ======================================================
// ======================================================
void streaming(const LBMParams& params,
               const velocity_array_t v,
               const real_t* fout_d,
               real_t* fin_d)
{

  // const int nx = params.nx;
  // const int ny = params.ny;

  // TODO : call kernel

} // streaming
