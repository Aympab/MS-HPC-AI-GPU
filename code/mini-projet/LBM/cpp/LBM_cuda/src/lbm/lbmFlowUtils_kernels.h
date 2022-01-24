#ifndef LBM_FLOW_UTILS_KERNELS_H
#define LBM_FLOW_UTILS_KERNELS_H

// ================================================================
// ================================================================
__global__ 
void macroscopic_kernel(const LBMParams params,
                        const velocity_array_t v,
                        const real_t* fin_d,
                        real_t* rho_d,
                        real_t* ux_d,
                        real_t* uy_d)
{

  // TODO

} // macroscopic_kernel

// ================================================================
// ================================================================
__global__ void equilibrium_kernel(const LBMParams params,
                                   const velocity_array_t v,
                                   const weights_t t,
                                   const real_t* __restrict rho_d,
                                   const real_t* __restrict ux_d,
                                   const real_t* __restrict uy_d,
                                   real_t* __restrict feq_d)
{

  const int nx = params.nx;
  const int ny = params.ny;
  const int npop = LBMParams::npop;

  //TODO 
  
  int index = 64 * blockIdx.x + threadIdx.x + blockIdx.y * nx;

  real_t usqr = 3.0 / 2 * (ux_d[index] * ux_d[index] +
                            uy_d[index] * uy_d[index]);

  for (int ipop = 0; ipop < npop; ++ipop) {
    real_t cu = 3 * (v(ipop,0) * ux_d[index] + 
                      v(ipop,1) * uy_d[index]);

    int index_f = index + ipop * nx * ny;
    feq_d[index_f] = rho_d[index] * t(ipop) * (1 + cu + 0.5*cu*cu - usqr);
  }
  // printf("Th %d - Block %d,%d, - Index : %d\n",
  //        threadIdx.x, blockIdx.x, blockIdx.y, index);


} // equilibrium_kernel

// ================================================================
// ================================================================
__global__ void border_outflow_kernel(const LBMParams params, 
                                      real_t *fin_d)
{

  // TODO

} // border_outflow_kernel

// ================================================================
// ================================================================
__global__ void border_inflow_kernel(const LBMParams params, 
                                     const real_t *fin_d,
                                     real_t *rho_d,
                                     real_t *ux_d,
                                     real_t *uy_d)
{

  // TODO

} // border_inflow_kernel

// ================================================================
// ================================================================
__global__ void update_fin_inflow_kernel(const LBMParams params, 
                                         const real_t *feq_d,
                                         real_t *fin_d)
{

  // TODO

} // border_inflow_kernel

// ================================================================
// ================================================================
__global__ void compute_collision_kernel(const LBMParams params,
                                         const real_t *fin_d, 
                                         const real_t *feq_d, 
                                         real_t *fout_d)
{

  // TODO

} // compute_collision_kernel

// ================================================================
// ================================================================
__global__ void update_obstacle_kernel(const LBMParams params,
                                       const real_t *fin_d, 
                                       const int *obstacle_d, 
                                       real_t *fout_d)
{

  // TODO

} // update_obstacle_kernel

// ================================================================
// ================================================================
__global__ void streaming_kernel(const LBMParams params,
                                 const velocity_array_t v,
                                 const real_t *fout_d,
                                 real_t *fin_d)
{

  // TODO

} // streaming_kernel

#endif // LBM_FLOW_UTILS_KERNELS_H
