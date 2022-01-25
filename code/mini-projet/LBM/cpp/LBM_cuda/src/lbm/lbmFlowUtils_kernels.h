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
  const int nx = params.nx;
  const int ny = params.ny;
  const int npop = LBMParams::npop;

  int base_index = 64 * blockIdx.x + threadIdx.x + blockIdx.y * nx;
  // int base_index = i + nx*j; 

  real_t rho_tmp = 0;
  real_t ux_tmp  = 0;
  real_t uy_tmp  = 0;
  for (int ipop = 0; ipop < npop; ++ipop) {
    
    int index = base_index + ipop*nx*ny;

    // Oth order moment
    rho_tmp +=             fin_d[index];

    // 1st order moment
    ux_tmp  += v(ipop,0) * fin_d[index];
    uy_tmp  += v(ipop,1) * fin_d[index];

  } // end for ipop

  rho_d[base_index] = rho_tmp;
  ux_d[base_index]  = ux_tmp/rho_tmp;
  uy_d[base_index]  = uy_tmp/rho_tmp;

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
  
  // if(threadIdx.x == 0 && blockIdx.x==0 && blockIdx.y==132)
  // printf("Th %d - Block %d,%d, - Index : %f\n",
  //        threadIdx.x, blockIdx.x, blockIdx.y, feq_d[index_f]);
  }


} // equilibrium_kernel

// ================================================================
// ================================================================
__global__ void border_outflow_kernel(const LBMParams params, 
                                      real_t* fin_d)
{

  const int nx = params.nx;
  const int ny = params.ny;

  //corresponds to j in the sequential version
  const int colidx = 64 * blockIdx.x + threadIdx.x;// * nx;// * npop;
  
  const int nxny = nx*ny;

  const int i1 = nx-1;
  const int i2 = nx-2;

  int index1 = i1 + nx * colidx;
  int index2 = i2 + nx * colidx;

  fin_d[index1 + 6*nxny] = fin_d[index2 + 6*nxny];
  fin_d[index1 + 7*nxny] = fin_d[index2 + 7*nxny];
  fin_d[index1 + 8*nxny] = fin_d[index2 + 8*nxny];

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
