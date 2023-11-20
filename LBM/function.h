#define pos(i,j,k) (nx*(ny*(k)+(j))+(i))
//#define Dump_Error_Val (-1./0.)
#define Dump_Error_Val NAN
#define PI 3.14159265358979323846  /* pi */
//=====

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
  if (code != cudaSuccess){
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
  }
}
//==========

int check_matrix( REAL *v, int size ){
  for ( int i = 0; i < size ; i++) {
    if (isnan(v[i])){ 
	    return(1);
	  }  
    else if (isinf(v[i])){
	  return(2);
	  }  
  }
  return(0);
}
//=====

typedef struct 
{
  //lbm
  int iter_max; // max iteration number
  int dump_frequency;// dump data frequency
  int nx; //system size, nz
  int ny; //system size, ny
  int nz; //system size, nz 
  int gpu_id; //gpu device id
  REAL w0,w1,w2,w3,w4,w5,w6,w7,w8; //weighting factor
  REAL u0,v0; //inlet velocity
  REAL nu;//kinematic viscosity
  REAL _omega;//LBM relaxation frequency
  REAL _Re; // Reynold number
} parameters;
//====================

void SET_DEFAULT_PARAMETERS(parameters *h_Pa){
  (*h_Pa).iter_max = 10;//max iteration number
  (*h_Pa).dump_frequency = 10;//dump data frequency
  (*h_Pa).nx = 101;//system size nx
  (*h_Pa).ny = 51;//system size ny
  (*h_Pa).nz = 1;//system size nz
  (*h_Pa).gpu_id = 7;//default
  //--- weighting factors
  (*h_Pa).w0 = 4.0/9.0;
  (*h_Pa).w1 = 1.0/9.0;
  (*h_Pa).w2 = 1.0/9.0;
  (*h_Pa).w3 = 1.0/9.0;
  (*h_Pa).w4 = 1.0/9.0;
  (*h_Pa).w5 = 1.0/36.0;
  (*h_Pa).w6 = 1.0/36.0;
  (*h_Pa).w7 = 1.0/36.0;
  (*h_Pa).w8 = 1.0/36.0;
  //---
  (*h_Pa).u0 = 0.1;
  (*h_Pa).v0 = 0.;
  (*h_Pa).nu = 1.;
  (*h_Pa)._omega = NAN;
  (*h_Pa)._Re = NAN;
}
//====================

void COMPUTE_PARAMETERS(parameters *h_Pa){
  (*h_Pa)._omega = 1./(3. * (*h_Pa).nu  + 0.5);
  (*h_Pa)._Re = (*h_Pa).u0 * (*h_Pa).ny/(*h_Pa).nu;
}
//====================

void DUMP_PARAMETERS(parameters *h_Pa){
  int nx = h_Pa->nx;
  int ny = h_Pa->ny;
  int nz = h_Pa->nz;
  int gpu_id = h_Pa->gpu_id;
  FILE *f = fopen("params.txt", "w" ); 
  fprintf(f, "nx\t%24d\n", nx);
  fprintf(f, "ny\t%24d\n", ny);
  fprintf(f, "nz\t%24d\n", nz);
  fprintf(f, "gpu_id\t%24d\n", gpu_id);
  fprintf(f, "iter_max\t%24d\n", h_Pa->iter_max);
  fprintf(f, "dump_frequency\t%24d\n", h_Pa->dump_frequency);
  fprintf(f, "w0\t%24f\n", h_Pa->w0);
  fprintf(f, "w1\t%24f\n", h_Pa->w1);
  fprintf(f, "w2\t%24f\n", h_Pa->w2);
  fprintf(f, "w3\t%24f\n", h_Pa->w3);
  fprintf(f, "w4\t%24f\n", h_Pa->w4);
  fprintf(f, "w5\t%24f\n", h_Pa->w5);
  fprintf(f, "w6\t%24f\n", h_Pa->w6);
  fprintf(f, "w7\t%24f\n", h_Pa->w7);
  fprintf(f, "w8\t%24f\n", h_Pa->w8);
  fprintf(f, "u0\t%24f\n", h_Pa->u0);
  fprintf(f, "v0\t%24f\n", h_Pa->v0);
  fprintf(f, "nu\t%24f\n", h_Pa->nu);
  fprintf(f, "_omega\t%24f\n", h_Pa->_omega);
  fprintf(f, "_Re\t%24f\n", h_Pa->_Re);
  fclose(f);
}
//==========

void command_line_parser(int nargs, char **args, parameters *h_Pa){// read parameters from command line  
  REAL dummy;
  int dummy_int;
  for (int i = 1; i < nargs; i++){
    if (strcmp(args[i], "-iter_max") == 0 && i < nargs -1){
	    sscanf(args[i+1], "%d", &(dummy_int));
	    h_Pa->iter_max = dummy_int;
	    i++;
	  }
    if (strcmp(args[i], "-dump_frequency") == 0 && i < nargs -1){
	    sscanf(args[i+1], "%d", &(dummy_int));
	    h_Pa->dump_frequency = dummy_int;
	    i++;
	  }
    if (strcmp(args[i], "-nx") == 0 && i < nargs -1){
	    sscanf(args[i+1], "%d", &(dummy_int));
	    h_Pa->nx = dummy_int;
	    i++;
	  }
    if (strcmp(args[i], "-ny") == 0 && i < nargs -1){
	    sscanf(args[i+1], "%d", &(dummy_int));
	    h_Pa->ny = dummy_int;
	    i++;
	  }
    if (strcmp(args[i], "-nz") == 0 && i < nargs -1){
	    sscanf(args[i+1], "%d", &(dummy_int));
	    h_Pa->nz = dummy_int;
	    i++;
	  }
    if (strcmp(args[i] , "-gpu_id") == 0 && i < nargs - 1){
	    sscanf(args[i+1], "%d", &(dummy_int));
	    h_Pa->gpu_id = dummy_int;
	    i++;
	  }
    if (strcmp(args[i] , "-u0") == 0 && i < nargs - 1){
	    sscanf(args[i+1], "%lf", &(dummy));
	    h_Pa->u0 = (REAL)dummy;
	    i++;
	  }
    if (strcmp(args[i] , "-v0") == 0 && i < nargs - 1){
	    sscanf(args[i+1], "%lf", &(dummy));
	    h_Pa->v0 = (REAL)dummy;
	    i++;
	  }
    if (strcmp(args[i] , "-nu") == 0 && i < nargs - 1){
	    sscanf(args[i+1], "%lf", &(dummy));
	    h_Pa->nu = (REAL)dummy;
	    i++;
	  } 
  }
}
//====================

__global__ void SetArrayValues(parameters *Pa, REAL *Vec, REAL Val){
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int k = threadIdx.z + blockIdx.z * blockDim.z;
  int nx = (*Pa).nx;
  int ny = (*Pa).ny;
  int nz = (*Pa).nz;
  int nx_min = 0; int nx_max = nx-1;
  int ny_min = 0; int ny_max = ny-1;
  int nz_min = 0; int nz_max = 0;
  int ALL_DOMAIN = (i >= nx_min && i <= nx_max && j >= ny_min && j <= ny_max && k >= nz_min && k <= nz_max);
  if (ALL_DOMAIN == 1){
    int index = pos(i,j,k);
    Vec[index] = Val;
    //Vec[index] = 0.0;
  }
}
//====================

__global__ void COMPUTE_RUV(parameters *Pa, REAL *gpu_mask,REAL *gpu_f0, 
          REAL *gpu_f1, REAL *gpu_f2,REAL *gpu_f3, REAL *gpu_f4,
			    REAL *gpu_f5, REAL *gpu_f6,REAL *gpu_f7, REAL *gpu_f8,
			    REAL *gpu_rho, REAL *gpu_u, REAL *gpu_v){
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int k = threadIdx.z + blockIdx.z * blockDim.z;
  int nx = (*Pa).nx;
  int ny = (*Pa).ny;
  int nz = (*Pa).nz;
  int nx_min = 0;  int nx_max = nx-1;
  int ny_min = 0; int ny_max = ny-1;
  int nz_min = 0;  int nz_max = 0;
  int INSIDE = (i > nx_min && i < nx_max && j > ny_min && j < ny_max); // 2d 
  int ALL_DOMAIN = (i >= nx_min && i <= nx_max && j >= ny_min && j <= ny_max && k >= nz_min && k <= nz_max);
  int index;
  if (ALL_DOMAIN == 1){
    index = pos(i,j,k);
    if (INSIDE ==  1){
	  //index = pos(i,j,k);
	  gpu_rho[index] = gpu_f0[index]
	    + gpu_f1[index] + gpu_f2[index] + gpu_f3[index] + gpu_f4[index]
	    + gpu_f5[index] + gpu_f6[index] + gpu_f7[index] + gpu_f8[index];
	  REAL temp;
	  temp = gpu_f5[index] + gpu_f1[index] + gpu_f8[index]
	    - gpu_f6[index] - gpu_f3[index] - gpu_f7[index];
	  gpu_u[index] = 1./gpu_rho[index] * temp;
	  temp =  gpu_f5[index] + gpu_f2[index] + gpu_f6[index]
	    - gpu_f8[index] - gpu_f4[index] - gpu_f7[index];
	  gpu_v[index] = 1./gpu_rho[index] * temp;
	  }
    /*
    else // BOUNDARIES{
	  //index = pos(i,j,k);
	  //gpu_rho[index] = NAN;
	  //gpu_u[index] = NAN;
	  //gpu_v[index] = NAN;
	  }
    */
  }
}
//=======

__global__ void Update_Streaming(parameters *Pa,REAL *gpu_f0,
			    REAL *gpu_f1,REAL *gpu_f2,REAL *gpu_f3,REAL *gpu_f4,
			    REAL *gpu_f5,REAL *gpu_f6,REAL *gpu_f7,REAL *gpu_f8,REAL *gpu_tf0,
          REAL *gpu_tf1,REAL *gpu_tf2,REAL *gpu_tf3,REAL *gpu_tf4,
			    REAL *gpu_tf5,REAL *gpu_tf6,REAL *gpu_tf7,REAL *gpu_tf8){
  // shift_i,shift_j, shift_k are streaming velocity vector.
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int k = threadIdx.z + blockIdx.z * blockDim.z;
  int nx = (*Pa).nx;
  int ny = (*Pa).ny;
  int nz = (*Pa).nz;
  int nx_min = 0;  int nx_max = nx-1;
  int ny_min = 0;  int ny_max = ny-1;
  int nz_min = 0;  int nz_max = 0;
  int index = pos(i,j,k) ;
  int ii,jj;
  int ALL_DOMAIN = (i >= nx_min && i <= nx_max && j >= ny_min && j <= ny_max && k >= nz_min && k <= nz_max);
  if (ALL_DOMAIN){
    gpu_f0[index]=gpu_tf0[pos(i,j,k)];
  }
  ii = i-1;
  if (ii >= nx_min && ii <= nx_max && j >= ny_min && j <= ny_max && k >= nz_min && k <= nz_max){
    gpu_f1[index]=gpu_tf1[pos(ii,j,k)];
  }
  jj = j-1;
  if (i >= nx_min && i <= nx_max && jj >= ny_min && jj <= ny_max && k >= nz_min && k <= nz_max){
    gpu_f2[index]=gpu_tf2[pos(i,jj,k)];
  }
  ii = i+1;
  if (ii >= nx_min && ii <= nx_max && j >= ny_min && j <= ny_max && k >= nz_min && k <= nz_max){
    gpu_f3[index]=gpu_tf3[pos(ii,j,k)];
  }    
  jj = j+1;    
  if (i >= nx_min && i <= nx_max && jj >= ny_min && jj <= ny_max && k >= nz_min && k <= nz_max){
    gpu_f4[index]=gpu_tf4[pos(i,jj,k)];
  }
  ii = i - 1 ;
  jj = j - 1 ;
  if (ii >= nx_min && ii <= nx_max && jj >= ny_min && jj <= ny_max && k >= nz_min && k <= nz_max){
    gpu_f5[index]=gpu_tf5[pos(ii,jj,k)];
  }
  ii = i + 1 ;
  jj = j - 1 ;        
  if (ii >= nx_min && ii <= nx_max && jj >= ny_min && jj <= ny_max && k >= nz_min && k <= nz_max){
    gpu_f6[index]=gpu_tf6[pos(ii,jj,k)];
  }  
  ii = i + 1 ;
  jj = j + 1 ;        
  if (ii >= nx_min && ii <= nx_max && jj >= ny_min && jj <= ny_max && k >= nz_min && k <= nz_max){
    gpu_f7[index]=gpu_tf7[pos(ii,jj,k)];
  }  
  ii = i - 1 ;
  jj = j + 1 ;
  if (ii >= nx_min && ii <= nx_max && jj >= ny_min && jj <= ny_max && k >= nz_min && k <= nz_max){
    gpu_f8[index]=gpu_tf8[pos(ii,jj,k)];
  }
}
//======

__global__ void Update_Collision(REAL *gpu_f0,
				 REAL *gpu_f1, REAL *gpu_f2,REAL *gpu_f3, REAL *gpu_f4,
				 REAL *gpu_f5, REAL *gpu_f6,REAL *gpu_f7, REAL *gpu_f8,
				 REAL *gpu_rho, REAL *gpu_u, REAL *gpu_v, parameters *Pa){
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int k = threadIdx.z + blockIdx.z * blockDim.z;
  int nx = (*Pa).nx;
  int ny = (*Pa).ny;
  int nz = (*Pa).nz;
  int nx_min = 0;  int nx_max = nx-1;
  int ny_min = 0; int ny_max = ny-1;
  int nz_min = 0;  int nz_max = 0;
  int ALL_DOMAIN = (i >= nx_min && i <= nx_max && j >= ny_min && j <= ny_max && k >= nz_min && k <= nz_max);
  if (ALL_DOMAIN == 1){
    int index = pos(i,j,k);
    //weight
    REAL w0 = (*Pa).w0;
    REAL w1 = (*Pa).w1;
    REAL w2 = (*Pa).w2;
    REAL w3 = (*Pa).w3;
    REAL w4 = (*Pa).w4;
    REAL w5 = (*Pa).w5;
    REAL w6 = (*Pa).w6;
    REAL w7 = (*Pa).w7;
    REAL w8 = (*Pa).w8;
    //---
    REAL omega = (*Pa)._omega;
    REAL rho = gpu_rho[index];
    REAL u = gpu_u[index];
    REAL v = gpu_v[index];
    REAL temp = 3./2. * (u * u + v * v);
    REAL u_2 = u * u;
    REAL v_2 = v * v;
	  //Compute the equilibrium distrbution function
    REAL f0_eq = w0 * rho * (1. - temp);
    REAL f1_eq = w1 * rho * (1. + 3. * u + 4.5 * u_2 - temp);
    REAL f2_eq = w2 * rho * (1. + 3. * v + 4.5 * v_2 - temp);
    REAL f3_eq = w3 * rho * (1. - 3. * u + 4.5 * u_2 - temp);
    REAL f4_eq = w4 * rho * (1. - 3. * v + 4.5 * v_2 - temp);
    REAL f5_eq = w5 * rho * (1. + 3. * (u + v) + 4.5 * (u + v) * (u + v) - temp);
    REAL f6_eq = w6 * rho * (1. + 3. * (-u + v) + 4.5 * (-u + v) * (-u + v) - temp);
    REAL f7_eq = w7 * rho * (1. + 3. * (-u - v) + 4.5 * (-u - v) * (-u - v) - temp);
    REAL f8_eq = w8 * rho * (1. + 3. * (u - v) + 4.5 * (u - v) * (u - v) - temp);
    // update distrbution function
    gpu_f0[index] = (1. - omega) * gpu_f0[index] + omega * f0_eq ; 
    gpu_f1[index] = (1. - omega) * gpu_f1[index] + omega * f1_eq ; 
    gpu_f2[index] = (1. - omega) * gpu_f2[index] + omega * f2_eq ; 
    gpu_f3[index] = (1. - omega) * gpu_f3[index] + omega * f3_eq ;
    gpu_f4[index] = (1. - omega) * gpu_f4[index] + omega * f4_eq ;
    gpu_f5[index] = (1. - omega) * gpu_f5[index] + omega * f5_eq ;
    gpu_f6[index] = (1. - omega) * gpu_f6[index] + omega * f6_eq ;
    gpu_f7[index] = (1. - omega) * gpu_f7[index] + omega * f7_eq ;
    gpu_f8[index] = (1. - omega) * gpu_f8[index] + omega * f8_eq ;
  }
}
//====================

__global__ void UPDATE_BC_FRUV(REAL *gpu_f0,
			       REAL *gpu_f1, REAL *gpu_f2,REAL *gpu_f3, REAL *gpu_f4,
			       REAL *gpu_f5, REAL *gpu_f6,REAL *gpu_f7, REAL *gpu_f8,
			       REAL *gpu_rho, REAL *gpu_u, REAL *gpu_v, parameters *Pa){
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int k = threadIdx.z + blockIdx.z * blockDim.z;
  int nx = (*Pa).nx;
  int ny = (*Pa).ny;
  int nz = (*Pa).nz;
  int nx_min = 0;  int nx_max = nx-1;
  int ny_min = 0; int ny_max = ny-1;
  int nz_min = 0;  int nz_max = 0;
  //---
  int WEST_EDGE = (i == 0);
  int EAST_EDGE = (i == nx_max);
  int NORTH_EDGE = (j == ny_max);
  int SOUTH_EDGE = (j == 0);
  //---
  int NW_CORNER = ((i == 0) && (j == ny_max));
  int SW_CORNER = ((i == 0) && (j == 0));
  int NE_CORNER = ((i == nx_max) && (j == ny_max));
  int SE_CORNER = ((i == nx_max) && (j == 0));
  int ALL_DOMAIN = (i >= nx_min && i <= nx_max && j >= ny_min && j <= ny_max && k >= nz_min && k <= nz_max);
  int index;
  if (ALL_DOMAIN == 1){
    index = pos(i,j,k);
    REAL f0 = gpu_f0[index];
    REAL f1 = gpu_f1[index];
    REAL f2 = gpu_f2[index];
    REAL f3 = gpu_f3[index];
    REAL f4 = gpu_f4[index];
    REAL f5 = gpu_f5[index];
    REAL f6 = gpu_f6[index];
    REAL f7 = gpu_f7[index];
    REAL f8 = gpu_f8[index];
    REAL rho, u, v;
    REAL temp;
    if (NORTH_EDGE == 1){
	    u = 0.;
	    v = 0.;
	    rho = 1./(1. + v) * (f0 + f1 + f3 + 2.* (f2 + f6 + f5));
	    gpu_rho[index] = rho;// update rho
	    // update f4, f7, f8
	    gpu_f4[index] = f2 - 2./3. * rho * v;
	    gpu_f7[index] = f5 + 1./2. * (f1 - f3) - 1./6. * rho * v - 1./2 * rho * u;
	    gpu_f8[index] = f6 - 1./2. * (f1 - f3) - 1./6. * rho * v + 1./2 * rho * u;
	    // update u, v
	    gpu_u[index] = u;
	    gpu_v[index] = v;
	  }
    if (SOUTH_EDGE == 1){
	    u = 0.;
	    v = 0.;
	    rho = 1./(1. - v) * (f0 + f1 + f3 + 2.* (f4 + f7 + f8));
	    gpu_rho[index] = rho; //update rho
	    // f2, f5, f6 are unknown
	    gpu_f2[index] = f4 + 2./3. * rho * v;
	    gpu_f5[index] = f7 - 1./2. * (f1 - f3) + 1./6. * rho * v + 1./2 * rho * u;
	    gpu_f6[index] = f8 + 1./2. * (f1 - f3) + 1./6. * rho * v - 1./2 * rho * u;
	    // update u, v
	    gpu_u[index] = u;
	    gpu_v[index] = v;
	  }
    if (WEST_EDGE == 1) {
      // inlet, u_w and v_w are given
	    u = (*Pa).u0; // inlet velocity
	    v = (*Pa).v0; // inlet velocity
	    // f1,f5,f8 are unknown
	    rho = 1./(1. - u) * (f0 + f2 + f4 + 2.* (f3 + f6 + f7));
	    gpu_rho[index] = rho;// update rho
	    //f1, f5, f8
	    gpu_f1[index] = f3 + 2./3. * rho * u;      
	    gpu_f5[index] = f7 - 1./2. * (f2 - f4) + 1./6.* rho * u + 1./2. * rho * v;
	    gpu_f8[index] = f6 + 1./2. * (f2 - f4) + 1./6.* rho * u - 1./2. * rho * v;
	  
	  //===== update rho
	  /*
	  gpu_rho[index] = gpu_f0[index]
	    + gpu_f1[index] + gpu_f2[index] + gpu_f3[index] + gpu_f4[index]
	    + gpu_f5[index] + gpu_f6[index] + gpu_f7[index] + gpu_f8[index];
	  gpu_rho[index] = 1./(1 -u) * (gpu_f0[index] + gpu_f2[index] + gpu_f4[index]+ 2. * (gpu_f3[index] + gpu_f6[index] + gpu_f7[index]));
	  */
	  // update u, v
	    gpu_u[index] = u;
	    gpu_v[index] = v;
	  }
    if (EAST_EDGE == 1) {
    // outlet, OPEN BOUNDARY CONDITION FOR CHANNEL FLOW 
    //f3,f6,f7 are unknown
	  gpu_f3[index] = gpu_f3[pos(i-1,j,k)];
	  gpu_f6[index] = gpu_f6[pos(i-1,j,k)];
	  gpu_f7[index] = gpu_f7[pos(i-1,j,k)];
	  gpu_rho[index] = gpu_f0[index]
	    + gpu_f1[index] + gpu_f2[index] + gpu_f3[index] + gpu_f4[index]
	    + gpu_f5[index] + gpu_f6[index] + gpu_f7[index] + gpu_f8[index];//update rho
	  // update u and v
	  temp = gpu_f5[index] + gpu_f1[index] + gpu_f8[index] - gpu_f6[index] - gpu_f3[index] - gpu_f7[index];
	  gpu_u[index] = 1./gpu_rho[index] * temp;
	  temp = gpu_f5[index] + gpu_f2[index] + gpu_f6[index] - gpu_f8[index] - gpu_f4[index] - gpu_f7[index];
	  gpu_v[index] = 1./gpu_rho[index] * temp;
	  }
    if (NW_CORNER == 1){
	  u = 0.;
	  v = 0.;
	  //take the rho to be value at the closest node in the bulk
	  rho = 0.5*(gpu_rho[pos(i+1,j,k)] + gpu_rho[pos(i,j-1,k)]);
	  gpu_rho[index] = rho; // update rho
	  // f1,f4,f8,f5,f7 are unknown
	  gpu_f1[index] = f3;
	  gpu_f4[index] = f2;
	  gpu_f8[index] = f6;
	  temp = f0 + 2.* (f2 + f3 + f6);
	  gpu_f5[index] = 1./2. * (rho - temp);
	  gpu_f7[index] = gpu_f5[index];
	  // update u and v
	  gpu_u[index] = u;
	  gpu_v[index] = v;
	  }
    if (SW_CORNER == 1){
	  u = 0.;
	  v = 0.;
    //take the rho to be value at the closest node in the bulk
	  rho = 0.5*(gpu_rho[pos(i+1,j,k)] + gpu_rho[pos(i,j+1,k)]);
	  gpu_rho[index] = rho;// update rho
	  // f1,f2,f5,f6,f8 are unknown
	  gpu_f1[index] = f3;
	  gpu_f2[index] = f4;
	  gpu_f5[index] = f7;
	  temp = f0 + 2.* (f3 + f4 + f7);
	  gpu_f6[index] = 1./2. * (rho - temp);
	  gpu_f8[index] = gpu_f6[index];
	  // update u and v
	  gpu_u[index] = u;
	  gpu_v[index] = v;
	  }
    if (NE_CORNER == 1){
	  u = 0.;
	  v = 0.;
    //take the rho to be value at the closest node in the bulk
	  rho = gpu_rho[pos(i-1,j-1,k)];
	  gpu_rho[index] = rho;// update rho
	  // f3,f4,f7,f6,f8 are unknown
	  gpu_f3[index] = f1;
	  gpu_f4[index] = f2;
	  gpu_f7[index] = f5;
	  temp = f0 + 2.* (f1 + f2 + f5);
	  gpu_f6[index] = 1./2. * (rho - temp);
	  gpu_f8[index] = gpu_f6[index];
	  // update u and v
	  gpu_u[index] = u;
	  gpu_v[index] = v;
	  }
    if (SE_CORNER == 1){
	  u = 0.;
	  v = 0.;
    //take the rho to be value at the closest node in the bulk
	  rho = gpu_rho[pos(i-1,j+1,k)];
	  gpu_rho[index] = rho;// update rho
	  // f2,f3,f6,f5,f7 are unknown
	  gpu_f2[index] = f4;
	  gpu_f3[index] = f1;
	  gpu_f6[index] = f8;
	  temp = f0 + 2.* (f4 + f1 + f8);
	  gpu_f5[index] = 1./2. * (rho - temp);
	  gpu_f7[index] = gpu_f5[index];
	  // update u and v
	  gpu_u[index] = u;
	  gpu_v[index] = v;
	  }
  }
}
//====================

__global__ void UPDATE_OBSTACLE_MASK_CIRCLE(parameters *Pa, REAL *gpu_mask){

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int k = threadIdx.z + blockIdx.z * blockDim.z;
  int nx = (*Pa).nx;
  int ny = (*Pa).ny;
  int nz = (*Pa).nz;
  int nx_min = 0;  int nx_max = nx-1;
  int ny_min = 0; int ny_max = ny-1;
  int nz_min = 0;  int nz_max = 0;
  int ALL_DOMAIN = (i >= nx_min && i <= nx_max && j >= ny_min && j <= ny_max && k >= nz_min && k <= nz_max);
  if (ALL_DOMAIN == 1){
    int index = pos(i,j,k);
    REAL radius = 25.;
    int ic = 750;
    int jc = 500;
    REAL dis_2 = (i - ic) * (i - ic) + (j - jc) * (j - jc);
    if (dis_2 <= radius * radius){
	    gpu_mask[index] = 1.;
	  }
  }
}
//===========================

__device__ void BOUNCE_BACK_f(int nx, int ny, int nz,
			      int i, int j, int k,
			      int shift_i, int shift_j,
			      REAL *gpu_mask, REAL *f_out, REAL *f_in){
  int index = pos(i,j,k);
  int i_next = i + shift_i;
  int j_next = j + shift_j; 
  int index_next = pos(i_next, j_next,k);
  if (gpu_mask[index] == 0 && gpu_mask[index_next] == 1){
    f_out[index] = f_in[index];
  }
}
//====================

__global__ void UPDATE_OBSTACLE_f(parameters *Pa, REAL *gpu_mask,
				  REAL *gpu_f0, REAL *gpu_f1, REAL *gpu_f2,REAL *gpu_f3, REAL *gpu_f4,
				  REAL *gpu_f5, REAL *gpu_f6,REAL *gpu_f7, REAL *gpu_f8){
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int k = threadIdx.z + blockIdx.z * blockDim.z;
  int nx = (*Pa).nx;
  int ny = (*Pa).ny;
  int nz = (*Pa).nz;
  int nx_min = 0;  int nx_max = nx-1;
  int ny_min = 0; int ny_max = ny-1;
  int nz_min = 0;  int nz_max = 0;
  REAL dis, dis_next;
  int shift_i, shift_j;
  int index, index_next;
  int i_next, j_next;
  int ALL_DOMAIN = (i >= nx_min && i <= nx_max && j >= ny_min && j <= ny_max && k >= nz_min && k <= nz_max);
  
  if (ALL_DOMAIN == 1){
    BOUNCE_BACK_f(nx,ny,nz,i,j,k,1,0,gpu_mask,gpu_f3,gpu_f1);
    BOUNCE_BACK_f(nx,ny,nz,i,j,k,0,1,gpu_mask,gpu_f4,gpu_f2);
    BOUNCE_BACK_f(nx,ny,nz,i,j,k,-1,0,gpu_mask,gpu_f1,gpu_f3);
    BOUNCE_BACK_f(nx,ny,nz,i,j,k,0,-1,gpu_mask,gpu_f2,gpu_f4);
    BOUNCE_BACK_f(nx,ny,nz,i,j,k,1,1,gpu_mask,gpu_f7,gpu_f5);
    BOUNCE_BACK_f(nx,ny,nz,i,j,k,-1,1,gpu_mask,gpu_f8,gpu_f6);
    BOUNCE_BACK_f(nx,ny,nz,i,j,k,-1,-1,gpu_mask,gpu_f5,gpu_f7);
    BOUNCE_BACK_f(nx,ny,nz,i,j,k,1,-1,gpu_mask,gpu_f6,gpu_f8);
  }
}
//======================

void DUMP_LAMMPS_STYLE_DATA(parameters *Pa,
			    REAL time,
			    int frame_num,
			    REAL *h_rho, REAL *h_u, REAL *h_v,
			    REAL *h_f0)
{
  int nx = (*Pa).nx;
  int ny = (*Pa).ny;
  int nz = (*Pa).nz;
  int n_total = (nx)*ny*(nz);
  char OutFileName[200];
  sprintf(OutFileName,"Output/Data_Frame_%04d.dump", frame_num);
  FILE *OutFile = fopen(OutFileName,"w");
  fprintf(OutFile,"ITEM: TIMESTEP\n");
  fprintf(OutFile,"%d\t%5.2f\n",frame_num, time);
  fprintf(OutFile,"ITEM: NUMBER OF ATOMS\n");
  fprintf(OutFile,"%d\n",n_total);
  fprintf(OutFile,"ITEM: BOX BOUNDS\n");
  int nx_min = 0;  int nx_max = nx-1;
  int ny_min = 0;  int ny_max = ny-1;
  int nz_min = 0;  int nz_max = 0;

  fprintf(OutFile,"%d\t%d\n",nx_min,nx_max);
  fprintf(OutFile,"%d\t%d\n",ny_min,ny_max);
  fprintf(OutFile,"%d\t%d\n",nz_min,nz_max);
  //  fprintf(OutFile,"ITEM: ATOMS id type x y z f0 %8s\n", "f1");
  fprintf(OutFile,"ITEM: ATOMS id type x y z rho u v f0\n");
  int id,type;
  type = 1;
  int sum = 0;
  int nan_flag = 0;
  for ( int k = nz_min; k <= nz_max; k++){
    for (int j = ny_min; j <= ny_max; j++){
	    for (int i = nx_min; i <= nx_max; i++){
	      id = pos(i,j,k);
	      sum++;
	      REAL f0_val = h_f0[id];
	      if (isnan(f0_val)){
		      f0_val = Dump_Error_Val;
		      nan_flag += 1;
		    }
	      REAL rho_val = h_rho[id];
	      if (isnan(rho_val)){
		      rho_val = Dump_Error_Val;
		      nan_flag += 1;
		    }
	      REAL u_val = h_u[id];
	      if (isnan(u_val)){
		      u_val = Dump_Error_Val;
		      nan_flag += 1;
		    }
	      REAL v_val = h_v[id];
	      if (isnan(v_val)){
		      v_val = Dump_Error_Val;
		      nan_flag += 1;
		    }
	      fprintf(OutFile, "%d\t%d\t%d\t%d\t%d\t",id,type,i,j,k);
	      fprintf(OutFile, "%7.7e\t",rho_val);
	      fprintf(OutFile, "%7.7e\t",u_val);
	      fprintf(OutFile, "%7.7e\t",v_val);
	      fprintf(OutFile, "%7.7e\t",f0_val);
	      fprintf(OutFile, "\n");
	    }
	  }
  }
  printf("frame_id = %d\n", frame_num);
  if (nan_flag != 0){
    printf("number of nan, %d \n",nan_flag);
  }
  //  printf("frame_id = %d, sum = %.2e\n", frame_num, sum);
  fclose(OutFile);
  fflush(stdout);
}


