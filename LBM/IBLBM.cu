#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#include <stdio.h>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <math.h>
#define REAL double //double //float
#define BLOCK_X		16//8
#define BLOCK_Y		16//8
#define BLOCK_Z		1

#include "function.h"
//=====
#define PI 3.14159265358979323846  /* pi */

int main(int nargs, char** args )
{
  parameters h_Pa[1]; //host parameters
  SET_DEFAULT_PARAMETERS(h_Pa);
  command_line_parser(nargs, args, h_Pa);
  COMPUTE_PARAMETERS(h_Pa);
  DUMP_PARAMETERS(h_Pa);
  int nx = (*h_Pa).nx;
  int ny = (*h_Pa).ny;
  int nz = (*h_Pa).nz;
  int gpu_id = (*h_Pa).gpu_id;
  //gpuErrchk(cudaSetDevice(gpu_id));
  //=====
  parameters *gpu_Pa;	//gpu parameters
  cudaMalloc((void**)&gpu_Pa,sizeof(parameters));
  cudaMemcpy(gpu_Pa, h_Pa, sizeof(parameters), cudaMemcpyHostToDevice);
  
  REAL dt = 1.0;
  REAL time = 0.;
  int frame_num = 0;
  int iter_max = (*h_Pa).iter_max;
  int dump_frequency = (*h_Pa).dump_frequency;
  //----
  int thread = 16;
  //----
  dim3 Threads_LBM(BLOCK_X, BLOCK_Y, BLOCK_Z);
  dim3 Blocks_LBM((int)ceil((nx)/(REAL)BLOCK_X),(int)ceil((ny)/(REAL)BLOCK_Y),(int)ceil((nz)/(REAL)BLOCK_Z));
  printf("Block_size_x,y,z %d, %d, %d\n",BLOCK_X,BLOCK_Y,BLOCK_Z);

  //---------- MEMORY ALLOCATIONS ----------
  size_t ArraySize = (nx)*(ny)*(nz)*sizeof(REAL);
  REAL *gpu_f0;
  REAL *gpu_f1, *gpu_f2, *gpu_f3, *gpu_f4;
  REAL *gpu_f5, *gpu_f6, *gpu_f7, *gpu_f8;
  REAL *gpu_tf0;
  REAL *gpu_tf1, *gpu_tf2, *gpu_tf3, *gpu_tf4;
  REAL *gpu_tf5, *gpu_tf6, *gpu_tf7, *gpu_tf8;
  //----
  cudaMalloc((void**)&gpu_f0, ArraySize);
  cudaMalloc((void**)&gpu_f1, ArraySize);
  cudaMalloc((void**)&gpu_f2, ArraySize);
  cudaMalloc((void**)&gpu_f3, ArraySize);
  cudaMalloc((void**)&gpu_f4, ArraySize);
  cudaMalloc((void**)&gpu_f5, ArraySize);
  cudaMalloc((void**)&gpu_f6, ArraySize);
  cudaMalloc((void**)&gpu_f7, ArraySize);
  cudaMalloc((void**)&gpu_f8, ArraySize);
  cudaMalloc((void**)&gpu_tf0, ArraySize);
  cudaMalloc((void**)&gpu_tf1, ArraySize);
  cudaMalloc((void**)&gpu_tf2, ArraySize);
  cudaMalloc((void**)&gpu_tf3, ArraySize);
  cudaMalloc((void**)&gpu_tf4, ArraySize);
  cudaMalloc((void**)&gpu_tf5, ArraySize);
  cudaMalloc((void**)&gpu_tf6, ArraySize);
  cudaMalloc((void**)&gpu_tf7, ArraySize);
  cudaMalloc((void**)&gpu_tf8, ArraySize);
  REAL *gpu_mask;
  cudaMalloc((void**)&gpu_mask, ArraySize);
  REAL *gpu_rho, *gpu_u, *gpu_v;
  cudaMalloc((void**)&gpu_rho, ArraySize);
  cudaMalloc((void**)&gpu_u, ArraySize);
  cudaMalloc((void**)&gpu_v, ArraySize);

  //=====CPU===
  REAL *h_f0 = (REAL*) malloc(ArraySize); // host, cpu
  REAL *h_f1 = (REAL*) malloc(ArraySize); // host, cpu
  REAL *h_f2 = (REAL*) malloc(ArraySize); // host, cpu
  REAL *h_f3 = (REAL*) malloc(ArraySize); // host, cpu
  REAL *h_f4 = (REAL*) malloc(ArraySize); // host, cpu
  REAL *h_f5 = (REAL*) malloc(ArraySize); // host, cpu
  REAL *h_f6 = (REAL*) malloc(ArraySize); // host, cpu
  REAL *h_f7 = (REAL*) malloc(ArraySize); // host, cpu
  REAL *h_f8 = (REAL*) malloc(ArraySize); // host, cpu

  REAL *h_rho = (REAL*) malloc(ArraySize); // host, cpu
  REAL *h_u = (REAL*) malloc(ArraySize); // host, cpu
  REAL *h_v = (REAL*) malloc(ArraySize); // host, cpu
  
  //===INITIALIZATION
  SetArrayValues<<<Blocks_LBM,Threads_LBM>>>(gpu_Pa,gpu_mask,0); //1:solid, 0:liquid.
  //
  SetArrayValues<<<Blocks_LBM,Threads_LBM>>>(gpu_Pa,gpu_f0, (*h_Pa).w0);
  SetArrayValues<<<Blocks_LBM,Threads_LBM>>>(gpu_Pa,gpu_f1, (*h_Pa).w1);
  SetArrayValues<<<Blocks_LBM,Threads_LBM>>>(gpu_Pa,gpu_f2, (*h_Pa).w2);
  SetArrayValues<<<Blocks_LBM,Threads_LBM>>>(gpu_Pa,gpu_f3, (*h_Pa).w3);
  SetArrayValues<<<Blocks_LBM,Threads_LBM>>>(gpu_Pa,gpu_f4, (*h_Pa).w4);
  SetArrayValues<<<Blocks_LBM,Threads_LBM>>>(gpu_Pa,gpu_f5, (*h_Pa).w5);
  SetArrayValues<<<Blocks_LBM,Threads_LBM>>>(gpu_Pa,gpu_f6, (*h_Pa).w6);
  SetArrayValues<<<Blocks_LBM,Threads_LBM>>>(gpu_Pa,gpu_f7, (*h_Pa).w7);
  SetArrayValues<<<Blocks_LBM,Threads_LBM>>>(gpu_Pa,gpu_f8, (*h_Pa).w8);
  //---
  SetArrayValues<<<Blocks_LBM,Threads_LBM>>>(gpu_Pa,gpu_rho, 1.);
  REAL u0 = (*h_Pa).u0; // inlet velocity in x
  REAL v0 = (*h_Pa).v0;// inlet velocity in y
  SetArrayValues<<<Blocks_LBM,Threads_LBM>>>(gpu_Pa,gpu_u, u0);
  SetArrayValues<<<Blocks_LBM,Threads_LBM>>>(gpu_Pa,gpu_v, v0);
  //===== 
  gpuErrchk(cudaMemcpy(h_rho, gpu_rho, ArraySize, cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(h_u, gpu_u,ArraySize, cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(h_v, gpu_v,ArraySize, cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(h_f0, gpu_f0,ArraySize, cudaMemcpyDeviceToHost));
  //----

  DUMP_LAMMPS_STYLE_DATA(h_Pa,time,frame_num,
			 h_rho, h_u, h_v, h_f0);// time = 0, frame_num  = 0 

  frame_num = 1;
  //===== main loop =====
  for (int iter = 1; iter <= iter_max; iter++){
    
    Update_Collision<<<Blocks_LBM,Threads_LBM>>>(gpu_f0,
						   gpu_f1,gpu_f2,gpu_f3,gpu_f4,
						   gpu_f5,gpu_f6,gpu_f7,gpu_f8,
						   gpu_rho, gpu_u, gpu_v, gpu_Pa); //collision step

    cudaMemcpy(gpu_tf0, gpu_f0, ArraySize, cudaMemcpyDeviceToDevice);
    cudaMemcpy(gpu_tf1, gpu_f1, ArraySize, cudaMemcpyDeviceToDevice);
    cudaMemcpy(gpu_tf2, gpu_f2, ArraySize, cudaMemcpyDeviceToDevice);
    cudaMemcpy(gpu_tf3, gpu_f3, ArraySize, cudaMemcpyDeviceToDevice);
    cudaMemcpy(gpu_tf4, gpu_f4, ArraySize, cudaMemcpyDeviceToDevice);
    cudaMemcpy(gpu_tf5, gpu_f5, ArraySize, cudaMemcpyDeviceToDevice);
    cudaMemcpy(gpu_tf6, gpu_f6, ArraySize, cudaMemcpyDeviceToDevice);
    cudaMemcpy(gpu_tf7, gpu_f7, ArraySize, cudaMemcpyDeviceToDevice);
    cudaMemcpy(gpu_tf8, gpu_f8, ArraySize, cudaMemcpyDeviceToDevice);

    //===UPDATE STREAMING===
    //int shift_i, shift_j, shift_k;
    //---f0 streaming
    Update_Streaming<<<Blocks_LBM,Threads_LBM>>>(gpu_Pa,gpu_f0,
						   gpu_f1,gpu_f2,gpu_f3,gpu_f4,
						   gpu_f5,gpu_f6,gpu_f7,gpu_f8,gpu_tf0,
               gpu_tf1,gpu_tf2,gpu_tf3,gpu_tf4,
						   gpu_tf5,gpu_tf6,gpu_tf7,gpu_tf8);//streaming step

    // UPDATE f on OBSTACLE
    UPDATE_OBSTACLE_MASK_CIRCLE<<<Blocks_LBM,Threads_LBM>>>(gpu_Pa,gpu_mask);

    UPDATE_OBSTACLE_f<<<Blocks_LBM,Threads_LBM>>>(gpu_Pa, gpu_mask, gpu_f0,
                gpu_f1,gpu_f2,gpu_f3,gpu_f4, gpu_f5,gpu_f6,gpu_f7,gpu_f8);

    //===UPDATE BC ===
    UPDATE_BC_FRUV<<<Blocks_LBM,Threads_LBM>>>(gpu_f0,
						      gpu_f1,gpu_f2,gpu_f3,gpu_f4,
						      gpu_f5,gpu_f6,gpu_f7,gpu_f8,
						      gpu_rho,gpu_u, gpu_v,gpu_Pa); //boundary condition
    //===COMPUTE RHO U AND V 
    COMPUTE_RUV<<<Blocks_LBM,Threads_LBM>>>(gpu_Pa, gpu_mask, gpu_f0, 
          gpu_f1, gpu_f2, gpu_f3, gpu_f4,
			    gpu_f5, gpu_f6, gpu_f7, gpu_f8,
			    gpu_rho, gpu_u, gpu_v); //update fluid density and velocity

    //===OUTPUT DATA===

    if  ((iter % dump_frequency) == 0){
	    time = dt * iter;
	    gpuErrchk(cudaMemcpy(h_rho, gpu_rho,ArraySize, cudaMemcpyDeviceToHost));
	    gpuErrchk(cudaMemcpy(h_u, gpu_u,ArraySize, cudaMemcpyDeviceToHost));
	    gpuErrchk(cudaMemcpy(h_v, gpu_v,ArraySize, cudaMemcpyDeviceToHost));
	    gpuErrchk(cudaMemcpy(h_f0, gpu_f0,ArraySize, cudaMemcpyDeviceToHost));
      //----
      DUMP_LAMMPS_STYLE_DATA(h_Pa,time,frame_num, h_rho, h_u, h_v, h_f0);
	    frame_num++;
	  }
  }

  // Free device memory
  cudaFree(gpu_f0);
  cudaFree(gpu_f1);
  cudaFree(gpu_f2);
  cudaFree(gpu_f3);
  cudaFree(gpu_f4);
  cudaFree(gpu_f5);
  cudaFree(gpu_f6);
  cudaFree(gpu_f7);
  cudaFree(gpu_f8);
  cudaFree(gpu_tf0);
  cudaFree(gpu_tf1);
  cudaFree(gpu_tf2);
  cudaFree(gpu_tf3);
  cudaFree(gpu_tf4);
  cudaFree(gpu_tf5);
  cudaFree(gpu_tf6);
  cudaFree(gpu_tf7);
  cudaFree(gpu_tf8);
  //---
  cudaFree(gpu_mask);
  //---
  cudaFree(gpu_rho);
  cudaFree(gpu_u);
  cudaFree(gpu_v);
}