/*
Name: Md Kamal Hossain Chowdhury
Email:mhchowdhury@crimson.ua.edu    
Course: CS 691
Homework #:3
Instructions to compile the program: (for example: sbatch hw3.sh hw3.cu
  nvcc -o hw3 hw3.cpp)
Instructions to run the program: (for example: ./hw3 1024 10000)
*/


/* 
   Sequential version of Jacobi iterative method and Gauss-Seidel method
   to solve a system of linear equations. 

   To compile: 
   gcc -O -Wall iterative.c
   nvcc -o hw3 -DDEBUG0 hw3.cu
   To run:
   ./a.out 1024 10000
 
*/
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>
#include <limits.h>
#include <float.h>
#define BLOCK_SIZE 32
//#define TILE_WIDTH 16
#define Tile_Width 32
// const int TILE_DIM = 64; //blockDim.x = TILE_DIM must be
#define block_width 64


/* function to measure time taken */
double gettime(void) {
  struct timeval tval;

  gettimeofday(&tval, NULL);

  return( (double)tval.tv_sec + (double)tval.tv_usec/1000000.0 );
}

inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

void printmatrix(double **a, double *b, int N) {
#ifdef DEBUG_PRINT
    int i, j;
    if (N < 20) {
       for (i = 0; i < N; i++) {
           for (j = 0; j < N; j++)
               printf("%10.6f ",a[i][j]);
           printf("  %10.6f\n",b[i]);
       }
       printf("\n");
    }
#endif
}

void jacobi(double *a, double *b, double *x, double *xnew, int N, 
            double tolerance, int maxiters) {
    int i, j, k;
    double sum, maxError=DBL_MAX, *temp;

    for (k = 0; maxError > tolerance && k < maxiters; k++) {
        for (i = 0; i < N; i++) {
            sum = 0.0;
            for (j = 0; j < i; j++) 
                sum += a[i*N+j]*x[j];
            for (j = i+1; j < N; j++) 
                sum += a[i*N+j]*x[j];
            xnew[i] = (b[i] - sum)/a[i*N+i];
        }
        maxError = 0.0;
        for (i = 0; i < N; i++)
            maxError = fmax(maxError, fabs(x[i] - xnew[i]));
#ifdef DEBUG0
    printf("After %d iterations, maxError = %g\n", k, maxError);
#endif
        temp = xnew;
        xnew = x;
        x = temp;
    }
    if (k == maxiters) 
       printf("Failed to converge after %d iterations, maxError = %g\n",
               k, maxError);
#ifdef DEBUG_PRINT
    printf("Returning after %d iterations, maxError = %g\n", k, maxError);
#endif
   
}


__global__ void jacobi_shared(double *a, double *b, double *x, double *xnew, int N) {
   
    int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ double sharedX[Tile_Width];

    unsigned int tid = threadIdx.x;

    sharedX[tid]=x[col];
    __syncthreads();
    double sum;
   for(int k=0;k<N/Tile_Width;k++){

        if(row<N && col<N){
             sum = 0.0;
            // int idx = col;
            for (int j=0; j<N; j++)
            {
                if (col != j)
                    sum += a[col*N + j] * sharedX[tid];
            }
             
            
        }

   }
   __syncthreads();
   if(row<N && col<N){
        xnew[col] = (b[col] - sum) / a[col*N + col];
   }
  
       
}

__global__ void jacobi_modified(double *a, double *b, double *x, double *xnew, int N) {
   
    extern   __shared__ double shared_x[];

  // TILE_DIM is shared mmeory size - equal to block_width
  int TILE_DIM = blockDim.x;
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  
    
  double sum;
  if (i < N)
  {
    //printf ("x_prev=%f\n", x[i]);
    sum = 0;
    for (int j=0; j<((N+TILE_DIM-1)/TILE_DIM); j++) //divides sum compute into (N/TILE_DIM) chunks for shmem
    {
      shared_x[threadIdx.x] = x[j*TILE_DIM + threadIdx.x];

      __syncthreads();

      for (int k=0; k<TILE_DIM; k++)
      {  
          //printf("j*TILE_DIM=%d",j*TILE_DIM);
        if (i != j*TILE_DIM + k){
           
            if(i<j*TILE_DIM + k)
            {    
                sum += a[(j*TILE_DIM+k)*N + i]*shared_x[k];
            }
            __syncthreads();
            if(i>j*TILE_DIM + k){
                sum += a[(j*TILE_DIM+k)*N + i]*xnew[k];

            }
            // else
            // {   __syncthreads();
            //     sum += a[(j*TILE_DIM+k)*N + i]*shared_x[k];

            // }
            

        }
          
      }

      
    }
  //printf ("sigma_value=%f\n", sum);
   
  xnew[i] = (b[i]-sum)/a[i*N+ i];
  }
       
}

__global__ void jacobi_naive(double *a, double *b, double *x, double *xnew, int N) {
    int  j;
    double sum;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
	
    if(col<N){
    
            sum = 0.0;
    
            for (j = 0; j < N; j++) 
                {
                    if(j!=col)
                        sum += a[col*N + j]*x[j];
                }
            
            xnew[col] = (b[col] - sum)/a[col*N+col];
           
        }
       
}

int compare_array(double *A,double *B, int N) {
  int flag=1;
  for (int i = 0; i < N; i++)
    {
        double maxError = 0.001;
  
      if(maxError<fabs(A[i]-B[i])){
        printf("Failed naive A[%d]=%lf B[%d]=%lf\n",i,A[i],i,B[i]);
        flag= 0;
        return flag;
      }


    }
    
      
  
   

  return flag;
}


void gauss_seidel(double *a, double *b, double *x, double *xnew, int N, 
                  double tolerance, int maxiters) {
    int i, j, k;
    double sum, maxError=DBL_MAX, *temp;

    for (k = 0; maxError > tolerance && k < maxiters; k++) {
        for (i = 0; i < N; i++) {
            sum = 0.0;
            for (j = 0; j < i; j++) 
                sum += a[i*N+j]*xnew[j];
            for (j = i+1; j < N; j++) 
                sum += a[i*N+j]*x[j];
            xnew[i] = (b[i] - sum)/a[i*N+i];
        }
        maxError = 0.0;
        for (i = 0; i < N; i++)
            maxError = fmax(maxError, fabs(x[i] - xnew[i]));
#ifdef DEBUG0
    printf("After %d iterations, maxError = %g\n", k, maxError);
#endif
        temp = xnew;
        xnew = x;
        x = temp;
    }
    if (k == maxiters) 
       printf("Failed to converge after %d iterations, maxError = %g\n",
               k, maxError);
#ifdef DEBUG_PRINT
    printf("Returning after %d iterations, maxError = %g\n", k, maxError);
#endif
   
    free(xnew);
}

int main(int argc, char **argv) {
    double *a, *b,  *x, *xnew, tolerance;
    int i, j, k, N, maxiters;

    double *d_a,*d_b, *d_x, *d_xnew;
    double *d_sm_a,*d_sm_b,*d_sm_x,*d_sm_xnew;
    double *d_gauss_a,*d_gauss_b, *d_gauss_x, *d_gauss_xnew;
    double *h_x,*h_xnew;
    double maxError=DBL_MAX;
    double t1, t2;

    if (argc < 3) {
        printf("Usage: %s <size> <maxiters>\n", argv[0]);
        exit(1);
    }

    N = atoi(argv[1]);
    maxiters = atoi(argv[2]);
    tolerance = pow(10, -6);
    printf("N = %d maxiters = %d tolerance = %g\n", N, maxiters, tolerance);

    /* allocate space for the matrix */
    a = (double *)malloc(sizeof(double) * N * N);
    b = (double *)malloc(sizeof(double) * N);
    x = (double *)malloc(sizeof(double) * N);
    xnew = (double *)malloc(sizeof(double) * N);
    
    h_x = (double *)malloc(sizeof(double) * N);
    h_xnew = (double *)malloc(sizeof(double) * N);


    //allocate memory in the device 
    checkCuda(cudaMalloc(&d_a, N*N*sizeof(double)));
    checkCuda(cudaMalloc(&d_b, N*sizeof(double)));
    checkCuda(cudaMalloc(&d_x, N*sizeof(double)));
    checkCuda(cudaMalloc(&d_xnew, N*sizeof(double)));
    
    
    
    
    checkCuda(cudaMalloc(&d_sm_a, N*N*sizeof(double)));
    checkCuda(cudaMalloc(&d_sm_b, N*sizeof(double)));
    checkCuda(cudaMalloc(&d_sm_x, N*sizeof(double)));
    checkCuda(cudaMalloc(&d_sm_xnew, N*sizeof(double)));


    checkCuda(cudaMalloc(&d_gauss_a, N*N*sizeof(double)));
    checkCuda(cudaMalloc(&d_gauss_b, N*sizeof(double)));
    checkCuda(cudaMalloc(&d_gauss_x, N*sizeof(double)));
    checkCuda(cudaMalloc(&d_gauss_xnew, N*sizeof(double)));
    //checkCuda(cudaMalloc(&d_temp, N*N*sizeof(double)));
    /* initialize the matrix and vectors */
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            a[i*N+j] = 0.0;
        }
    }
    for (i = 0; i < N; i++) {
        a[i*N+i] = 3.0;
        b[i] = 5.0; // drand48();
        x[i] = 0.0;
        xnew[i] = 0.0;
    }
    for (i = 1; i < N; i++) {
        a[(i-1)*N+i] = a[i*N+(i-1)] = 1.0;
    }
    b[0] = b[N-1] = 4;

    /* print the matrix and right hand side */
    //printmatrix(a, b, N);

    checkCuda(cudaMemcpy(d_a, a, N*N*sizeof(double),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_b, b, N*sizeof(double),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_x, x, N*sizeof(double),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_xnew, xnew, N*sizeof(double),cudaMemcpyHostToDevice));
    
    
    
    checkCuda(cudaMemcpy(d_sm_a, a, N*N*sizeof(double),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_sm_b, b, N*sizeof(double),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_sm_x, x, N*sizeof(double),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_sm_xnew, xnew, N*sizeof(double),cudaMemcpyHostToDevice));

    checkCuda(cudaMemcpy(d_gauss_a, a, N*N*sizeof(double),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_gauss_b, b, N*sizeof(double),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_gauss_x, x, N*sizeof(double),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_gauss_xnew, xnew, N*sizeof(double),cudaMemcpyHostToDevice));


    //Naïve CPU Version 
    t1 = gettime();
    /* solve */
    jacobi(a, b, x, xnew, N, tolerance, maxiters);
    // gauss_seidel(a, b, x, xnew, N, tolerance, maxiters);
    t2 = gettime();
   printf("CPU Jacobi time taken for size = %d is = %f ms\n",
          N, (t2-t1)*1000);
    
     
    dim3 threadsPerBlock(((N+BLOCK_SIZE-1)/BLOCK_SIZE), ((N+BLOCK_SIZE-1)/BLOCK_SIZE), 1);
    dim3 blocksPerGrid(BLOCK_SIZE, BLOCK_SIZE, 1);

    cudaEvent_t startEvent, stopEvent;
    checkCuda( cudaEventCreate(&startEvent) );
    checkCuda( cudaEventCreate(&stopEvent) );
    float ms=0;


    checkCuda(cudaEventRecord(startEvent, 0));
    checkCuda(cudaEventRecord(stopEvent, 0));
    
    
     for (k = 0; maxError > tolerance && k < maxiters; k++) {
         if(k&1){
              jacobi_naive<<<threadsPerBlock,blocksPerGrid>>>(d_a, d_b, d_x,d_xnew, N);
            //    jacobi_shared <<<threadsPerBlock,blocksPerGrid>>>(d_a, d_b, d_x,d_xnew, N);
            }
        else
            {
                jacobi_naive<<<threadsPerBlock,blocksPerGrid>>>(d_a, d_b, d_xnew,d_x, N);
                // jacobi_shared <<<threadsPerBlock,blocksPerGrid>>>(d_a, d_b, d_xnew,d_x, N);
            }
        checkCuda(cudaMemcpy(h_x, d_x, N*sizeof(double), cudaMemcpyDeviceToHost));
        checkCuda(cudaMemcpy(h_xnew, d_xnew, N*sizeof(double), cudaMemcpyDeviceToHost));
        maxError = 0.0;
        for (i = 0; i < N; i++)
            maxError = fmax(maxError, fabs(h_x[i] - h_xnew[i]));
        #ifdef DEBUG0
            printf("Naive Jacobi After %d iterations, maxError = %g\n", k, maxError);
        #endif
                // temp = xnew;
                // xnew = x;
                // x = temp;
            }
            if (k == maxiters) 
            printf("Naive Jacobi Failed to converge after %d iterations, maxError = %g\n",
                    k, maxError);
        #ifdef DEBUG_PRINT
            printf("Naive Jacobi Returning after %d iterations, maxError = %g\n", k, maxError);
        #endif
    
    
     
    checkCuda( cudaEventSynchronize(stopEvent) );
    checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );

    

    printf("Simple GPU Version time Jacabi for size = %d time taken=%f ms\n",N,ms);
    

   
    checkCuda(cudaEventRecord(startEvent, 0));
    checkCuda(cudaEventRecord(stopEvent, 0));
    
    maxError=DBL_MAX;
     for (k = 0; maxError > tolerance && k < maxiters; k++) {
         if(k&1){
            // jacobi_naive<<<threadsPerBlock,blocksPerGrid>>>(d_a, d_b, d_x,d_xnew, N);
            //    jacobi_shared <<<threadsPerBlock,blocksPerGrid>>>(d_sm_a, d_sm_b, d_sm_x,d_sm_xnew, N);
               jacobi_shared <<<(N+block_width-1)/block_width,block_width,block_width*sizeof(double)>>>(d_sm_a, d_sm_b, d_sm_x,d_sm_xnew, N);
            //    jacobi_modified <<<(N+block_width-1)/block_width,block_width,block_width*sizeof(double)>>>(d_sm_a, d_sm_b, d_sm_x,d_sm_xnew, N);

               
            }
        else
            {
                // jacobi_naive<<<threadsPerBlock,blocksPerGrid>>>(d_a, d_b, d_xnew,d_x, N);
               // jacobi_shared <<<threadsPerBlock,blocksPerGrid>>>(d_sm_a, d_sm_b, d_sm_xnew,d_sm_x, N);
               jacobi_shared <<<(N+block_width-1)/block_width,block_width,block_width*sizeof(double)>>>(d_sm_a, d_sm_b, d_sm_xnew,d_sm_x, N);
            //    jacobi_modified <<<(N+block_width-1)/block_width,block_width,block_width*sizeof(double)>>>(d_sm_a, d_sm_b, d_sm_xnew,d_sm_x, N);

            }
        checkCuda(cudaMemcpy(h_x, d_sm_x, N*sizeof(double), cudaMemcpyDeviceToHost));
        checkCuda(cudaMemcpy(h_xnew, d_sm_xnew, N*sizeof(double), cudaMemcpyDeviceToHost));
        maxError = 0.0;
        for (i = 0; i < N; i++)
            maxError = fmax(maxError, fabs(h_x[i] - h_xnew[i]));
        
        #ifdef DEBUG0
            printf("Shared memory Optimized GPU Version After %d iterations, maxError = %g\n", k, maxError);
            
        #endif
                // temp = xnew;
                // xnew = x;
                // x = temp;
            }
            if (k == maxiters) 
            printf("Shared memory Optimized GPU Version Failed to converge after %d iterations, maxError = %g\n",
                    k, maxError);
        #ifdef DEBUG_PRINT
            printf("Shared memory Optimized GPU Version Returning after %d iterations, maxError = %g\n", k, maxError);
        #endif
    
    
    checkCuda( cudaEventSynchronize(stopEvent) );
    checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );

    

    printf("Shared memory Optimized GPU Version time Jacabi for size = %d time taken=%f ms\n",N,ms);
   
    checkCuda(cudaEventRecord(startEvent, 0));
    checkCuda(cudaEventRecord(stopEvent, 0));
    maxError=DBL_MAX;
     for (k = 0; maxError > tolerance && k < maxiters; k++) {
         if(k&1){
            // jacobi_naive<<<threadsPerBlock,blocksPerGrid>>>(d_a, d_b, d_x,d_xnew, N);
            //    jacobi_shared <<<threadsPerBlock,blocksPerGrid>>>(d_sm_a, d_sm_b, d_sm_x,d_sm_xnew, N);
            //    jacobi_shared <<<(N+block_width-1)/block_width,block_width,block_width*sizeof(double)>>>(d_sm_a, d_sm_b, d_sm_x,d_sm_xnew, N);
               jacobi_modified <<<(N+block_width-1)/block_width,block_width,block_width*sizeof(double)>>>(d_gauss_a, d_gauss_b, d_gauss_x,d_gauss_xnew, N);

               
            }
        else
            {
                // jacobi_naive<<<threadsPerBlock,blocksPerGrid>>>(d_a, d_b, d_xnew,d_x, N);
               // jacobi_shared <<<threadsPerBlock,blocksPerGrid>>>(d_sm_a, d_sm_b, d_sm_xnew,d_sm_x, N);
            //    jacobi_shared <<<(N+block_width-1)/block_width,block_width,block_width*sizeof(double)>>>(d_sm_a, d_sm_b, d_sm_xnew,d_sm_x, N);
               jacobi_modified <<<(N+block_width-1)/block_width,block_width,block_width*sizeof(double)>>>(d_gauss_a, d_gauss_b, d_gauss_xnew,d_gauss_x, N);

            }
        checkCuda(cudaMemcpy(h_x, d_gauss_x, N*sizeof(double), cudaMemcpyDeviceToHost));
        checkCuda(cudaMemcpy(h_xnew, d_gauss_xnew, N*sizeof(double), cudaMemcpyDeviceToHost));
        maxError = 0.0;
        for (i = 0; i < N; i++)
            maxError = fmax(maxError, fabs(h_x[i] - h_xnew[i]));
        
        #ifdef DEBUG0
            printf("Modified GPU  Guass_Seidel After %d iterations, maxError = %g\n", k, maxError);
            
        #endif
                // temp = xnew;
                // xnew = x;
                // x = temp;
            }
            if (k == maxiters) 
            printf("Modified GPU Guass_Seidel Failed to converge after %d iterations, maxError = %g\n",
                    k, maxError);
        #ifdef DEBUG_PRINT
            printf("Modified GPU Guass_Seidel Returning after %d iterations, maxError = %g\n", k, maxError);
        #endif
    
   
    checkCuda( cudaEventSynchronize(stopEvent) );
    checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );

    

    printf("Modified  GPU Guass_Seidel Version time Jacabi for size = %d time taken=%f ms\n",N,ms);

    //compare_array(x,h_x,N);

    /* print solution */
    if (N < 20) {
       for (i=0; i<N; i++)
           printf("%f ", x[i]);
       printf("\n");
    }
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_x);
    cudaFree(d_xnew);

    // free(a);
    // free(b);
    // free(temp);
    // free(x);
    // free(xnew);
    


    return 0;
}
