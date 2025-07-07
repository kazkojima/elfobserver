#include <algorithm>
#include <array>
#include <vector>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <getopt.h>

#define USE_CUDA
#ifdef USE_CUDA
#include "cuda_runtime.h"
#endif

#include <chrono>
#include <unistd.h>

using namespace std;

struct report
{
  float mean_value;
  float max_value;
  int32_t maxarg_x;
  int32_t maxarg_y;
};

const int SIGPORT = 5992;

// signal packet data (48*4*4) bytes
//const int DNUM = 48*4;
//const int PSIZE = DNUM * sizeof(float);

// quadratic phase detector

const int FS = 48000;
const int SHIFT = FS/100;
const int N = 50;
const int M = 200;
const int TS = 2;
const float MU = 50.0/FS;

const double alpha0 = 2300.0;
const double alpha_step = 1.0;
const double t0 = 0.7;
const double t_step = 0.5/(FS/2-1);
const double PI = 3.141592653589793;

//const int QNUM = FS/2/DNUM; // 125

float sigbuf[FS];
float vc[FS/2][M];
float vs[FS/2][M];
float array_I[M][N+1];
float array_Q[M][N+1];

float meanval[M];
float peakval[M];
int peakarg[M];

#ifdef USE_CUDA
__global__ void
conv_partial(float result[M][N+1], float *sig, float coeff[FS/2][M],
		     int width, int tm, int ofs, float mu)
{
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  int tidy = blockIdx.y * blockDim.y + threadIdx.y;
  __shared__ float tsum[M][TS];
  float total = 0.0;
  float prev = 0.0;
  int start, end;
  if (tidx >= width)
    return;
  start = FS/2 - tidy*((FS/2)/TS) - 1;
  end = start - ((FS/2)/TS) + 1;
  //float c0 = 1.0 + (float)(alpha_step/alpha0)*tidx;
  for (int j = start; j >= end; j--)
    {
      prev = mu*sig[ofs+j]*coeff[j][tidx] + (1-mu)*prev;
      //float c = c0*coeff[0][j];
      //prev = mu*sig[ofs+j]*cos(c) + (1-mu)*prev;
      //prev = sig[ofs+j]*cos(c);
      total = total + prev;
    }
  tsum[tidx][tidy] = total;
  __syncthreads();
  total = 0.0;
  for (int i = 0; i < TS; i++)
    total += tsum[tidx][i];
  result[tidx][tm] = total;
}
#else
void conv_partial_cpu(float result[M][N+1], float *sig, float coeff[FS/2][M],
		     int aidx, int tidx, int ofs, float mu)
{
  float total = 0.0;
  float prev = 0.0;

  for (int j = FS/2-1; j >= 0; j--)
    {
      prev = mu*sig[ofs+j]*coeff[j][aidx] + (1-mu)*prev;
      total = total + prev;
    }
  result[aidx][tidx] = total;
  if (0 && aidx == 0 && tidx == 0)
    {
      cout << "ofs " << ofs << " mu " << mu << endl;
      cout << "total " << total << endl;
    }
}
#endif

void qpd(float *sig, float result_I[M][N+1], float result_Q[M][N+1],
	     float c_I[FS/2][M], float c_Q[FS/2][M], int sft, float mu)
{
  for (int j = 0; j < N+1; j++)
    {
      int ofs = j * sft;
#ifdef USE_CUDA
      dim3 dimGrid(1,1);
      dim3 dimBlock(M,TS);
      conv_partial<<<dimGrid,dimBlock>>>(result_I, sig, c_I, M, j, ofs, mu);
      conv_partial<<<dimGrid,dimBlock>>>(result_Q, sig, c_Q, M, j, ofs, mu);
#else      
      for (int i = 0; i < M; i++)
	{
	  conv_partial_cpu(result_I, sig, c_I, i, j, ofs, mu);
	  conv_partial_cpu(result_Q, sig, c_Q, i, j, ofs, mu);
	}
#endif
    }
}

void peak(float result_I[M][N+1], float result_Q[M][N+1],
	  float mean[M], float val[M], int arg[M])
{
  for (int i = 0; i < M; i++)
    {
      float peakval = 0.0;
      float meanval = 0.0;
      int peakarg = -1;
      for (int j = 0; j < N; j++)
	{
	  float a = result_I[i][j]*result_I[i][j] + result_Q[i][j]*result_Q[i][j];
	  if (a > peakval)
	    {
	      peakval = a;
	      peakarg = j;
	    }
	  meanval += a;
	}
      mean[i] = meanval/N;
      val[i] = peakval;
      arg[i] = peakarg;
    }
}

int readn(int fd, char *buf, int nbytes)
{
  int nleft, nread;
  nleft = nbytes;
  while (nleft > 0)
    {
      nread = read(fd, buf, nleft);
      if (nread < 0)
	return nread;
      else if (nread == 0)
	break;
      nleft -= nread;
      buf += nread;
    }
  return (nbytes - nleft);
}

int main(int argc, char **argv)
{
  option longopts[] = {
    {"verbose", no_argument, NULL, 'v'}, {0}};
  int c;
  bool verbose = false;

  while (1)
    {
      int option_index = 0;
      c = getopt_long(argc, argv, "v", longopts, &option_index);
      if (c == -1)
        break;
      switch (c)
        {
        case 'v':
	  verbose = true;
          break;
        default:
          break;
        }
    }

// Compute test function tables
  double alpha = alpha0;
  for (int i = 0; i < M; i++)
    {
      double t = t0;
      for (int j = 0; j < FS/2; j++)
	{
#if 0
	  vc[i][j] = (float) (2*PI*(alpha/t));
	  vs[i][j] = (float) (PI/2 - 2*PI*(alpha/t));
#else
	  vc[j][i] = (float) cos(2*PI*(alpha/t));
	  vs[j][i] = (float) sin(2*PI*(alpha/t));
#endif
	  t += t_step;
	}
      alpha += alpha_step;
    }

#ifdef USE_CUDA
  float *d_sig;
  float (*d_vc)[M], (*d_vs)[M];
  float (*d_I)[N+1], (*d_Q)[N+1];
  cudaMalloc((void**) &d_sig, sizeof(sigbuf));
  cudaMalloc((void**) &d_vc, sizeof(vc));
  cudaMalloc((void**) &d_vs, sizeof(vs));
  cudaMalloc((void**) &d_I, sizeof(array_I));
  cudaMalloc((void**) &d_Q, sizeof(array_Q));

  cudaMemcpy(d_vc, (void*)vc, sizeof(vc), cudaMemcpyHostToDevice);
  cudaMemcpy(d_vs, (void*)vs, sizeof(vs), cudaMemcpyHostToDevice);
#endif

  int err = 0;
  int sigsock = -1;

  // tcp server
  int servsock = socket(AF_INET, SOCK_STREAM, 0);
  if (servsock < 0)
    {
      err = 1;
      std::cout << "can't open socket" << std::endl;
      goto end;
    }

  struct sockaddr_in sa, ca;
  memset(&sa, 0, sizeof(sa));
  sa.sin_family = AF_INET;
  sa.sin_addr.s_addr = INADDR_ANY;
  sa.sin_port = htons(SIGPORT);
  if (bind(servsock, (struct sockaddr *)&sa, sizeof(sa)) < 0)
    {
      err = 1;
      std::cout << "can't bind socket" << std::endl;
      goto end;
    }

  listen(servsock, 5);

  while (1)
    {
      socklen_t clilen = sizeof(ca);
      sigsock = accept(servsock, (struct sockaddr *)&ca, &clilen);
      if (sigsock < 0)
	{
	  err = 1;
	  std::cout << "fatal: can't accept" << std::endl;
	  goto end;
	}

      // read signals for 1sec from client
      int n = readn(sigsock, (char *)sigbuf, FS*sizeof(float));
      if (n < 0)
	{
	  err = 1;
	  std::cout << "fatal: read failed" << std::endl;
	  goto end;
	}
      else if (n != FS*sizeof(float))
	{
	  close(sigsock);
	  std::cout << "data abort" << std::endl;
	  // wait the next client
	  continue;
	}

      while (1)
	{
	  // QPD
	  //auto start = chrono::system_clock::now();
#ifdef USE_CUDA
	  cudaMemcpy(d_sig, sigbuf, sizeof(sigbuf), cudaMemcpyHostToDevice);
	  qpd(d_sig, d_I, d_Q, d_vc, d_vs, SHIFT, MU);
	  cudaMemcpy((void*)array_I, d_I, sizeof(array_I), cudaMemcpyDeviceToHost);
	  cudaMemcpy((void*)array_Q, d_Q, sizeof(array_Q), cudaMemcpyDeviceToHost);
#else
	  qpd(sigbuf, array_I, array_Q, vc, vs, SHIFT, MU);
#endif
	  //auto end = chrono::system_clock::now();
	  //const double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
	  //std::cout << "time " << elapsed << " ms" << std::endl;

	  peak(array_I, array_Q, meanval, peakval, peakarg);
	  int maxargx = 0;
	  int maxargy = 0;
	  float maxval = 0.0;
	  float mean = 0.0;
	  for (int i = 0; i < M; i++)
	    {
	      mean += meanval[i];
	      if (peakval[i] > maxval)
		{
		  maxval = peakval[i];
		  maxargx = peakarg[i];
		  maxargy = i;
		}
	    }
	  mean /= M;
	  if (verbose)
	    {
	      std::cout << "mean " << mean << std::endl;
	      std::cout << "max " << maxval << std::endl;
	      std::cout << "maxval " << maxargx << ", " << maxargy << std::endl;
	    }
	  struct report repo;
	  repo.mean_value = mean;
	  repo.max_value = maxval;
	  repo.maxarg_x = maxargx;
	  repo.maxarg_y = maxargy;
	  n = write(sigsock, (char *)&repo, sizeof(repo));
	  if (n < 0)
	    {
	      err = 1;
	      std::cout << "fatal: write failed" << std::endl;
	      goto end;
	    }
	  else if (n != sizeof(repo))
	    {
	      close(sigsock);
	      if (verbose)
		std::cout << "maybe disconnected" << std::endl;
	      // ?dump signal buffer?
	      // wait the next client
	      break;
	    }
#if 0
	  if (maxval > 100)
	    {
	      close(sigsock);
	      std::cout << "maybe invalid max value. Dump signal.dat" << std::endl;
	      std::ofstream dumpFile;
	      dumpFile.open("signal.dat", std::ios::out | std::ios::binary);
	      if (dumpFile.is_open())
		{
		  dumpFile.write(reinterpret_cast<const char*>(sigbuf), sizeof(sigbuf));
		  dumpFile.close();
		}
	      // wait the next client
	      break;
	    }
#endif

	  memmove((char *)sigbuf, (char *)&sigbuf[FS/2], FS/2*sizeof(float));
	  // read signals for 0.5sec
	  n = readn(sigsock, (char *)&sigbuf[FS/2], FS/2*sizeof(float));
	  if (verbose)
	    std::cout << "read " << n << " bytes" << std::endl;
	  if (n < 0)
	    {
	      err = 1;
	      std::cout << "fatal: read failed" << std::endl;
	      goto end;
	    }
	  else if (n != FS/2*sizeof(float))
	    {
	      close(sigsock);
	      if (verbose)
		std::cout << "maybe disconnected" << std::endl;
	      // wait the next client
	      break;
	    }
	}
    }

 end:
  if (sigsock >= 0)
    close(sigsock);
  if (servsock >= 0)
    close(servsock);

#ifdef USE_CUDA
  cudaFree(d_sig);
  cudaFree(d_vc);
  cudaFree(d_vs);
  cudaFree(d_I);
  cudaFree(d_Q);
#endif

  if (err)
    {
      std::cout << "QPD server failed" << std::endl;
      return EXIT_FAILURE;
    }
}
