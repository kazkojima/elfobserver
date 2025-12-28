#include <algorithm>
#include <array>
#include <vector>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <cmath>

#define USE_CUDA
#ifdef USE_CUDA
#include "cuda_runtime.h"
#endif

#include <chrono>
#include <unistd.h>

#include <H5Cpp.h>

using namespace std;

const int FS = 48000;

// WAV
typedef struct __attribute__((packed)) WAV_HEADER_COM {
  /* RIFF Chunk Descriptor */
  uint8_t RIFF[4] = {'R', 'I', 'F', 'F'}; // RIFF Header Magic header
  uint32_t ChunkSize;                    // RIFF Chunk Size
  uint8_t WAVE[4] = {'W', 'A', 'V', 'E'}; // WAVE Header
} wav_hdr_com;

typedef struct __attribute__((packed)) SUB_CHUNK_HEADER {
  /* SubChunk Descriptor */
  uint8_t SubChunkID[4];  // SubChunk id
  uint32_t SubChunkSize; // SubChunk Size
} subchunk_hdr;

typedef struct __attribute__((packed)) WAV_HEADER {
  /* RIFF Chunk Descriptor */
  uint8_t RIFF[4] = {'R', 'I', 'F', 'F'}; // RIFF Header Magic header
  uint32_t ChunkSize;                    // RIFF Chunk Size
  uint8_t WAVE[4] = {'W', 'A', 'V', 'E'}; // WAVE Header
  /* "fmt" sub-chunk */
  uint8_t fmt[4] = {'f', 'm', 't', ' '}; // FMT header
  uint32_t Subchunk1Size = 18; // Size of the fmt chunk
  uint16_t AudioFormat = 3;  // 3=LE float
  uint16_t NumOfChan = 1;   // 1=Mono 2=Sterio
  uint32_t SamplesPerSec = FS;  // Sampling Frequency in Hz
  uint32_t bytesPerSec = FS * 4; // bytes per second
  uint16_t blockAlign = 4;          // 2=16-bit mono, 4=16-bit stereo
  uint16_t bitsPerSample = 32;      // Number of bits per sample
  uint16_t cbSize = 0;
  /* "data" sub-chunk */
  uint8_t Subchunk2ID[4] = {'d', 'a', 't', 'a'}; // "data"  string
  uint32_t Subchunk2Size;                        // Sampled data length
} wav_hdr;

typedef struct __attribute__((packed)) WAV_HEADER_EX {
  /* RIFF Chunk Descriptor */
  uint8_t RIFF[4] = {'R', 'I', 'F', 'F'}; // RIFF Header Magic header
  uint32_t ChunkSize;                    // RIFF Chunk Size
  uint8_t WAVE[4] = {'W', 'A', 'V', 'E'}; // WAVE Header
  /* "fmt" sub-chunk */
  uint8_t fmt[4] = {'f', 'm', 't', ' '}; // FMT header
  uint32_t Subchunk1Size = 18; // Size of the fmt chunk
  uint16_t AudioFormat = 3;  // 3=LE float
  uint16_t NumOfChan = 1;   // 1=Mono 2=Sterio
  uint32_t SamplesPerSec = FS;  // Sampling Frequency in Hz
  uint32_t bytesPerSec = FS * 4; // bytes per second
  uint16_t blockAlign = 4;          // 2=16-bit mono, 4=16-bit stereo
  uint16_t bitsPerSample = 32;      // Number of bits per sample
  uint16_t cbSize = 0;
  /* "fact" sub-chunk */
  uint8_t SubchunkEID[4] = {'f', 'a', 'c', 't'};
  uint32_t SubchunkESize = 4;
  uint8_t SubchunkEData[4] = { 0x80,0x32,0x02,0x00};
  /* "data" sub-chunk */
  uint8_t Subchunk2ID[4] = {'d', 'a', 't', 'a'}; // "data"  string
  uint32_t Subchunk2Size;                        // Sampled data length
} wav_hdr_ex;

typedef struct __attribute__((packed)) WAV_FMT_SUBCHUNK {
  uint16_t AudioFormat = 3;  // 3=LE float
  uint16_t NumOfChan = 1;   // 1=Mono 2=Sterio
  uint32_t SamplesPerSec = FS;  // Sampling Frequency in Hz
  uint32_t bytesPerSec = FS * 4; // bytes per second
  uint16_t blockAlign = 4;          // 2=16-bit mono, 4=16-bit stereo
  uint16_t bitsPerSample = 32;      // Number of bits per sample
  uint16_t cbSize = 0;
} wav_fmt;

// quadratic phase detector

const int SHIFT = FS/100;
const int N = 50;
const int M = 200;
const int L = 1;
const int TS = 2;
const float MU = 50.0/FS;

const double alpha0 = 2300.0;
const double alpha_step = 1.0;
const double t0 = 0.7;
const double t_step = (L*0.5)/(L*FS/2-1);
const double PI = 3.141592653589793;

float sigbuf[L*FS];
float vc[L*FS/2][M];
float vs[L*FS/2][M];
float array_I[M][L*N+1];
float array_Q[M][L*N+1];

float meanval[M];
float peakval[M];
int peakarg[M];

#ifdef USE_CUDA
__global__ void
conv_partial(float result[M][L*N+1], float *sig, float coeff[L*FS/2][M],
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
  start = L*FS/2 - tidy*((L*FS/2)/TS) - 1;
  end = start - ((L*FS/2)/TS) + 1;
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
void conv_partial_cpu(float result[M][L*N+1], float *sig, float coeff[L*FS/2][M],
		     int aidx, int tidx, int ofs, float mu)
{
  float total = 0.0;
  float prev = 0.0;

  for (int j = L*FS/2-1; j >= 0; j--)
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

void qpd(float *sig, float result_I[M][L*N+1], float result_Q[M][L*N+1],
	     float c_I[L*FS/2][M], float c_Q[L*FS/2][M], int sft, float mu)
{
  for (int j = 0; j < L*N+1; j++)
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

void peak(float result_I[M][L*N+1], float result_Q[M][L*N+1],
	  float mean[M], float val[M], int arg[M])
{
  for (int i = 0; i < M; i++)
    {
      float peakval = 0.0;
      float meanval = 0.0;
      int peakarg = -1;
      for (int j = 0; j < L*N; j++)
	{
	  float a = result_I[i][j]*result_I[i][j] + result_Q[i][j]*result_Q[i][j];
	  if (a > peakval)
	    {
	      peakval = a;
	      peakarg = j;
	    }
	  meanval += a;
	}
      mean[i] = meanval/(L*N);
      val[i] = peakval;
      arg[i] = peakarg;
    }
}

int main(int argc, char **argv)
{
  char *ifname;
  char *ofname;
  int optind = 1;
  wav_hdr_com wavcom;
  wav_hdr wav;
  wav_hdr_ex wavex;
  wav_fmt wavfmt;

  if (optind < argc)
    ifname = argv[optind++];
  else
    ifname = (char *)"data.wav";

  if (optind < argc)
    ofname = argv[optind];
  else
    ofname = (char *)"qpdout.h5";

  std::cout << "input filename " << ifname << std::endl;
  std::cout << "output filename " << ofname << std::endl;
  //return 0;

  std::string ifnstr(ifname);
  std::ifstream iwavFile;
  uint32_t fsize;

  iwavFile.open(ifnstr, std::ios::in | std::ios::binary);
  if (!iwavFile.is_open())
     {
      std::cout << "can't open " << ifnstr << std::endl;
      return EXIT_FAILURE;
    }

  iwavFile.read(reinterpret_cast<char*>(&wavcom), sizeof(wavcom));
  if (iwavFile.fail())
    {
      std::cout << "can't read file" << std::endl;
      iwavFile.close();
      return EXIT_FAILURE;
    }
  if (memcmp(wavcom.RIFF, "RIFF", 4)  != 0
      || memcmp(wavcom.WAVE, "WAVE", 4)  != 0)
    {
      std::cout << "not WAV file" << std::endl;
      iwavFile.close();
      return EXIT_FAILURE;
    }

  while (true)
    {
      subchunk_hdr subchunk;
      uint32_t chunksize = 0;
      iwavFile.read(reinterpret_cast<char*>(&subchunk), sizeof(subchunk));
      if (iwavFile.fail())
	{
	  std::cout << "can't read file" << std::endl;
	  iwavFile.close();
	  return EXIT_FAILURE;
	}
      if (memcmp(subchunk.SubChunkID, "fmt ", 4)  == 0)
	{
	  chunksize = subchunk.SubChunkSize;
#if 1
	  if (chunksize != sizeof(wav_fmt) && chunksize != sizeof(wav_fmt) - 2)
	    {
	      std::cout << "unknown fmt chunk size " << chunksize << std::endl;
	      iwavFile.close();
	      return EXIT_FAILURE;
	    }
#endif
	  iwavFile.read(reinterpret_cast<char*>(&wavfmt), chunksize);
	  if (wavfmt.AudioFormat != 3  // 3=LE float
	      || wavfmt.NumOfChan != 1   // 1=Mono 2=Sterio
	      || wavfmt.SamplesPerSec != FS  // Sampling Frequency in Hz
	      || wavfmt.bytesPerSec != FS * 4 // bytes per second
	      || wavfmt.blockAlign != 4          // 2=16-bit mono, 4=16-bit stereo
	      || wavfmt.bitsPerSample != 32)      // Number of bits per sample
	    {
	      std::cout << "wav fmt should be 48k monoral 32-bit float" << std::endl;
	      iwavFile.close();
	      return EXIT_FAILURE;
	    }
	}
      else if (memcmp(subchunk.SubChunkID, "fact", 4)  == 0)
	{
	  chunksize = subchunk.SubChunkSize;
	  iwavFile.seekg(chunksize, std::ios_base::cur);
	}
      else if (memcmp(subchunk.SubChunkID, "data", 4)  == 0)
	{
	  fsize = subchunk.SubChunkSize;
	  break;
	}
      else
	{
	  chunksize = subchunk.SubChunkSize;
	  std::cout << "unknown subchunk id "
		    << subchunk.SubChunkID[0]
		    << subchunk.SubChunkID[1]
		    << subchunk.SubChunkID[2]
		    << subchunk.SubChunkID[3]
		    << std::endl;
	  iwavFile.seekg(chunksize, std::ios_base::cur);
	}
      if (iwavFile.fail())
	{
	  std::cout << "read error" << std::endl;
	  iwavFile.close();
	  return EXIT_FAILURE;
	}
    }

  // Check WAV format
  std::cout << "wav chunk size " << fsize << std::endl;
 
  // Compute test function tables
  double alpha = alpha0;
  for (int i = 0; i < M; i++)
    {
      double t = t0;
      for (int j = 0; j < L*FS/2; j++)
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
  //cout << vc[0][0] << endl;
  //cout << vc[L*FS/2-1][0] << endl;
  
#ifdef USE_CUDA
  float *d_sig;
  float (*d_vc)[M], (*d_vs)[M];
  float (*d_I)[L*N+1], (*d_Q)[L*N+1];
  cudaMalloc((void**) &d_sig, sizeof(sigbuf));
  cudaMalloc((void**) &d_vc, sizeof(vc));
  cudaMalloc((void**) &d_vs, sizeof(vs));
  cudaMalloc((void**) &d_I, sizeof(array_I));
  cudaMalloc((void**) &d_Q, sizeof(array_Q));

  cudaMemcpy(d_vc, (void*)vc, sizeof(vc), cudaMemcpyHostToDevice);
  cudaMemcpy(d_vs, (void*)vs, sizeof(vs), cudaMemcpyHostToDevice);
#endif

  iwavFile.read(reinterpret_cast<char*>(sigbuf), L*FS*sizeof(float));
#if 0
  // Clean up
  for (int i = 0; i < L*FS; i++)
   if (!isfinite(sigbuf[i]))
      {
	sigbuf[i] = 0;
	//std::cout << "not finite signal value" << std::endl;
      }
#endif

  H5::Exception::dontPrint(); // make this progam silent against exceptions.
  H5::H5File file(ofname, H5F_ACC_TRUNC);
  hsize_t dims[] = { L*N, M }, dimsmax[] = { H5S_UNLIMITED, M };
  H5::DataSpace dataspace;
  dataspace.setExtentSimple(sizeof(dims) / sizeof(hsize_t), dims, dimsmax);
  H5::DSetCreatPropList prop;
  hsize_t dimschunk[] = { L*N, M };
  prop.setChunk(sizeof(dims) / sizeof(hsize_t), dimschunk);
  H5::DataSet dataset = file.createDataSet("/qpddataset", H5::PredType::NATIVE_FLOAT, dataspace, prop);
  
  int err = 0;
  int index = 0;
  while (1)
    {
      // QPD
      auto start = chrono::system_clock::now();
#ifdef USE_CUDA
      cudaMemcpy(d_sig, sigbuf, sizeof(sigbuf), cudaMemcpyHostToDevice);
      qpd(d_sig, d_I, d_Q, d_vc, d_vs, SHIFT, MU);
      cudaMemcpy((void*)array_I, d_I, sizeof(array_I), cudaMemcpyDeviceToHost);
      cudaMemcpy((void*)array_Q, d_Q, sizeof(array_Q), cudaMemcpyDeviceToHost);
#else
      qpd(sigbuf, array_I, array_Q, vc, vs, SHIFT, MU);
#endif
      auto end = chrono::system_clock::now();
      const double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
      //cout << "time " << elapsed << " ms" << endl;

      H5::DataSpace dataspace = dataset.getSpace();
      int ndims = dataspace.getSimpleExtentNdims();
      hsize_t dims[ndims], dimsmax[ndims];
      dataspace.getSimpleExtentDims(dims, dimsmax);

      hsize_t dimsinc[] = { L*N, dims[1] };
      hsize_t dimsext[] = { dims[0] + dimsinc[0], dims[1] };
      dataset.extend( dimsext );
      hsize_t dimsoffset[] = { dims[0] - L*N, 0 };
      dataspace.selectHyperslab( H5S_SELECT_SET, dimsinc, dimsoffset );
      float data[L*N*M];
      for (int j = 0; j < L*N; j++)
	for (int i = 0; i < M; i++)
	  {
	    data[j*M+i] = array_I[i][j] * array_I[i][j] + array_Q[i][j] * array_Q[i][j];
	  }

      H5::DataSpace memspace(ndims, dimsinc);
      dataset.write( data, H5::PredType::NATIVE_FLOAT, memspace, dataspace);
#if 0
      if (elapsed < 500)
	usleep((500-elapsed)*1000);
#endif
#if 0
      for (int j = 0; j < L*N; j++)
	{
	  cout << array_I[0][j] << endl;
	}
      break;
#endif
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
      float rank = maxval/mean;
      std::cout << "rank " << rank << " (mean " << mean << " max " << maxval  << ")" << std::endl;
      std::cout << "toff " << maxargx*0.01 + index*0.5 << " alpha " << alpha0 + maxargy*alpha_step << std::endl;
      //cout << "mean " << mean << endl;
      //cout << "max " << maxval << endl;
      //cout << "maxval " << maxargx << ", " << maxargy << endl;
      index++;
      //continue;
      //cout << "qpd" << endl;
      // ...
      memmove((char *)sigbuf, (char *)&sigbuf[L*FS/2], L*FS/2*sizeof(float));
      iwavFile.read(reinterpret_cast<char*>(&sigbuf[L*FS/2]), L*FS/2*sizeof(float));
      if (iwavFile.eof())
	break;
      if (iwavFile.fail())
	{
	  err = 1;
	  break;
	}
#if 0
      // Clean up
      for (int i = L*FS/2; i < L*FS; i++)
	if (!isfinite(sigbuf[i]))
	  {
	    sigbuf[i] = 0;
	    //std::cout << "not finite signal value" << std::endl;
	  }
#endif
    }

  dataset.close();
  file.close();

  iwavFile.close();
#ifdef USE_CUDA
  cudaFree(d_sig);
  cudaFree(d_vc);
  cudaFree(d_vs);
  cudaFree(d_I);
  cudaFree(d_Q);
#endif

  if (err)
    {
      cout << "input file is corupted" << endl;
      return EXIT_FAILURE;
    }
  cout << "done" << endl;
}
