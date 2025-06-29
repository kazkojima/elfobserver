#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <boost/circular_buffer.hpp>
#include <cstring>
#include <netinet/in.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <time.h>
#include <getopt.h>

using namespace std;

const int FS = 48000;

// QPD report
struct report {
  float mean_value;
  float max_value;
  int32_t maxarg_x;
  int32_t maxarg_y;
};

int watermark = 15;

// WAV
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
#if 0
  uint8_t SubchunkEID[4] = {'f', 'a', 'c', 't'};
  uint8_t SubchunkEData[8] = { 0x04,0x00,0x00,0x00,0x80,0x32,0x02,0x00};
#endif
  /* "data" sub-chunk */
  uint8_t Subchunk2ID[4] = {'d', 'a', 't', 'a'}; // "data"  string
  uint32_t Subchunk2Size;                        // Sampled data length
} wav_hdr;

// Imp]ulse noise detector

template<typename Sample, size_t Length> class ImpulseNoiseDetector {
private:
  std::array<Sample, Length> x{};
  Sample mx;
  size_t last;
  //const Sample mu = 0.005;

public:
  void reset()
  {
    x.fill(0);
    mx = (Sample) 0;
    last = 0;
  }

  bool process(Sample input, Sample *output, size_t d, int ratio = 11, const Sample mu = 10.0/FS)
  {
    size_t i = last + 1;
    if (i >= Length)
      i = 0;
    last = i;
    x[i] = input;
    mx = (1-mu)*mx + mu*abs(input);

    size_t delta = d/2;
    Sample yh;
    if (i >= delta)
      yh = x[i-delta];
    else
      yh = x[i+Length-delta];

    bool e = false;
    for (size_t j = 0; j < d; ++j)
      {
	int xi = i - d + j;
	if (xi >= (int)Length)
	  xi -= Length;
	else if (xi < 0)
	  xi += Length;
	if (abs(x[xi]) > ratio*mx)
	  {
	    e = true;
	    break;
	  }
      }
    *output = yh;
    return e;
  }
};

// NLMS filter

template<typename Sample, size_t Length> class NLMS {
private:
  std::array<Sample, Length> x{};
  std::array<Sample, Length> w{};
  size_t last;

public:
  void reset()
  {
    x.fill(0);
    w.fill(0);
    last = 0;
  }

  Sample process(Sample input, Sample ref, Sample *output, const Sample mu, bool update = true, const Sample a = 1e-6)
  {
    size_t i = last + 1;
    if (i >= Length)
      i = 0;
    last = i;
    x[i] = input;
  
    size_t wi = 0;
    Sample wTx = 0;
    Sample xsq = 0;

    for (size_t xi = i; xi < Length; xi++)
      {
	Sample fxi = x[xi];
	wTx += fxi*w[wi];
	xsq += fxi*fxi;
	wi += 1;
      }
    for (size_t xi = 0; xi < i; xi++)
      {
	Sample fxi = x[xi];
	wTx += fxi*w[wi];
	xsq += fxi*fxi;
	wi += 1;
      }

    Sample yh = wTx;
    Sample e = ref - yh;
    if (update)
      {
	Sample mue = mu*e/(xsq + a);
	wi = 0;
	for (size_t xi = i; xi < Length; xi++)
	  {
	    w[wi] += mue*x[xi];
	    wi += 1;
	  }
	for (size_t xi = 0; xi < i; xi++)
	  {
	    w[wi] += mue*x[xi];
	    wi += 1;
	  }
      }

    *output = yh;
    return e;
  }
};

int main(int argc, char **argv)
{
  option longopts[] = {
    {"input", required_argument, NULL, 'i'},
    {"output", required_argument, NULL, 'o'},
    {"skip-impulse", no_argument, NULL, 'p'},
    {"skip-white", no_argument, NULL, 'w'},
    {"skip-line", no_argument, NULL, 'l'},
    {"line-delay", required_argument, NULL, 'd'},
    {0}};
  int c;
  char *ifname;
  char *ofname;
  //bool format_wav = true;
  bool skip_impulse = false;
  bool skip_white = false;
  bool skip_line = false;
  int line_delay = 1920;
  wav_hdr wav;

  ifname = (char *)"data.wav";
  ofname = (char *)"clean.wav";

  while (1)
    {
      int option_index = 0;
      c = getopt_long(argc, argv, "i:o:pwld:", longopts, &option_index);
      if (c == -1)
	break;
      switch (c)
	{
	case 'd':
	  if (optarg)
	    {
	      char nbuf[8];
	      strncpy(nbuf, optarg, 8);
	      nbuf[7] = 0;
	      std::string nstr(nbuf);
	      int delay = std::stoi(nstr);
	      if (delay >= 480 && delay < 480*6)
		{
		  line_delay = delay;
		  std::cout << "setting line-delay to " << delay << std::endl;
		}
	      else
		std::cout << "wrong line-delay value " << delay << std::endl;
	    }
	  else
	    {
	      std::cout << "-d option requires finename arg" << std::endl;
	    }
	  break;
	case 'i':
	  if (optarg)
	    {
	      ifname = optarg;
	      //std::cout << "input filename " << ifname << std::endl;
	    }
	  else
	    {
	      std::cout << "-i option requires finename arg" << std::endl;
	    }
	  break;
	case 'o':
	  if (optarg)
	    {
	      ofname = optarg;
	      //std::cout << "output filename " << ofname << std::endl;
	    }
	  else
	    {
	      std::cout << "-o option requires finename arg" << std::endl;
	    }
	  break;
	case 'p':
	  skip_impulse = true;
	  break;
	case 'w':
	  skip_white = true;
	  break;
	case 'l':
	  skip_line = true;
	  break;
	default:
	  break;
	}
    }

  std::cout << "input filename " << ifname << std::endl;
  std::cout << "output filename " << ofname << std::endl;
  //return 0;

  float *ibuf;
  float *obuf;
  std::string ifnstr(ifname);
  std::ifstream iwavFile;
  uint32_t fsize;

  iwavFile.open(ifnstr, std::ios::in | std::ios::binary);
  if (!iwavFile.is_open())
     {
      std::cout << "can't open " << ifnstr << std::endl;
      return EXIT_FAILURE;
    }

  iwavFile.read(reinterpret_cast<char*>(&wav), sizeof(wav));
  // Check WAV format
  fsize = wav.Subchunk2Size;
  cout << "wav chunk size " << fsize << endl;
  ibuf = (float *)malloc(fsize);
  obuf = (float *)malloc(fsize);
  iwavFile.read(reinterpret_cast<char*>(ibuf), fsize);
  if (iwavFile.fail())
    {
      cout << "can't read file" << endl;
      iwavFile.close();
      return EXIT_FAILURE;
    }
  else
    iwavFile.close();

  ImpulseNoiseDetector<float, 960> impdet;
  NLMS<float, 480> reduceimp;
  NLMS<float, 480> reducewhite;
  NLMS<float, 960> reduceline;
  impdet.reset();
  reduceimp.reset();
  reducewhite.reset();
  reduceline.reset();

  boost::circular_buffer<float> x(2);
  boost::circular_buffer<float> y(2);
  boost::circular_buffer<float> z(11);
  boost::circular_buffer<float> w(line_delay+1);


#if 0
  auto start = std::chrono::steady_clock::now();
#endif

  for (size_t i = 0; i < fsize/sizeof(float); i++)
    {
      size_t delta = 120;
      size_t M = 960;
      float sig_in;
      float xi, yi, zi;
      float e;
      bool imp;
      float tmp;

      sig_in = ibuf[i];

      if (i < M)
	{
	  impdet.process(sig_in, &xi, 2*delta);
	  x.push_back(xi);
	  yi = xi;
	  reduceimp.process(x.front(), x.back(), &tmp, 0.05);
	  y.push_back(yi);
	  z.push_back(yi);
	  reducewhite.process(z.back(), z.front(), &zi, 0.1);
	  w.push_back(zi);
	  e = reduceline.process(w.back(), w.front(), &tmp, 0.01);
	  obuf[i] = e;
	}
      else
	{
	  if (skip_impulse)
	    yi = sig_in;
	  else
	    {
	      imp = impdet.process(sig_in, &xi, 2*delta);
	      x.push_back(xi);
	      if (imp == false)
		{
		  yi = xi;
		  reduceimp.process(x.front(), x.back(), &tmp, 0.05);
		}
	      else
		{
		  reduceimp.process(y.back(), 0.0, &yi, 0.05, false);
		}
	      y.push_back(yi);
	    }
	  if (skip_white)
	    zi = yi;
	  else
	    {
	      z.push_back(yi);
	      reducewhite.process(z.back(), z.front(), &zi, 0.1);
	    }
	  if (skip_line)
	    e = zi;
	  else
	    {
	      w.push_back(zi);
	      e = reduceline.process(w.back(), w.front(), &tmp, 0.01);
	    }
	  obuf[i] = e;
	}
    }

#if 0
  auto end = std::chrono::steady_clock::now();
  const double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
  std::cout << "time " << elapsed << "ms " << std::endl;
#endif

  std::string ofnstr(ofname);
  std::ofstream owavFile;
  owavFile.open(ofnstr, std::ios::out | std::ios::binary);
  if (!owavFile.is_open())
    {
      std::cout << "can't open " << ofnstr << std::endl;
      return EXIT_FAILURE;
    }
  owavFile.write(reinterpret_cast<const char*>(&wav), sizeof(wav));
  if (owavFile.fail())
    {
      cout << "can't write file header" << endl;
      owavFile.close();
      return EXIT_FAILURE;
    }
  owavFile.write(reinterpret_cast<const char*>(obuf), fsize);
  if (owavFile.fail())
    {
      cout << "can't write file" << endl;
      owavFile.close();
      return EXIT_FAILURE;
    }

  owavFile.close();
}
