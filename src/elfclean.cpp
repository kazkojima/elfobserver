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
  int32_t argmax_x;
  int32_t argmax_y;
};

int watermark = 15;

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

// Imp]ulse noise detector

template<typename Sample, size_t Length> class ImpulseNoiseDetector {
private:
  std::array<Sample, Length> x{};
  Sample mx;
  size_t last;
  int counter;
  //const Sample mu = 0.005;

public:
  void reset()
  {
    x.fill(0);
    mx = (Sample) 0;
    last = 0;
    counter = 0;
  }

  bool process(Sample input, Sample *output, size_t d, int ratio = 11, const Sample mu = 10.0/FS)
  {
    size_t i = last + 1;
    if (i >= Length)
      i = 0;
    last = i;
    x[i] = input;
    mx = (1-mu)*mx + mu*abs(input);
    if (abs(input) >= ratio*mx/2)
      counter = Length;
    else if (counter > 0)
      --counter;

    size_t delta = d/2;
    Sample yh;
    if (i >= delta)
      yh = x[i-delta];
    else
      yh = x[i+Length-delta];
    *output = yh;

    if (counter == 0)
      return false;

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
    {"print-impulse-section", no_argument, NULL, 's'},
    {"line-delay", required_argument, NULL, 'd'},
    {"detect-ratio", required_argument, NULL, 'r'},
    {"help", no_argument, NULL, 'h'},
    {0}};
  int c;
  char *ifname;
  char *ofname;
  //bool format_wav = true;
  bool skip_impulse = false;
  bool skip_white = false;
  bool skip_line = false;
  bool print_section = false;
  int line_delay = 960;
  float detect_ratio = 11;

  wav_hdr_com wavcom;
  wav_hdr wav;
  wav_hdr_ex wavex;

  ifname = NULL;
  ofname = NULL;

  while (1)
    {
      int option_index = 0;
      c = getopt_long(argc, argv, "i:o:pwlsd:r:h", longopts, &option_index);
      if (c == -1)
	break;
      switch (c)
	{
	case 'd':
	  if (optarg)
	    {
	      char *end;
	      int delay = std::strtol(optarg, &end, 10);
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
	      std::cout << "-d option requires int arg" << std::endl;
	    }
	  break;
	case 'r':
	  if (optarg)
	    {
	      char *end;
	      float ratio = std::strtof(optarg, &end);
	      if (ratio >= 6 && ratio < 20)
		{
		  detect_ratio = ratio;
		  std::cout << "setting detect-ratio to " << ratio << std::endl;
		}
	      else
		std::cout << "wrong detect-ratio value " << ratio << std::endl;
	    }
	  else
	    {
	      std::cout << "-r option requires float arg" << std::endl;
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
	case 's':
	  print_section = true;
	  break;
	case 'h':
	  std::cout << "usage: elfclean [option]... [input file] [output file]" << std::endl;
	  std::cout << std::endl;
	  std::cout << "A simple noise reduction tool for ELF WAV" << std::endl;
	  std::cout << std::endl;
	  std::cout << "-i, --input=filename\tgive input file as an option" << std::endl;
	  std::cout << "-o, --output=filename\tgive output file as an option" << std::endl;
	  std::cout << "-p, --skip-impulse\tskip impulse noise canceller" << std::endl;
	  std::cout << "-w, --skip-white\tskip white noise canceller" << std::endl;
	  std::cout << "-l, --skip-line\t\tskip line reducer" << std::endl;
	  std::cout << "-s, --print-impulse-section\tprint impulse section" << std::endl;
	  std::cout << "-d, --line-delay=N\tset the delay for line reducer" << std::endl;
	  std::cout << "-r, --detect-ratio=R\tset the ratio for impulse detector" << std::endl;
	  std::cout << "-h, --help\t\tdisplay this help and exit" << std::endl;

	  return 0;
	default:
	  break;
	}
    }

  if (ifname == NULL)
    {
      if (optind < argc )
	ifname = argv[optind++];
      else
	ifname = (char *)"data.wav";
    }
  if (ofname == NULL)
    {
      if (optind < argc )
	ofname = argv[optind];
      else
	ofname = (char *)"clean.wav";
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
	}
      else if (memcmp(subchunk.SubChunkID, "fact", 4)  == 0)
	{
	  chunksize = subchunk.SubChunkSize;
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
	}
      iwavFile.seekg(chunksize, std::ios_base:: cur);
      if (iwavFile.fail())
	{
	  std::cout << "read error" << std::endl;
	  iwavFile.close();
	  return EXIT_FAILURE;
	}
    }

  // Check WAV format
  std::cout << "wav chunk size " << fsize << std::endl;
  ibuf = (float *)malloc(fsize);
  obuf = (float *)malloc(fsize);
  iwavFile.read(reinterpret_cast<char*>(ibuf), fsize);
  if (iwavFile.fail())
    {
      std::cout << "can't read file" << std::endl;
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

  size_t delta = 120;
  size_t M = 960;
  float sig_in;
  float xi, yi, zi;
  float e;
  bool imp;
  bool imp_prev = false;
  float tmp;

  // training for 1.0sec
  for (size_t i = 0; i < FS && i < fsize/sizeof(float) ; i++)
    {
      sig_in = ibuf[i];

      if (i < M)
	{
	  impdet.process(sig_in, &xi, 2*delta, detect_ratio);
	  x.push_back(xi);
	  yi = xi;
	  reduceimp.process(x.front(), x.back(), &tmp, 0.05);
	  y.push_back(yi);
	  z.push_back(yi);
	  reducewhite.process(z.back(), z.front(), &zi, 0.1);
	  w.push_back(zi);
	  reduceline.process(w.back(), w.front(), &tmp, 0.01);
	}
      else
	{
	  if (skip_impulse)
	    yi = sig_in;
	  else
	    {
	      imp = impdet.process(sig_in, &xi, 2*delta, detect_ratio);
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
	  if (!skip_line)
	    {
	      w.push_back(zi);
	      reduceline.process(w.back(), w.front(), &tmp, 0.01);
	    }
	}
    }

  for (size_t i = 0; i < fsize/sizeof(float); i++)
    {
      sig_in = ibuf[i];

      if (skip_impulse)
	yi = sig_in;
      else
	{
	  imp = impdet.process(sig_in, &xi, 2*delta, detect_ratio);
	  if (print_section)
	    {
	      float t = (float)i/FS;
	      if (imp_prev == false && imp == true)
		std::cout << "impulse section rising " << t << std::endl;
	      if (imp_prev == true && imp == false)
		std::cout << "impulse section falling " << t << std::endl;
	      imp_prev = imp;
	    }
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
  wav_hdr owav;
  owav.ChunkSize = fsize + sizeof(wav_hdr) - 8;
  owav.Subchunk2Size = fsize;
  owavFile.write(reinterpret_cast<const char*>(&owav), sizeof(owav));
  if (owavFile.fail())
    {
      std::cout << "can't write file header" << std::endl;
      owavFile.close();
      return EXIT_FAILURE;
    }
  owavFile.write(reinterpret_cast<const char*>(obuf), fsize);
  if (owavFile.fail())
    {
      std::cout << "can't write file" << std::endl;
      owavFile.close();
      return EXIT_FAILURE;
    }

  owavFile.close();
}
