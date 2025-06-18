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

const int FS = 96000;
const int SIGNAL_RATE_PER_MS = 96;
const int VSIZE = (4 + SIGNAL_RATE_PER_MS)*4;

// net
const int SIGPORT = 5992;
char *qpd_host_addr = (char *)"10.253.253.12";
int qpd_tcp_port = 5992;

// QPD report
struct report {
  float mean_value;
  float max_value;
  int32_t maxarg_x;
  int32_t maxarg_y;
};

int watermark = 15;

// half band raw signal and report double buffers
const int NBUF = 20;
const int RAWSIZE = FS/2/2;
float raw_buf[2][RAWSIZE*(NBUF+1)];
struct report repo_buf[2][NBUF];
uint32_t raw_time[2];
int raw_cur = 0; // 0 or 1
int raw_index = 0;
std::mutex repo_mtx;
int repo_cur = 0; // 0 or 1
int repo_index = 0;
bool repo_recorded = false;

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
  uint32_t SamplesPerSec = 48000;  // Sampling Frequency in Hz
  uint32_t bytesPerSec = 48000 * 4; // bytes per second
  uint16_t blockAlign = 4;          // 2=16-bit mono, 4=16-bit stereo
  uint16_t bitsPerSample = 32;      // Number of bits per sample
  uint16_t cbSize = 0;
  /* "data" sub-chunk */
  uint8_t Subchunk2ID[4] = {'d', 'a', 't', 'a'}; // "data"  string
  uint32_t Subchunk2Size;                        // Sampled data length
} wav_hdr;

// Producer-Consumer

template<typename T> class ThreadSafeQueue {
private:
    std::queue<T> queue;
    mutable std::mutex mtx;
    std::condition_variable not_empty;
    std::condition_variable not_full;
    size_t capacity;

public:
    explicit ThreadSafeQueue(size_t max_size) : capacity(max_size) {}

    void produce(T item) {
        std::unique_lock<std::mutex> lock(mtx);
        not_full.wait(lock, [this]() { return queue.size() < capacity; });
        queue.push(std::move(item));
        lock.unlock();
        not_empty.notify_one();
    }

    T consume() {
        std::unique_lock<std::mutex> lock(mtx);
        not_empty.wait(lock, [this]() { return !queue.empty(); });
        T item = std::move(queue.front());
        queue.pop();
        lock.unlock();
        not_full.notify_one();
        return item;
    }
};

// Half band filter

template<typename T> struct HalfBandCoefficient {
  static constexpr std::array<T, 9> h0_a{
    T(0.0765690656031399), T(0.264282270318935),  T(0.47939467893641907),
    T(0.661681722389424),  T(0.7924031566294969), T(0.8776927911111817),
    T(0.9308500986629166), T(0.9640156636878193), T(0.9862978287283355),
  };
  static constexpr std::array<T, 10> h1_a{
    T(0.019911761024506557), T(0.16170648261075027), T(0.37320978687920564),
    T(0.5766558985008232),   T(0.7334355636406803),  T(0.8399227128761151),
    T(0.9074601780285125),   T(0.9492937701934973),  T(0.9760539731706528),
    T(0.9955323321150525),
  };
};

template<typename T> struct HalfBandCoefficientHiir {
  static constexpr std::array<T, 9> h0_a{
    T(0.0765690656031399), T(0.264282270318935),  T(0.4793946789364191),
    T(0.661681722389424),  T(0.792403156629497),  T(0.8776927911111816),
    T(0.9308500986629166), T(0.9640156636878193), T(0.9862978287283355),
  };
  static constexpr std::array<T, 10> h1_a{
    T(0.019911761024506557), T(0.16170648261075027), T(0.37320978687920564),
    T(0.5766558985008232),   T(0.7334355636406803),  T(0.8399227128761151),
    T(0.9074601780285125),   T(0.9492937701934973),  T(0.9760539731706528),
    T(0.9955323321150525),
  };
};

template<typename Sample> class FirstOrderAllpass {
private:
  Sample x1 = 0;
  Sample y1 = 0;

public:
  void reset()
  {
    x1 = 0;
    y1 = 0;
  }

  Sample process(Sample x0, Sample a)
  {
    y1 = a * (x0 - y1) + x1;
    x1 = x0;
    return y1;
  }
};

template<typename Sample, typename Coefficient> class HalfBandIIRSplit {
private:
  std::array<FirstOrderAllpass<Sample>, Coefficient::h0_a.size()> ap0;
  std::array<FirstOrderAllpass<Sample>, Coefficient::h1_a.size()> ap1;

public:
  void reset()
  {
    for (auto &ap : ap0) ap.reset();
    for (auto &ap : ap1) ap.reset();
  }

  // For down-sampling. input[0] must be earlier sample.
  Sample processDown(std::array<Sample, 2> &input)
  {
    auto s0 = input[0];
    for (size_t i = 0; i < ap0.size(); ++i) s0 = ap0[i].process(s0, Coefficient::h0_a[i]);
    auto s1 = input[1];
    for (size_t i = 0; i < ap1.size(); ++i) s1 = ap1[i].process(s1, Coefficient::h1_a[i]);
    return Sample(0.5) * (s0 + s1);
  }
};

template<typename Sample, size_t nSection> class FirstOrderAllpassSections {
private:
  std::array<Sample, nSection> x{};
  std::array<Sample, nSection> y{};

public:
  void reset()
  {
    x.fill(0);
    y.fill(0);
  }

  Sample process(Sample input, const std::array<Sample, nSection> &a)
  {
    for (size_t i = 0; i < nSection; ++i) {
      y[i] = a[i] * (input - y[i]) + x[i];
      x[i] = input;
      input = y[i];
    }
    return y.back();
  }
};

template<typename Sample, typename Coefficient> class HalfBandIIRArray {
private:
  FirstOrderAllpassSections<Sample, Coefficient::h0_a.size()> ap0;
  FirstOrderAllpassSections<Sample, Coefficient::h1_a.size()> ap1;

public:
  void reset()
  {
    ap0.reset();
    ap1.reset();
  }

  // For down-sampling. input[0] must be earlier sample.
  Sample processDown(std::array<Sample, 2> &input)
  {
    auto s0 = ap0.process(input[0], Coefficient::h0_a);
    auto s1 = ap1.process(input[1], Coefficient::h1_a);
    return Sample(0.5) * (s0 + s1);
  }

  // For up-sampling.
  std::array<Sample, 2> processUp(Sample input)
  {
    return {
      ap1.process(input, Coefficient::h1_a),
      ap0.process(input, Coefficient::h0_a),
    };
  }
};

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

  bool process(Sample input, Sample *output, size_t d, int ratio = 9, const Sample mu = 0.001)
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

// Pipeline workers

class ImpulseReductionPipe {
public:
  void process(int id,  ThreadSafeQueue<float> &input,  ThreadSafeQueue<float> &output)
  {
    ImpulseNoiseDetector<float, 960> impdet;
    NLMS<float, 480> reduceimp;
    NLMS<float, 480> reducewhite;
    impdet.reset();
    reduceimp.reset();
    reducewhite.reset();

    boost::circular_buffer<float> x(2);
    boost::circular_buffer<float> y(2);
    boost::circular_buffer<float> z(11);

    size_t delta = 120;
    size_t M = 960;
    float sig_in;
    float xi, yi, zi;
    bool imp;
    float tmp;

    for (size_t i = 0; i < M; i++)
      {
	sig_in = input.consume();
	impdet.process(sig_in, &xi, 2*delta);
	x.push_back(xi);
	yi = xi;
	reduceimp.process(x.front(), x.back(), &tmp, 0.05);
	y.push_back(yi);
	z.push_back(yi);
	reducewhite.process(z.back(), z.front(), &zi, 0.1);
	output.produce(zi);
      }

    for (;;)
      {
	sig_in = input.consume();
#if 1
	if (sig_in < -1.0 || sig_in > 1.0)
	  {
	    std::cout << "ImpulseReduction input value check failed " << sig_in << std::endl;
	  }
#endif
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
	z.push_back(yi);
	reducewhite.process(z.back(), z.front(), &zi, 0.1);
	output.produce(zi);
      }
  }
};

bool writerepo(struct report *r)
{
  bool recorded = false;
  for (int i = 0; i < NBUF; i++)
    {
      if (r[i].mean_value * watermark < r[i].max_value)
	{
	  recorded = true;
	  break;
	}
    }
  if (recorded)
    {
      // Write repo buf to file like as
      // "20251208_224501_3.repo"
      char fn[64];
      time_t t = std::byteswap(raw_time[~raw_cur & 1]);
      struct tm s;
      gmtime_r(&t, &s);
      memset(fn, 0, sizeof(fn));
      sprintf(fn, "%4d%02d%02d_%02d%02d%02d.repo",
	      s.tm_year+1900, s.tm_mon+1, s.tm_mday, s.tm_hour, s.tm_min, s.tm_sec);
      std::cout << "record repo file " << fn << std::endl;
#if 1
      std::ofstream repoFile;
      std::string fnstr(fn);
      repoFile.open(fnstr, std::ios::out | std::ios::binary);
      if (repoFile.is_open())
	{
	  repoFile.write(reinterpret_cast<const char*>(r), NBUF*sizeof(struct report));
	  repoFile.close();
	}
#endif
    }
  return recorded;
}

class LineReductionPipe {
public:
  void process(int id,  ThreadSafeQueue<float> &input, int qpdsock)
  {
    NLMS<float, 480> reduceline;
    reduceline.reset();

    boost::circular_buffer<float> x(1960+1);

    float sig_in;
    float e;
    float tmp;

    const int DNUM = 48*4; // 4ms
    const int HNUM = 125;  // 125*4=500
    float dbuf[DNUM];
    int didx = 0;
    int hcount = 0;
    bool first = true;
    int n;
    for (;;)
      {
	sig_in = input.consume();
#if 1
	if (sig_in < -1.0 || sig_in > 1.0)
	  {
	    std::cout << "LineReduction input value check failed " << sig_in << std::endl;
	  }
#endif
	x.push_back(sig_in);
	e = reduceline.process(x.back(), x.front(), &tmp, 0.01);
	dbuf[didx] = e;
	if (++didx == DNUM)
	  {
	    didx = 0;
	    n = write(qpdsock, (char *)dbuf, DNUM*sizeof(float));
	    if (n < 0)
	      {
		std::cout << "tcp write error" << std::endl;
	      }
	    else if (n != DNUM*sizeof(float))
	      {
		std::cout << "tcp write too few" << std::endl;
	      }
	    if (++hcount == HNUM)
	      {
		hcount = 0;
		if (!first)
		  {
		    struct report repo;
		    n = read(qpdsock, (char *)&repo, sizeof(repo));
		    if (n < 0)
		      {
			std::cout << "tcp read error" << std::endl;
		      }
		    else if (n != sizeof(repo))
		      {
			std::cout << "tcp read too few" << std::endl;
		      }
#if 0
		    std::cout << "mean " << repo.mean_value << std::endl;
		    std::cout << "max " << repo.max_value << std::endl;
		    std::cout << "maxval " << repo.maxarg_x << ", " << repo.maxarg_y << std::endl;
#endif
		    std::unique_lock<std::mutex> lock(repo_mtx);
		    repo_buf[repo_cur][repo_index] = repo;
		    if (++repo_index == NBUF)
		      {
			// Write repo if needed
			repo_recorded = writerepo(repo_buf[repo_cur]);
			repo_index = 0;
			repo_cur = ~repo_cur & 1;
		      }
		    repo_mtx.unlock();
		  }
		else
		  first = false;
	      }
	  }
	//std::cout << scount++ << " " << e << std::endl;
      }
  }
};

int main(int argc, char **argv)
{
  ThreadSafeQueue<float> q0(FS/2), q1(FS/2);
  ImpulseReductionPipe p0;
  LineReductionPipe p1;
  option longopts[] = {
    {"format", required_argument, NULL, 'f'},
    {"qpdhost", required_argument, NULL, 'h'}, 
    {"watermark", required_argument, NULL, 'w'}, {0}};
  int c;
  char hostname[16];
  bool format_wav = true;
  wav_hdr wav;

  while (1)
    {
      int option_index = 0;
      c = getopt_long(argc, argv, "f:h:w:", longopts, &option_index);
      if (c == -1)
	break;
      switch (c)
	{
	case 'f':
	  if (optarg)
	    {
	      if (0 == strcmp(optarg, "raw"))
		format_wav = false;
	      else if (0 == strcmp(optarg, "wav"))
		format_wav = true;
	      else
		std::cout << "only raw or wav format supported. Assume wav" << std::endl;
	    }
	  break;
	case 'h':
	  if (optarg)
	    {
	      char pbuf[24];
	      strncpy(pbuf, optarg, 24);
	      pbuf[23] = 0;
	      char *cp = strchr(pbuf, ':');
	      if (cp)
		{
		  *cp = 0;
		  std::string portstr(cp+1);
		  int port = std::stoi(portstr);
		  if (port > 1024 && port < 65536)
		    qpd_tcp_port = port;
		}
	      strncpy(hostname, pbuf, sizeof(hostname));
	      hostname[sizeof(hostname)-1] = 0;
	      qpd_host_addr = hostname;
	      std::cout << "setting qpd host:port to " << qpd_host_addr << ':' << qpd_tcp_port << std::endl;
	    }
	  break;
	case 'w':
	  if (optarg)
	    {
	      std::string wmstr(optarg);
	      int wm = std::stoi(wmstr);
	      if (wm > 0 && wm < 30)
		{
		  watermark = wm;
		  std::cout << "setting water mark to " << wm << std::endl;
		}
	      else
		std::cout << "water mark " << wm << "out of range (0, 30)" << std::endl;
	    }
	  break;
	default:
	  break;
	}
    }

  //return 0;

  int sigsock = socket(AF_INET, SOCK_DGRAM, 0);
  if (sigsock < 0)
    {
      return EXIT_FAILURE;
    }

  struct sockaddr_in ca;
  memset(&ca, 0, sizeof(ca));
  ca.sin_family = AF_INET;
  ca.sin_addr.s_addr = INADDR_ANY;
  ca.sin_port = htons(SIGPORT);

  if (bind(sigsock, (struct sockaddr *)&ca, sizeof(ca)) < 0)
    {
      close(sigsock);
      std::cout << "can't bind input socket" << std::endl;
      return EXIT_FAILURE;
    }

  int qpdsock = socket(AF_INET, SOCK_STREAM, 0);
  if (qpdsock < 0)
    {
      close(sigsock);
      std::cout << "can't get socket for qpd" << std::endl;
      return EXIT_FAILURE;
    }

  struct sockaddr_in sa;
  memset(&sa, 0, sizeof(sa));
  sa.sin_family = AF_INET;
  if (inet_pton(AF_INET, qpd_host_addr, (void *)&sa.sin_addr.s_addr) != 1)
    {
      close(sigsock);
      std::cout << "invalid ip address format" << std::endl;
      return EXIT_FAILURE;
    }
  sa.sin_port = htons(qpd_tcp_port);
  if (connect(qpdsock, (struct sockaddr *)&sa, sizeof(sa)) < 0)
    {
      close(qpdsock);
      close(sigsock);
      std::cout << "can't connect qpd server" << std::endl;
      return EXIT_FAILURE;
    }
  std::cout << "qpd server connected" << std::endl;

  std::thread ith(&ImpulseReductionPipe::process, &p0, 1, std::ref(q0), std::ref(q1));
  std::thread lth(&LineReductionPipe::process, &p1, 2, std::ref(q1), qpdsock);

  std::array<float, 2> phases{};

#if 0
  auto start = std::chrono::steady_clock::now();
#endif

  HalfBandIIRSplit<float, HalfBandCoefficient<float>> halfbandiir;
  halfbandiir.reset();

  char vbuf[VSIZE];
  int32_t *sigbuf = (int32_t *)(vbuf + 4*4);
  while (true)
    {
      int n = recv(sigsock, vbuf, VSIZE, 0);
      if (n < 0)
	{
	  std::cout << "recv failed" << std::endl;
	  break;
	}

      for (size_t i = 0; i < SIGNAL_RATE_PER_MS; i += 2)
	{
	  phases[0] = sigbuf[i]/float(1LL<<31);
	  phases[1] = sigbuf[i + 1]/float(1LL<<31);

	  float hbsig = halfbandiir.processDown(phases);
#if 1
	  if (hbsig < -1.0 || hbsig > 1.0)
	    {
	      std::cout << "hbsig value check failed " << hbsig << std::endl;
	      return EXIT_FAILURE;
	    }
#endif
	  q0.produce(hbsig);
	  raw_buf[raw_cur][raw_index] = hbsig;
	  if (raw_index == 0 || raw_index == RAWSIZE*NBUF)
	    raw_time[raw_cur] = ((uint32_t *)vbuf)[1];
	  if (++raw_index == RAWSIZE*(NBUF+1))
	    {
	      memcpy((char *)&raw_buf[~raw_cur & 1][0],
		     (char *)&raw_buf[raw_cur][RAWSIZE*NBUF],
		     RAWSIZE*sizeof(float));
	      raw_index = RAWSIZE;
	      raw_cur = ~raw_cur & 1;
	    }
	  if (raw_index % RAWSIZE == 0)
	    {
	      std::unique_lock<std::mutex> lock(repo_mtx);
	      if (repo_recorded)
		{
		  repo_recorded = false;
		  char fn[64];
		  time_t t = std::byteswap(raw_time[~raw_cur & 1]);
		  struct tm s;
		  gmtime_r(&t, &s);
		  memset(fn, 0, sizeof(fn));
		  const char *ext = format_wav ? "wav" : "raw";
		  sprintf(fn, "%4d%02d%02d_%02d%02d%02d.%s",
			  s.tm_year+1900, s.tm_mon+1, s.tm_mday, s.tm_hour, s.tm_min, s.tm_sec, ext);
		  std::cout << "record signal data file " << fn << std::endl;
#if 1
		  std::string fnstr(fn);
		  std::ofstream rawFile;
		  rawFile.open(fnstr, std::ios::out | std::ios::binary);
		  if (rawFile.is_open())
		    {
		      uint32_t fsize = RAWSIZE*(NBUF+1)*sizeof(float);
		      if (format_wav)
			{
			  wav.ChunkSize = fsize + sizeof(wav_hdr) - 8;
			  wav.Subchunk2Size = fsize;
			  rawFile.write(reinterpret_cast<const char*>(&wav), sizeof(wav));
			}
		      rawFile.write(reinterpret_cast<const char*>(&raw_buf[~raw_cur & 1][0]), fsize);
		      rawFile.close();
		    }
#endif
		}
	      repo_mtx.unlock();
	    }
	}
    }
#if 0
  auto end = std::chrono::steady_clock::now();
  const double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
  std::cout << "time " << elapsed << "ms " << std::endl;
#endif

  close(sigsock);
 
  ith.join();
  lth.join();
}
