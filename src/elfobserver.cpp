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
#include <random>
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

const uint32_t TS_RESET = std::byteswap(0xfffffffe);

// net
const int SIGPORT = 5992;
char *qpd_host_addr = (char *)"10.253.253.12";
int qpd_tcp_port = 5992;

// QPD report
struct report {
  float mean_value;
  float max_value;
  int32_t argmax_x;
  int32_t argmax_y;
};

int watermark = 15;
float output_rate = 1.0;
float input_volume = 1.0;
float detect_ratio = 11;

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
  static constexpr std::array<T, 19> h_a {
    T(0.03638363347126123),
    T(-2.334400920957579e-05),
    T(-0.03401449150543288),
    T(-2.1398290149378164e-05),
    T(0.05503008321776456),
    T(-9.426083410643402e-06),
    T(-0.1007807593581111),
    T(1.1574718364718162e-05),
    T(0.3164821770223705),
    T(0.5000199761073955),
    T(0.3164821770223705),
    T(1.1574718364718162e-05),
    T(-0.1007807593581111),
    T(-9.426083410643402e-06),
    T(0.05503008321776456),
    T(-2.1398290149378164e-05),
    T(-0.03401449150543288),
    T(-2.334400920957579e-05),
    T(0.03638363347126123),
  };
};

template<typename Sample, typename Coefficient> class HalfBandFIR {
private:
  boost::circular_buffer<Sample> c_buf;

public:
  void reset()
  {
    c_buf.resize(Coefficient::h_a.size());
  }
  
  Sample processDown(std::array<Sample, 2> &input)
  {
    auto s0 = input[0];
    c_buf.push_back(s0);
    s0 = Sample(0.0);
    for (size_t i = 0; i < c_buf.size(); ++i)
      s0 += c_buf[i]*Coefficient::h_a[i];
    auto s1 = input[1];
    c_buf.push_back(s1);
    s1 = Sample(0.0);
    for (size_t i = 0; i < c_buf.size(); ++i)
      s1 += c_buf[i]*Coefficient::h_a[i];
    return Sample(0.5) * (s0 + s1);
  }
};

// Impulse noise detector

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

  bool process(Sample input, Sample *output, size_t d, int ratio = 11, const Sample mu = 10.0/(FS/2))
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

// Pipeline workers

class ImpulseReductionPipe {
public:
  void process(int id,  ThreadSafeQueue<float> &input,  ThreadSafeQueue<float> &output)
  {
    ImpulseNoiseDetector<float, 960> impdet;
    NLMS<float, 480> reduceimp;
    impdet.reset();
    reduceimp.reset();

    boost::circular_buffer<float> x(2);
    boost::circular_buffer<float> y(2);

    size_t delta = 120;
    size_t M = 960;
    float mu = 0.05;
    float ratio = detect_ratio;
    float sig_in;
    float xi, yi;//, zi;
    bool imp;
    float tmp;

    for (size_t i = 0; i < M; i++)
      {
	sig_in = input.consume();
	impdet.process(sig_in, &xi, 2*delta, ratio);
	x.push_back(xi);
	yi = xi;
	reduceimp.process(x.front(), x.back(), &tmp, mu);
	y.push_back(yi);
	output.produce(yi);
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
	imp = impdet.process(sig_in, &xi, 2*delta, ratio);
	x.push_back(xi);
	if (imp == false)
	  {
	    yi = xi;
	    reduceimp.process(x.front(), x.back(), &tmp, mu);
	  }
	else
	  {
	    reduceimp.process(y.back(), 0.0, &yi, mu, false);
	  }
#if 0
	if (yi < -1.0 || yi > 1.0)
	  {
	    std::cout << "ImpulseReduction output value check failed " << yi << std::endl;
	  }
#endif
	// Clamp yi
	yi = std::clamp(yi, -0.5f, 0.5f);
	y.push_back(yi);
	output.produce(yi);
      }
  }
};

class WhiteNoiseReductionPipe {
public:
  void process(int id,  ThreadSafeQueue<float> &input,  ThreadSafeQueue<float> &output)
  {
    NLMS<float, 480> reducewhite;
    reducewhite.reset();

    boost::circular_buffer<float> z(11);
    float sig_in, zi;

    for (;;)
      {
	sig_in = input.consume();
	z.push_back(sig_in);
	reducewhite.process(z.back(), z.front(), &zi, 0.1);
	output.produce(zi);
      }
  }
};

bool print_info = true;
bool print_statistics = false;
size_t statistics_count;
const int LOWWM = 12;
const int HIGHWM = 20;
size_t ranksize[HIGHWM+1];

std::random_device seed_gen;
std::mt19937 random_engine(seed_gen());

bool writeRepo(struct report *r)
{
  bool recorded = false;
  float maxrank = 0;
  for (int i = 0; i < NBUF; i++)
    {
      float rank = r[i].max_value/r[i].mean_value;
      if (rank > maxrank)
	maxrank = rank;
#if 1
      if (400 * input_volume < r[i].max_value)
	std::cout << "suspicious high max: " << r[i].max_value << std::endl;
#endif
     }
  if (output_rate != 1.0)
    {
      constexpr std::size_t bits = std::numeric_limits<float>::digits;
      float result = std::generate_canonical<float, bits>(random_engine);
      if (output_rate > result && maxrank > watermark)
	recorded = true;
    }
  else
    recorded = maxrank > watermark ? true : false;
      
  if (print_statistics)
    {
      int idx = std::clamp((int)maxrank, LOWWM, HIGHWM);
      ranksize[idx]++;
      // writeRepo is called every 10sec. Print statistics every 1hour.
      if (++statistics_count % 360 == 0)
	{
	  std::cout << "[- " << LOWWM << "]:" << ranksize[LOWWM];
	  for (int i = LOWWM+1; i < HIGHWM; i++)
	    std::cout << " [" << i << "]:" << ranksize[i];
	  std::cout << " [" << HIGHWM << " -]:" << ranksize[HIGHWM];
	  std::cout << std::endl;
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
      if (print_info)
	std::cout << "record repo file " << fn << " rank " << maxrank << std::endl;
      std::ofstream repoFile;
      std::string fnstr(fn);
      repoFile.open(fnstr, std::ios::out | std::ios::binary);
      if (repoFile.is_open())
	{
	  repoFile.write(reinterpret_cast<const char*>(r), NBUF*sizeof(struct report));
	  repoFile.close();
	}
#if 0
      for (size_t i = 0; i < NBUF; i++)
	{
	  float rank = r[i].max_value/r[i].mean_value;
	  std::cout << "rank " << rank << " (mean " << r[i].mean_value << " max " << r[i].max_value  << ")" << std::endl;
	  std::cout << "toff " << r[i].argmax_x*0.01 + i*0.5 << std::endl;
	  std::cout << "alpha " << r[i].argmax_y + 2300 << std::endl;
	}
#endif
    }
  return recorded;
}

class LineReductionPipe {
public:
  void process(int id,  ThreadSafeQueue<float> &input, int qpdsock)
  {
    NLMS<float, 960> reduceline;
    reduceline.reset();

    boost::circular_buffer<float> x(960+1);

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
		    std::cout << "maxval " << repo.argmax_x << ", " << repo.argmax_y << std::endl;
#endif
		    std::unique_lock<std::mutex> lock(repo_mtx);
		    repo_buf[repo_cur][repo_index] = repo;
		    if (++repo_index == NBUF)
		      {
			// Write repo if needed
			repo_recorded = writeRepo(repo_buf[repo_cur]);
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
  ThreadSafeQueue<float> q0(FS/2), q1(FS/2), q2(FS/2);
  ImpulseReductionPipe p0;
  WhiteNoiseReductionPipe p1;
  LineReductionPipe p2;
  option longopts[] = {
    {"format", required_argument, NULL, 'f'},
    {"qpdhost", required_argument, NULL, 'q'}, 
    {"watermark", required_argument, NULL, 'w'},
    {"rate", required_argument, NULL, 'r'},
    {"volume", required_argument, NULL, 'v'},
    {"detect-ratio", required_argument, NULL, 'd'},
    {"noinfo", no_argument, NULL, 'n'},
    {"statistics", no_argument, NULL, 's'},
    {"help", no_argument, NULL, 'h'},
   {0}};
  int c;
  char hostname[16];
  bool format_wav = true;
  wav_hdr wav;

  while (1)
    {
      int option_index = 0;
      c = getopt_long(argc, argv, "f:q:w:r:v:d:nsh", longopts, &option_index);
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
	case 'q':
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
	      if (wm >= 0 && wm < 30)
		{
		  watermark = wm;
		  std::cout << "setting watermark to " << wm << std::endl;
		}
	      else
		std::cout << "watermark " << wm << "out of range [0, 30)" << std::endl;
	    }
	  break;
	case 'r':
	  if (optarg)
	    {
	      char *end;
	      float rate = std::strtof(optarg, &end);
	      if (rate >= 0 && rate <= 1)
		{
		  output_rate = rate;
		  std::cout << "setting output rate to " << rate << std::endl;
		}
	      else
		std::cout << "wrong output rate value " << rate << std::endl;
	    }
	  else
	    {
	      std::cout << "-r option requires float arg" << std::endl;
	    }
	  break;
	case 'v':
	  if (optarg)
	    {
	      char *end;
	      float vol = std::strtof(optarg, &end);
	      if (vol >= 0 && vol <= 2.0)
		{
		  input_volume = vol;
		  std::cout << "setting input volume to " << vol << std::endl;
		}
	      else
		std::cout << "wrong input volume value " << vol << std::endl;
	    }
	  else
	    {
	      std::cout << "-v option requires float arg" << std::endl;
	    }
	  break;
	case 'd':
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
	case 'n':
	  print_info = false;
	  break;
	case 's':
	  print_statistics = true;
	  break;
	case 'h':
	  std::cout << "usage: elfobserver [option]..." << std::endl;
	  std::cout << std::endl;
	  std::cout << "observe ELF and record/report interesting activities" << std::endl;
	  std::cout << std::endl;
	  std::cout << "-f, --format=[wav,raw]\tset input file format" << std::endl;
	  std::cout << "-q, --qpdhost=host:port\tset qpd host address&port" << std::endl;
	  std::cout << "-w, --watermark=N\tset watermark" << std::endl;
	  std::cout << "-r, --rate=F\tset output rate" << std::endl;
	  std::cout << "-v, --volume=F\tset input volume" << std::endl;
	  std::cout << "-d, --detect-ratio=R\tset the ratio for impulse detector" << std::endl;
	  std::cout << "-n, --noinfo\tdon't print info for repo file" << std::endl;
	  std::cout << "-s, --statistics\tprint rank-size every 1 hour" << std::endl;
	  std::cout << "-h, --help\t\tdisplay this help and exit" << std::endl;
	  return 0;
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

  // Pipeline
  std::thread ith(&ImpulseReductionPipe::process, &p0, 1, std::ref(q0), std::ref(q1));
  std::thread wth(&WhiteNoiseReductionPipe::process, &p1, 1, std::ref(q1), std::ref(q2));
  std::thread lth(&LineReductionPipe::process, &p2, 2, std::ref(q2), qpdsock);

  std::array<float, 2> phases{};

#if 0
  auto start = std::chrono::steady_clock::now();
#endif

  HalfBandFIR<float, HalfBandCoefficient<float>> halfbandfir;
  halfbandfir.reset();

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

      if (((uint32_t *)vbuf)[1] == TS_RESET)
	{
	  raw_index = 0;
	  //std::cout << "got reset packet" << std::endl;
	  continue;
	}

      for (size_t i = 0; i < SIGNAL_RATE_PER_MS; i += 2)
	{
	  phases[0] = input_volume * (sigbuf[i]/float(1LL<<31));
	  phases[1] = input_volume * (sigbuf[i + 1]/float(1LL<<31));

	  float hbsig = halfbandfir.processDown(phases);
#if 1
	  if (hbsig < -1.0 || hbsig > 1.0)
	    {
	      std::cout << "hbsig value check failed " << hbsig << std::endl;
	      //return EXIT_FAILURE;
	    }
#endif
	  q0.produce(hbsig);
	  raw_buf[raw_cur][raw_index] = hbsig;
	  if (raw_index == 0)
	    raw_time[raw_cur] = ((uint32_t *)vbuf)[1];
	  else if (raw_index == RAWSIZE*NBUF)
	    raw_time[~raw_cur & 1] = ((uint32_t *)vbuf)[1];
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
		  if (print_info)
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

  // Formal cleanup
  ith.join();
  wth.join();
  lth.join();
}
