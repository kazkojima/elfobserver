#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <iostream>
#include <fstream>
#include <cstring>
#include <unistd.h>
#include <time.h>
#include <getopt.h>

const int FS = 48000;
const int NR = 20;

// QPD report
struct report {
  float mean_value;
  float max_value;
  int32_t argmax_x;
  int32_t argmax_y;
} repo[NR];

int main(int argc, char **argv)
{
  option longopts[] = {
    {"input", required_argument, NULL, 'i'},
    {"print-all", no_argument, NULL, 'a'},
    {0}};
  int c;
  char *ifname;
  bool print_all = false;

  ifname = NULL;
  while (1)
    {
      int option_index = 0;
      c = getopt_long(argc, argv, "i:a", longopts, &option_index);
      if (c == -1)
	break;
      switch (c)
	{
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
	case 'a':
	  print_all = true;
	  break;
	default:
	  break;
	}
    }

  if (ifname == NULL)
    ifname = argv[optind];
  if (ifname == NULL)
    {
      std::cout << "no input file" << std::endl;
      return EXIT_FAILURE;
    }

  std::string ifnstr(ifname);
  std::ifstream iFile;

  iFile.open(ifnstr, std::ios::in | std::ios::binary);
  if (!iFile.is_open())
     {
      std::cout << "can't open " << ifnstr << std::endl;
      return EXIT_FAILURE;
    }

  iFile.read(reinterpret_cast<char*>(&repo), sizeof(repo));
  if (iFile.fail())
    {
      std::cout << "can't read report file" << std::endl;
      iFile.close();
      return EXIT_FAILURE;
    }
  else
    iFile.close();

  float maxrank = 0;
  float meanmaxrank = 1;
  float maxmaxrank = 0;
  int argmaxrank_x = 0;
  int argmaxrank_y = 0;
  int indexmax = 0;
  for (size_t i = 0; i < NR; i++)
    {
      float rank = repo[i].max_value/repo[i].mean_value;
      if (rank > maxrank)
	{
	  maxrank = rank;
	  meanmaxrank = repo[i].mean_value;
	  maxmaxrank = repo[i].max_value;
	  argmaxrank_x = repo[i].argmax_x;
	  argmaxrank_y = repo[i].argmax_y;
	  indexmax = i;
	}
      if (print_all)
	{
	  std::cout << "rank " << rank << " (mean " << repo[i].mean_value << " max " << repo[i].max_value  << ")" << std::endl;
	  std::cout << "toff " << repo[i].argmax_x*0.01 + i*0.5 << std::endl;
	  std::cout << "alpha " << repo[i].argmax_y + 2300 << std::endl;
	}
    }
  if (!print_all)
    {
      std::cout << "rank " << maxrank << " (mean " << meanmaxrank << " max " << maxmaxrank << ")" << std::endl;
      std::cout << "toff " << argmaxrank_x*0.01 + indexmax*0.5 << std::endl;
      std::cout << "alpha " << argmaxrank_y + 2300 << std::endl;
   }
      
}
