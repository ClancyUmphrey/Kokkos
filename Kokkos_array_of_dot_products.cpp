/*
  Example of using the hardware agnostic Kokkos C++ library to do a simple dot product
  operation in parallel for an array of vectors.
*/

// Clancy Umphrey

// Reference:
// https://github.com/kokkos/kokkos-tutorials/tree/master/Intro-Full/SNL2015/Exercises/06_10_ArrayOfDotProducts 

#include<Kokkos_Core.hpp>
#include<cmath>
#include<cstdio>
#include<cstdlib>
#include<cstring>
#include<sys/time.h>

// Function Prototypes
//----------------------------------------------------------------------------------------
void parse_arguments(int argc, char* argv[], int &num_vectors, int &len, int &nrepeat);

template <typename ViewType1D>
void print_results(int num_vectors, int len, int nrepeat, ViewType1D h_c, double time);

timeval start_timer();

double time_lapse(timeval start);

//========================================================================================
// Choose Kokkos-View Memory Layout 
//========================================================================================
/*
  Kokkos Views are multidimensional arrays that are polymorphic, i.e., their layout in
  memory can be changed at compile time.
    LayoutRight: Right-most index is stride 1, "row-major" in 2D.
                 Optimal for caching on the CPU.
    LayoutLeft:  Left-most index is stride 1, "column-major" in 2D.
                 Optimal for coalescing on the GPU.
*/
#define LAYOUT LayoutRight // LayoutRight OR LayoutLeft 


//========================================================================================
// Kokkos Functor 
//========================================================================================
/*
  Kokkos parallel patterns (i.e., for, reduce, scan) require either a functor or a lambda
  function to execute.  Below is an example of a functor that computes an array of dot
  products.
*/
template <typename ViewType2D, typename ViewType1D>
struct dotProduct_Functor {

  ViewType2D a;
  ViewType2D b;
  ViewType1D c;
  
  int len;
  
  dotProduct_Functor(const ViewType2D &a, const ViewType2D &b, const ViewType1D &c,
                     int len): a(a), b(b), c(c), len(len) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const size_t i) const {

      double ctmp = 0.0;
      for(int j = 0; j < len; j++) {
        ctmp += a(i,j) * b(i,j);
      }
      c(i) = ctmp;
  }
};


//========================================================================================
// MAIN
//========================================================================================
int main(int argc, char* argv[]) {

  // Set default parameters
  int num_vectors = 1000;  // number of vectors
  int len         = 10000; // length of vectors 
  int nrepeat     = 10;    // number of repeats of the test

  parse_arguments(argc, argv, num_vectors, len, nrepeat); 

  Kokkos::initialize(argc, argv);

  /*
    2D and 1D View types are created here with the layout defined by the LAYOUT macro at
    the top of this file.
  */
  typedef Kokkos::View<double**,Kokkos::LAYOUT> View2D;
  typedef Kokkos::View<double* ,Kokkos::LAYOUT> View1D;

  // Allocate space for vectors to do num_vectors dot products of length len
  View2D a("A",num_vectors,len);
  View2D b("B",num_vectors,len);
  View1D c("C",num_vectors);

  /*
    Create a mirror view for the host.  Memory is only allocated if this code was compiled
    for the GPU, meaning c is not on the host already.
  */
  auto h_c = Kokkos::create_mirror_view(c);
  
  /*
    This is an example of simply using a lambda function instead of a functor to
    initialize the vectors.
  */
  Kokkos::parallel_for( num_vectors, KOKKOS_LAMBDA (const int& i) {
    for(int j = 0; j < len; j++) {
      a(i,j) = i+1;
      b(i,j) = j+1;
    }
    c(i) = 0.0;
  });


  // Time the dot products
  timeval start = start_timer();

  for(int repeat = 0; repeat < nrepeat; repeat++) {

    // Compute the dot products nrepeat times
    Kokkos::parallel_for( num_vectors, dotProduct_Functor<View2D, View1D>(a, b, c, len) );

  }

  double time = time_lapse(start);

  /*
    Here is a deep copy of the result from the device to the host (only if this code was
    compiled for the GPU, meaning h_c is not just pointing to c which is on the host
    already).  A deep copy is never hidden in Kokkos and must be explicitly called since
    it can be an expensive operation.
  */
  Kokkos::deep_copy(h_c, c);

  // Print results (time (s), problem size (MB), and bandwidth (GB/s))
  print_results(num_vectors, len, nrepeat, h_c, time);

  Kokkos::finalize();
}


//========================================================================================
// Helper code I took from the Kokkos tutorials and wrapped into functions
//========================================================================================

// Reference:
// https://github.com/kokkos/kokkos-tutorials/tree/master/Intro-Full/SNL2015/Exercises/06_10_ArrayOfDotProducts

// Parse the command line arguments
//----------------------------------------------------------------------------------------
void parse_arguments(int argc, char* argv[], int &num_vectors, int &len, int &nrepeat) {

  for(int i=0; i<argc; i++) {
    if( (strcmp(argv[i], "-v") == 0) || (strcmp(argv[i], "-num_vectors") == 0)) {
      num_vectors = atoi(argv[++i]);
    } else if( (strcmp(argv[i], "-l") == 0) || (strcmp(argv[i], "-v") == 0)) {
      len = atof(argv[++i]);
    } else if( strcmp(argv[i], "-nrepeat") == 0) {
      nrepeat = atoi(argv[++i]);
    } else if( (strcmp(argv[i], "-h") == 0) || (strcmp(argv[i], "-help") == 0)) {
      printf("ArrayOfDotProducts Options:\n");
      printf("  -num_vectors (-v)  <int>: number of vectors (default: 1000)\n");
      printf("  -length (-l) <int>:       vector length (default: 10000)\n");
      printf("  -nrepeat <int>:           number of repitions (default: 10)\n");
      printf("  -help (-h):               print this message\n");
    }
  }
}


// Check for errors and print results
//----------------------------------------------------------------------------------------
template <typename ViewType>
void print_results(int num_vectors, int len, int nrepeat, ViewType h_c, double time) {

  // Check for errors in the calculations
  int error = 0;
  for(int i = 0; i < num_vectors; i++) {
    double diff = ((h_c(i) - 1.0*(i+1)*len*(len+1)/2))/((i+1)*len*(len+1)/2);
    if ( diff*diff>1e-20 ) { 
      error = 1;
      printf("Error: %i %i %i %lf %lf %e %lf\n",i,num_vectors,len,h_c(i),
             1.0*(i+1)*len*(len+1)/2,h_c(i) - 1.0*(i+1)*len*(len+1)/2,diff);
    }
  }

  // Print the problem size, time and bandwidth in GB/s
  if(error==0) { 
    printf("#NumVector Length Time(s) ProblemSize(MB) Bandwidth(GB/s)\n");
    printf("%i %i %e %lf %lf\n",num_vectors,len,time,1.0e-6*num_vectors*len*2*8,
           1.0e-9*num_vectors*len*2*8*nrepeat/time);
  }
  else printf("Error\n");
}


// Get the start time 
//----------------------------------------------------------------------------------------
timeval start_timer() {

  timeval start;
  gettimeofday(&start,NULL);

  return start;
}


// Return the time (s) since start 
//----------------------------------------------------------------------------------------
double time_lapse(timeval start) {

  timeval end;
  gettimeofday(&end,NULL);

  // Calculate time
  double time = 1.0*(end.tv_sec-start.tv_sec) + 1.0e-6*(end.tv_usec-start.tv_usec);

  return time;
}


// END OF FILE ---------------------------------------------------------------------------
