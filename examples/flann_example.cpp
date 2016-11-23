
#include <flann/flann.h>

#include <stdio.h>
#include <stdlib.h>
#include <chrono>

unsigned char* read_points(const char* filename, int rows, int cols) {
  unsigned char * data;
  unsigned char *p;
  FILE* fin;

  fin = fopen(filename, "r");
  if (!fin) {
    printf("Cannot open input file.\n");
    exit(1);
  }

  data = (unsigned char*) malloc(rows * cols * sizeof(unsigned char));
  if (!data) {
    printf("Cannot allocate memory.\n");
    exit(1);
  }
  p = data;
  size_t count = 0;
  while(count < rows*cols){
    count += fread(p+count, sizeof(unsigned char), rows*cols - count, fin);
  }

  fclose(fin);

  return data;
}

void write_results(const char* filename, int *data, int rows, int cols) {
  FILE* fout;
  int* p;
  int i, j;

  fout = fopen(filename, "w");
  if (!fout) {
    printf("Cannot open output file.\n");
    exit(1);
  }

  p = data;
  for (i = 0; i < rows; ++i) {
    for (j = 0; j < cols; ++j) {
      fprintf(fout, "%d ", *p);
      p++;
    }
    fprintf(fout, "\n");
  }
  fclose(fout);
}

int main(int argc, char** argv) {
  unsigned char* dataset;
  unsigned char* testset;
  int nn;
  int* result;
  float* dists;
  struct FLANNParameters p;
  float speedup;
  flann_index_t index_id;

  int rows = 8000000;
  int cols = 128;
  int tcount = 1000000;

  /*
   * The files dataset.dat and testset.dat can be downloaded from:
   * http://people.cs.ubc.ca/~mariusm/uploads/FLANN/datasets/dataset.dat
   * http://people.cs.ubc.ca/~mariusm/uploads/FLANN/datasets/testset.dat
   */
  printf("Reading input data file.\n");
  dataset = read_points(argv[1], rows, cols);
  printf("Reading test data file.\n");
  testset = read_points(argv[2], tcount, cols);

  nn = 3;
  result = (int*) malloc(tcount * nn * sizeof(int));
  dists = (float*) malloc(tcount * nn * sizeof(float));

  p = DEFAULT_FLANN_PARAMETERS;
  p.algorithm = FLANN_INDEX_KDTREE;
  p.trees = 8;
  p.log_level = FLANN_LOG_INFO;
  p.checks = 64;
  p.cores = 8;

  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();
  printf("Computing index.\n");
  index_id = flann_build_index_byte(dataset, rows, cols, &speedup, &p);

  printf("done building the forest\n");
  end = std::chrono::system_clock::now();
  std::cout << "building the trees took " << std::chrono::duration_cast<std::chrono::seconds>(end-start).count() << " seconds" << std::endl;
  flann_find_nearest_neighbors_index_byte(index_id, testset, tcount, result, dists,
      nn, &p);

  write_results("results.dat", result, tcount, nn);

  flann_free_index(index_id, &p);
  free(dataset);
  free(testset);
  free(result);
  free(dists);

  return 0;
}
