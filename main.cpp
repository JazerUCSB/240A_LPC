#include <iostream>
#include <mlpack.hpp>

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("We need a .wav file\n");
    return 1;
  }

  ///////////////////////////////////////////////////////////////////////////
  /// Reading in a .WAV file ////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////

  drwav* pWav = nullptr;

  pWav = drwav_open_file(argv[1]);
  if (pWav == nullptr) {
    printf("We could not read that .wav file\n");
    return -1;
  }

  float* pSampleData = new float[pWav->totalPCMFrameCount * pWav->channels];
  drwav_read_f32(pWav, pWav->totalPCMFrameCount, pSampleData);
  drwav_close(pWav);

  printf("The first sample is %f\n", pSampleData[0]);

  ///////////////////////////////////////////////////////////////////////////
  /// Doing a linear solve //////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////

  using namespace arma;  // arma::vec, arma::mat, arma::solve_opts::fast

  mat A(5, 5, fill::randu);
  vec b(5, fill::randu);
  mat B(5, 5, fill::randu);

  // several ways to do the solve given the system above...
  //

  // #1 ~ return a vector
  vec x1 = solve(A, b);

  // #2 ~ return success or failure
  vec x2;  // result vector
  bool status = solve(x2, A, b);

  // #3 ~ return a matrix
  mat X1 = solve(A, B);

  // #4 ~ enable fast mode
  mat X2 = solve(A, B, solve_opts::fast);

  // #5 ~ indicate that A is triangular
  mat X3 = solve(trimatu(A), B);

  ///////////////////////////////////////////////////////////////////////////
  /// Writing out a .WAV file ///////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////

  drwav_data_format format;
  format.container = drwav_container_riff;
  format.format = DR_WAVE_FORMAT_IEEE_FLOAT;
  format.channels = 1;
  format.sampleRate = 44100;
  format.bitsPerSample = 32;

  pWav = drwav_open_file_write("out.wav", &format);
  for (double d = 0; d < 10000; d += 1) {
    float f = std::cos(d);
    drwav_write(pWav, 1, &f);
  }
  drwav_close(pWav);
}
