#include <iostream>
#include <armadillo>
#include <mlpack/core.hpp>
#include <mlpack/methods/lmnn/lmnn.hpp>
#include <cmath>
#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

void hannWindow(float *signal, int length)
{
  for (int i = 0; i < length; ++i)
  {
    signal[i] *= 0.5 * (1 - std::cos(2 * M_PI * i / (length - 1)));
  }
}

void constantOverlapAdd(float *inputSignal, int inputLength, float *outputSignal, int windowSize, int hopSize)
{
  // Iterate over the input signal with overlapping windows
  for (int i = 0; i < inputLength - windowSize; i += hopSize)
  {
    // Apply Hann window to the current window
    hannWindow(inputSignal + i, windowSize);

    // Add the windowed portion to the output signal
    for (int j = 0; j < windowSize; ++j)
    {
      outputSignal[i + j] += inputSignal[i + j];
    }
  }
}

void colaAutoCorrelation(float *signal, int signalLength, int frameSize, int hopSize)
{
  // Initialize buffer for overlap-add
  int bufferSize = frameSize + hopSize;
  float buffer[bufferSize] = {0};

  // Process overlapping frames
  for (int i = 0; i < signalLength - frameSize; i += hopSize)
  {
    // Copy samples to buffer with overlap
    for (int j = 0; j < frameSize; ++j)
    {
      buffer[j] = buffer[j + hopSize];
      buffer[j + hopSize] = signal[i + j];
    }

    // Apply Hann window to the buffer
    hannWindow(buffer, bufferSize);

    float autoCorr[signalLength];
    for (int lag = 0; lag < signalLength; ++lag)
    {
      autoCorr[lag] = 0.0;
      for (int i = 0; i < signalLength - lag; ++i)
      {
        autoCorr[lag] += inputSignal[i] * inputSignal[i + lag];
      }
    }
  }
  float maxI = findMaxIndex(autoCoor, signalLength);
  float pitch = calculatePitch(44100, maxI);
  return pitch;
}

int findMaxIndex(const float *array, int length)
{
  int maxIndex = 0;
  float maxValue = array[0];
  for (int i = 1; i < length; ++i)
  {
    if (array[i] > maxValue)
    {
      maxValue = array[i];
      maxIndex = i;
    }
  }
  return maxIndex;
}

float calculatePitch(int sampleRate, int maxIndex)
{
  // Convert the lag to pitch (in Hz)
  return sampleRate / static_cast<float>(maxIndex);
}

int main(int argc, char *argv[])
{
  if (argc != 2)
  {
    printf("We need a .wav file\n");
    return 1;
  }

  ///////////////////////////////////////////////////////////////////////////
  /// Reading in a .WAV file ////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////

  drwav *pWav = nullptr;

  pWav = drwav_open_file(argv[1]);
  if (pWav == nullptr)
  {
    printf("We could not read that .wav file\n");
    return -1;
  }

  float *pSampleData = new float[pWav->totalPCMFrameCount * pWav->channels];
  drwav_read_f32(pWav, pWav->totalPCMFrameCount, pSampleData);
  drwav_close(pWav);

  printf("The first sample is %f\n", pSampleData[0]);

  float outWindow[]

      float autoCorr[totalPCMFrameCount];
  for (int lag = 0; lag < totalPCMFrameCount; ++lag)
  {
    autoCorr[lag] = 0.0;
    for (int i = 0; i < totalPCMFrameCount - lag; ++i)
    {
      autoCorr[lag] += pSampleData[i] * pSampleData[i + lag];
    }
  }
  ///////////////////////////////////////////////////////////////////////////
  /// Doing a linear solve //////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////

  using namespace arma; // arma::vec, arma::mat, arma::solve_opts::fast

  mat A(5, 5, fill::randu);
  vec b(5, fill::randu);
  mat B(5, 5, fill::randu);

  // several ways to do the solve given the system above...
  //

  // #1 ~ return a vector
  vec x1 = solve(A, b);

  // #2 ~ return success or failure
  vec x2; // result vector
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
  for (double d = 0; d < 10000; d += 1)
  {
    float f = std::cos(d);
    drwav_write(pWav, 1, &f);
  }
  drwav_close(pWav);
}

// Function to calculate auto-correlation of a signal
void autoCorrelation(const arma::vec &signal, arma::vec &result)
{
  int length = signal.n_elem;
  result.set_size(length);

  for (int lag = 0; lag < length; ++lag)
  {
    result(lag) = arma::accu(signal.head(length - lag) % signal.tail(length - lag));
  }
}

// Function to generate filter coefficients using linear prediction
void linearPrediction(const arma::vec &autoCorr, int order, arma::vec &coefficients)
{
  int length = autoCorr.n_elem;
  arma::mat R = arma::toeplitz(autoCorr.head(order + 1));

  // Solve the system of linear equations using mlpack's linear regression
  mlpack::regression::LinearRegression lr(R, autoCorr.subvec(1, order));
  coefficients = lr.Parameters();
}

// int main() {
//     // Example parameters
//     const int order = 10;  // LPC order

//     arma::vec signal;  // Replace this with your actual signal data
//     // ...

//     // Apply Hann window to the signal (if necessary)
//     // ...

//     // Calculate auto-correlation
//     arma::vec autoCorr(order + 1);
//     autoCorrelation(signal, autoCorr);

//     // Generate filter coefficients using linear prediction
//     arma::vec coefficients;
//     linearPrediction(autoCorr, order, coefficients);

//     // Print the coefficients (replace this with your actual usage)
//     std::cout << "Filter Coefficients:" << std::endl;
//     std::cout << coefficients.t() << std::endl;

//     return 0;
// }

void applyLPCFilter(const arma::vec &inputSignal, const arma::vec &lpcCoefficients, arma::vec &outputSignal)
{
  int order = lpcCoefficients.n_elem - 1; // LPC order

  // Initialize the output signal
  outputSignal.set_size(inputSignal.n_elem);

  // Apply the all-pole filter
  for (size_t i = 0; i < inputSignal.n_elem; ++i)
  {
    outputSignal(i) = inputSignal(i);
    for (int j = 1; j <= order; ++j)
    {
      outputSignal(i) -= lpcCoefficients(j) * outputSignal(i - j);
    }
  }
}