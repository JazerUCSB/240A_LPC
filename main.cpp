#include <iostream>
#include <mlpack/core.hpp>
#include <mlpack/methods/lmnn/lmnn.hpp>
#include <armadillo>
#include <cmath>

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

int main(int argc, char *argv[])
{

  // read in wav file

  if (argc != 2)
  {
    printf("We need a .wav file\n");
    return 1;
  }
  drwav *pWav = nullptr;

  pWav = drwav_open_file(argv[1]);
  if (pWav == nullptr)
  {
    printf("We could not read that .wav file\n");
    return -1;
  }

  // store in float array pSampleData
  float *pSampleData = new float[pWav->totalPCMFrameCount * pWav->channels];
  drwav_read_f32(pWav, pWav->totalPCMFrameCount, pSampleData);
  drwav_close(pWav);

  // length of file
  int sigLength = pWav->totalPCMFrameCount;

  // make a vec to store the signal
  arma::vec origSig = arma::vec(sigLength);

  // write float array into vec
  for (int i = 0; i < sigLength; i++)
  {
    origSig[i] = pSampleData[i];
  }

  // create vec for the output signal
  arma::vec newSig = arma::vec(sigLength);

  // define window size
  // if sample rate is 44100 a window of 300 samples would get us a little below 150hz, right?
  // may need to change this depending where samplerate ends up
  int windowSize = 300;
  int hopSize = widowSize / 2;

  // create vec for the autocorrelation vector. the length is windowSize
  arma::vec coor = arma::vec(windowSize);

  // outer loop for windowing
  for (int bin = 0; bin < (sigLength / hopSize); bin++)
  {
    // inner window loop
    for (int j = 0; j < windowSize; j++)
    {

      // temp vec for windowed signal
      arma::vec tempSig = arma : vec(windowSize);

      // current value with hann window applied
      float curHann = origSig[bin * hopSize + j] * 0.5 * (1 - std::cos(2 * M_PI * j / (windowSize - 1)));

      // write to temp
      tempSig[j] = curHann;
    }

    // perform autocorrelation on the tempSig vector
    for (int lag = 0; lag < windowSize; lag++)
    {
      coor[lag] = arma : accu(tempSig.head(windowSize - lag) % tempSig.tail(windowSize - lag));
    }

    // turn coor into a toeplitz
    arma::mat COOR = arma::toeplitz(coor);

    // make a rowvec out of coor for mlpack
    arma::rowvec coorRow = arma::vec(windowSize);

    // write coor into coorRow
    for (int l = 0; l < windowSize; l++)
    {
      coorRow[l] = coor[l];
    }

    // perform linear regression
    int order = 10;
    mlpack::LinearRegression lr(COOR, coorRow.subvec(1, order));
    coeff = lr.Parameters();

    // find max index of coor
    int maxIndex = 0;
    float maxValue = coor[0];
    for (int m = 1; i < windowSize; m++)
    {
      if (coor[i] > maxValue)
      {
        maxValue = coor[i];
        maxIndex = m;
      }
    }

    // calcate pitch in hz
    float pitch = sampleRate / static_cast<float>(maxIndex);

    // decide on voiced or unvoiced. I have no idea what the threshold value should. Total correlation should be windowSize
    float threshold = 150.0;
    bool voiced = false;
    if (maxValue > threshold)
    {
      voiced = true;
    }

    // if voiced multiply the temp sig by a cos wave
    if (voiced)
    {
      for (int n = 0; n < windowSize; n++)
      {
        tempSig[n] *= std::cos(2.0 * M_PI * pitch * n);
      }
    }

    // apply filter coefficients as an all pole filter
    int filterOrder = coeff.length - 1;

    // create temp output
    arma::vec tempOut = arma::vec(windowSize);

    for (int p = 0; p < windowSize; p++)
    {
      tempOut[p] = coeff(0) * tempSig[p]; // Initialize with the direct term

      for (int r = 1; r <= filterOrder; r++)
      {
        // Check if the index is within bounds before accessing the outputSignal vector
        if (p >= r)
        {
          tempOut[p] += coeff(r) * tempOut(p - r);
        }
      }
    }

    // write tempOut to newSig
    for (int s = 0; s < windowSize; s++)
    {
      newSig[hopSize * bin + s] = tempOut[s];
    }
  }

  // write newSig to file
  drwav_data_format format;
  format.container = drwav_container_riff;
  format.format = DR_WAVE_FORMAT_IEEE_FLOAT;
  format.channels = 1;
  format.sampleRate = 44100;
  format.bitsPerSample = 32;

  pWav = drwav_open_file_write("out.wav", &format);
  for (double d = 0; d < newSig.length; d += 1)
  {
    float f = newSig[d];
    drwav_write(pWav, 1, &f);
  }
  drwav_close(pWav);
}

// Ignore |
//        v

// void hannWindow(float *signal, int length)
// {
//   for (int i = 0; i < length; ++i)
//   {
//     signal[i] *= 0.5 * (1 - std::cos(2 * M_PI * i / (length - 1)));
//   }
// }

// // Function to calculate auto-correlation of a signal
// void autoCorrelation(const arma::vec &signal, arma::vec &result)
// {
//   int length = signal.n_elem;
//   result.set_size(length);

//   for (int lag = 0; lag < length; ++lag)
//   {
//     result[lag] = arma::accu(signal.head(length - lag) % signal.tail(length - lag));
//   }
// }

// // Function to generate filter coefficients using linear prediction
// void linearPrediction(const arma::vec &autoCorr, int order, arma::vec &coefficients)
// {
//   int length = autoCorr.n_elem;
//   arma::mat R = arma::toeplitz(autoCorr.head(order + 1));

//   // Solve the system of linear equations using arma linear solve
//   arma::vec coeffients = solve(R,autoCorr.subvec(1, order));

//   //mlpack::LinearRegression lr(R, autoCorr.subvec(1, order));
//   //coefficients = lr.Parameters();
// }

// int findMaxIndex(const float *array, int length)
// {
//   int maxIndex = 0;
//   float maxValue = array[0];
//   for (int i = 1; i < length; ++i)
//   {
//     if (array[i] > maxValue)
//     {
//       maxValue = array[i];
//       maxIndex = i;
//     }
//   }
//   return maxIndex;
// }

// float calculatePitch(int sampleRate, int maxIndex)
// {
//   // Convert the lag to pitch (in Hz)
//   return sampleRate / static_cast<float>(maxIndex);
// }

// void applyLPCFilter(const arma::vec &inputSignal, const arma::vec &lpcCoefficients, arma::vec &outputSignal)
// {
//   int order = lpcCoefficients.n_elem - 1; // LPC order

//   // Initialize the output signal
//   outputSignal.set_size(inputSignal.n_elem);

//   // Apply the all-pole filter
//   for (size_t i = 0; i < inputSignal.n_elem; ++i)
//   {
//     outputSignal(i) = lpcCoefficients(0) * inputSignal(i); // Initialize with the direct term

//     for (int j = 1; j <= order; ++j)
//     {
//       // Check if the index is within bounds before accessing the outputSignal vector
//       if (i >= j)
//       {
//         outputSignal(i) += lpcCoefficients(j) * outputSignal(i - j);
//       }
//     }
//   }
// }

// int main(int argc, char *argv[])
// {
//   if (argc != 2)
//   {
//     printf("We need a .wav file\n");
//     return 1;
//   }

//   ///////////////////////////////////////////////////////////////////////////
//   /// Reading in a .WAV file ////////////////////////////////////////////////
//   ///////////////////////////////////////////////////////////////////////////

//   drwav *pWav = nullptr;

//   pWav = drwav_open_file(argv[1]);
//   if (pWav == nullptr)
//   {
//     printf("We could not read that .wav file\n");
//     return -1;
//   }

//   float *pSampleData = new float[pWav->totalPCMFrameCount * pWav->channels];
//   drwav_read_f32(pWav, pWav->totalPCMFrameCount, pSampleData);
//   drwav_close(pWav);

//   printf("The first sample is %f\n", pSampleData[0]);

//   //float outWindow[]

//       float autoCorr[pWav->totalPCMFrameCount];
//   for (int lag = 0; lag < pWav->totalPCMFrameCount; ++lag)
//   {
//     autoCorr[lag] = 0.0;
//     for (int i = 0; i < pWav->totalPCMFrameCount - lag; ++i)
//     {
//       autoCorr[lag] += pSampleData[i] * pSampleData[i + lag];
//     }
//   }
//   ///////////////////////////////////////////////////////////////////////////
//   /// Doing a linear solve //////////////////////////////////////////////////
//   ///////////////////////////////////////////////////////////////////////////

//   using namespace arma; // arma::vec, arma::mat, arma::solve_opts::fast

//   mat A(5, 5, fill::randu);
//   vec b(5, fill::randu);
//   mat B(5, 5, fill::randu);

//   // several ways to do the solve given the system above...
//   //

//   // #1 ~ return a vector
//   vec x1 = solve(A, b);

//   // #2 ~ return success or failure
//   vec x2; // result vector
//   bool status = solve(x2, A, b);

//   // #3 ~ return a matrix
//   mat X1 = solve(A, B);

//   // #4 ~ enable fast mode
//   mat X2 = solve(A, B, solve_opts::fast);

//   // #5 ~ indicate that A is triangular
//   mat X3 = solve(trimatu(A), B);

//   ///////////////////////////////////////////////////////////////////////////
//   /// Writing out a .WAV file ///////////////////////////////////////////////
//   ///////////////////////////////////////////////////////////////////////////

//   drwav_data_format format;
//   format.container = drwav_container_riff;
//   format.format = DR_WAVE_FORMAT_IEEE_FLOAT;
//   format.channels = 1;
//   format.sampleRate = 44100;
//   format.bitsPerSample = 32;

//   pWav = drwav_open_file_write("out.wav", &format);
//   for (double d = 0; d < 10000; d += 1)
//   {
//     float f = std::cos(d);
//     drwav_write(pWav, 1, &f);
//   }
//   drwav_close(pWav);
// }

// void constantOverlapAdd(float *inputSignal, int inputLength, float *outputSignal, int windowSize, int hopSize)
// {
//   // Iterate over the input signal with overlapping windows
//   for (int i = 0; i < inputLength - windowSize; i += hopSize)
//   {
//     // Apply Hann window to the current window
//     hannWindow(inputSignal + i, windowSize);

//     // Add the windowed portion to the output signal
//     for (int j = 0; j < windowSize; ++j)
//     {
//       outputSignal[i + j] += inputSignal[i + j];
//     }
//   }
// }

// void colaAutoCorrelation(float *signal, int signalLength, int frameSize, int hopSize)
// {
//   // Initialize buffer for overlap-add
//   int bufferSize = frameSize + hopSize;
//   float buffer[bufferSize];

//   for(int i = 0;i<bufferSize;i++){
//       buffer[i]=0.;

//   }
// Process overlapping frames
//   for (int i = 0; i < signalLength - frameSize; i += hopSize)
//   {
//     // Copy samples to buffer with overlap
//     for (int j = 0; j < frameSize; ++j)
//     {
//       buffer[j] = buffer[j + hopSize];
//       buffer[j + hopSize] = signal[i + j];
//     }

//     // Apply Hann window to the buffer
//     hannWindow(buffer, bufferSize);

//     float autoCorr[signalLength];
//     for (int lag = 0; lag < signalLength; ++lag)
//     {
//       autoCorr[lag] = 0.0;
//       for (int i = 0; i < signalLength - lag; ++i)
//       {
//         autoCorr[lag] += inputSignal[i] * inputSignal[i + lag];
//       }
//     }
//   }
//   float maxI = findMaxIndex(autoCoor, signalLength);
//   float pitch = calculatePitch(44100, maxI);
//   return pitch;
// }