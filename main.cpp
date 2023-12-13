#include <iostream>
#include <mlpack/core.hpp>
#include <mlpack/methods/lmnn/lmnn.hpp>
#include <armadillo>
#include <cmath>

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

#define SAMPLERATE (48000)

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

  // length of file
  int sigLength = pWav->totalPCMFrameCount;
  int decLength = sigLength / 6;
  drwav_close(pWav);

  // make a vec to store the signal
  arma::vec origSig = arma::vec(sigLength);

  // write float array into vec
  for (int i = 0; i < sigLength; i++)
  {
    // if (i % 6 == 0)
    // {
    origSig[i] = pSampleData[i];
    // }
    // std::cout << origSig[i] << std::endl;
  }

  // create vec for the output signal
  arma::vec newSig = arma::vec(sigLength);

  // define window size
  // if sample rate is 44100 a window of 300 samples would get us a little below 150hz, right?
  // may need to change this depending where samplerate ends up
  int windowSize = 100;
  int hopSize = windowSize / 2;
  int bins = sigLength / hopSize;
  // create vec for the autocorrelation vector. the length is windowSize
  arma::vec corr = arma::vec(windowSize);

  // outer loop for windowing
  for (int bin = 0; bin < bins; bin++)
  {
    // temp vec for windowed signal
    arma::vec chunk = arma::vec(windowSize);

    // inner window loop
    for (int j = 0; j < windowSize; j++)
    {

      // current value with hann window applied
      float curHann = origSig[bin * hopSize + j] * 0.5 * (1 - std::cos(2 * M_PI * j / (windowSize - 1)));
      // std::cout << curHann << std::endl;
      // std::cout << origSig[bin * hopSize + j] << std::endl;
      //   write to temp
      chunk[j] = curHann;
    }

    // perform autocorrelation on the chunk vector
    for (int s = 0; s < windowSize; s++)
    {
      float thisCorr = 0;
      for (int lag = 1; lag <= windowSize; lag++)
      {
        if ((s - lag) >= 0)
        {
          thisCorr += chunk[s] * chunk[s - lag];
        }
      }
      corr[s] = thisCorr;
      // std::cout << corr[s] << std::endl;
    }

    // find max index of corr
    int maxIndex = 0;
    float maxValue = corr[0];
    for (int t = 1; t < windowSize; t++)
    {
      if (corr[t] > maxValue)

      {
        maxValue = corr[t];
        maxIndex = t;
      }
    }

    // calculate pitch in hz
    double pitch = SAMPLERATE / (static_cast<float>(maxIndex));

    // normalize corr
    for (int u = 0; u < windowSize; u++)
    {

      corr[u] /= maxValue;
    }

    // turn corr into a toeplitz
    arma::mat CORR = arma::toeplitz(corr);
    arma::rowvec corrRow = arma::rowvec(windowSize);
    // arma::mat invCORR = arma::pinv(CORR);
    for (int h = 0; h < windowSize; h++)
    {
      corrRow[h] = corr[h];
    }

    // solve for the filter coefficients
    mlpack::LinearRegression lr(CORR, corrRow);
    auto coeff = lr.Parameters();

    // arma::vec coeff = arma::solve(CORR, corr);
    // arma::vec coeff = invCORR * corr;
    // calculate pitch in hz
    double pitch = SAMPLERATE / (static_cast<float>(maxIndex));
    // std::cout << pitch << std::endl;
    // std::cout << maxValue << std::endl;
    //       decide on voiced or unvoiced. I have no idea what the threshold value should. Total correlation should be windowSize
    float threshold = .01;

    if (maxValue > threshold)
    {
      for (int a = 0; a < windowSize; a++)
      {
        // int rate = SAMPLERATE / pitch;
        // int peak = (a % rate) < 4;
        // chunk[a] = peak;
        double wave = M_PI * static_cast<double>(pitch) * static_cast<double>(a) / (static_cast<double>(SAMPLERATE));
        chunk[a] = std::abs(std::sin(wave));
      }
    }
    // if unvoiced randu
    else if (maxValue <= threshold)
    {
      for (int w = 0; w < windowSize; w++)
      {
        chunk[w] = (arma::randu() * 2.0 - 1.0) * 0.5;
      }
    }
    // std::cout << voiced << std::endl;
    //   apply filter coefficients as an all pole filter
    int filterOrder = coeff.size() - 1;

    // create temp output
    arma::vec newChunk = arma::vec(windowSize);

    for (int p = 0; p < windowSize; p++)
    {
      newChunk[p] = coeff[0] * chunk[p]; // Initialize with the direct term
      // std::cout << chunk[p] << std::endl;
      for (int r = 1; r <= filterOrder; r++)
      {
        if (p > r)
        {
          newChunk[p] += coeff[r] * chunk[p - r];
        }
      }
      // std::cout << newChunk[p] << std::endl;
      // std::cout << chunk[p] << std::endl;
      // std::cout << coeff[p] << std::endl;
    }

    // write newChunk to newSig
    for (int s = 0; s < windowSize; s++)
    {
      float curHann = newChunk[s] * 0.5 * (1 - std::cos(2 * M_PI * s / (windowSize - 1)));
      newSig[hopSize * bin + s] += curHann;
      // std::cout << newChunk[s] << std::endl;
    }
  }

  // write newSig to file
  drwav_data_format format;
  format.container = drwav_container_riff;
  format.format = DR_WAVE_FORMAT_IEEE_FLOAT;
  format.channels = 1;
  format.sampleRate = SAMPLERATE;
  format.bitsPerSample = 32;

  pWav = drwav_open_file_write("out.wav", &format);
  for (int d = 0; d < newSig.size(); d++)
  {
    float f = newSig[d];
    drwav_write(pWav, 1, &f);
  }
  drwav_close(pWav);
}
