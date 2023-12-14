#include <iostream>
#include <mlpack/core.hpp>
#include <mlpack/methods/lmnn/lmnn.hpp>
#include <armadillo>
#include <cmath>

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

#define SAMPLERATE (48000)
#define DOWN_SAMPLERATE (8000)
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
    //}
  }

  // create vec for the output signal
  arma::vec newSig = arma::vec(sigLength);

  // define window size
  int windowSize = 50;
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

      chunk[j] = curHann;
    }

    // perform autocorrelation on the chunk vector
    for (int lag = 1; lag < windowSize; lag++)
    {
      corr[lag] = arma::accu(chunk.head(windowSize - lag) % chunk.tail(windowSize - lag));
    }

    // find max index of corr
    int maxIndex = 0;
    float maxValue = 0;
    for (int t = 0; t < windowSize; t++)
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
    // for (int u = 0; u < windowSize; u++)
    // {

    //   corr[u] /= maxValue;
    // }
    // std::cout << maxValue << std::endl;
    //   create rowvec for mlpack
    arma::rowvec corrRow = arma::rowvec(windowSize);
    for (int h = 0; h < windowSize; h++)
    {
      corrRow[h] = corr[h];
    }

    // turn corr into a toeplitz
    arma::mat CORR = arma::toeplitz(corr);
    // float de = arma::det(CORR);
    // std::cout << "de=" << de << std::endl;
    // solve for the filter coefficients
    mlpack::LinearRegression lr(CORR, corrRow);
    auto coeff = lr.Parameters();
    // arma::vec coeff = arma::solve(CORR, corr);
    //   std::cout << coeff[12] << std::endl;

    // decide on voiced or unvoiced.
    float threshold = .002;

    // if voiced make a pulse train
    if (maxValue > threshold)
    {
      for (int a = 0; a < windowSize; a++)
      {
        int rate = SAMPLERATE / pitch;
        float val = a % rate < 2;
        chunk[a] = val; // std::abs(cos(M_PI * pitch * a / windowSize));
      }
    }
    // if unvoiced randu
    else if (maxValue <= threshold)
    {
      for (int w = 0; w < windowSize; w++)
      {
        chunk[w] = (arma::randu() * 2.0 - 1.0) * 0.25;
      }
    }

    //   apply filter coefficients as an all pole filter
    int filterOrder = 10; // coeff.size() - 1;

    // create temp output
    arma::vec newChunk = arma::vec(windowSize);

    for (int p = 0; p < windowSize; p++)
    {
      newChunk[p] = coeff[0] * chunk[p]; // Initialize with the direct term

      for (int r = 1; r <= filterOrder; r++)
      {
        if (p > r)
        {
          newChunk[p] += coeff[r] * chunk[p - r];
        }
      }
      // std::cout << chunk[p] << std::endl;
      // std::cout << newChunk[p] << std::endl;
    }

    // write newChunk to newSig
    for (int s = 0; s < windowSize; s++)
    {
      float curHann = newChunk[s] * 0.5 * (1 - std::cos(2 * M_PI * s / (windowSize - 1)));
      newSig[hopSize * bin + s] += curHann;
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
