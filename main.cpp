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

  drwav_close(pWav);

  // make a vec to store the signal
  arma::vec origSig = arma::vec(sigLength);

  // write float array into vec
  for (int i = 0; i < sigLength; i++)
  {

    origSig[i] = pSampleData[i];
    // std::cout << origSig[i] << std::endl;
  }

  // create vec for the output signal
  arma::vec newSig = arma::vec(sigLength);

  // define window size
  // if sample rate is 44100 a window of 300 samples would get us a little below 150hz, right?
  // may need to change this depending where samplerate ends up
  int windowSize = 200;
  int hopSize = windowSize / 2;

  // create vec for the autocorrelation vector. the length is windowSize
  arma::vec corr = arma::vec(windowSize);

  // outer loop for windowing
  for (int bin = 0; bin < (sigLength / hopSize); bin++)
  {
    // temp vec for windowed signal
    arma::vec chunk = arma ::vec(windowSize);

    // inner window loop
    for (int j = 0; j < windowSize; j++)
    {

      // current value with hann window applied
      float curHann = origSig[bin * hopSize + j] * 0.5 * (1 - std::cos(2 * M_PI * j / (windowSize - 1)));

      // write to temp
      chunk[j] = curHann;
    }

    // perform autocorrelation on the chunk vector
    for (int s = 0; s < windowSize; s++)
    {
      float thisCorr = 0;
      for (int lag = 1; lag < windowSize; lag++)
      {
        if ((s + lag) < windowSize)
        {
          thisCorr += chunk[s] * chunk[s + lag];
        }
      }
      corr[s] = thisCorr;
    }

    // turn corr into a toeplitz
    arma::mat CORR = arma::toeplitz(corr);

    // solve for the filter coefficients
    arma::vec coeff = solve(CORR, corr);

    // find max index of corr
    int maxIndex = 0;
    float maxValue = corr[0];
    for (int m = 1; m < windowSize; m++)
    {
      if (corr[m] > maxValue)
      {
        maxValue = corr[m];
        maxIndex = m;
      }
    }

    // calcate pitch in hz
    float pitch = SAMPLERATE / (static_cast<float>(maxIndex));
    // std::cout << pitch;
    std::cout << maxValue << std::endl;
    //   decide on voiced or unvoiced. I have no idea what the threshold value should. Total correlation should be windowSize
    float threshold = 0.6;
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
        chunk[n] *= std::cos(2.0 * M_PI * pitch * n * (windowSize / SAMPLERATE));
      }
    }

    // apply filter coefficients as an all pole filter
    int filterOrder = coeff.size() - 1;

    // create temp output
    arma::vec tempOut = arma::vec(windowSize);

    for (int p = 0; p < windowSize; p++)
    {
      tempOut[p] = coeff[0] * chunk[p]; // Initialize with the direct term

      for (int r = 1; r <= filterOrder; r++)
      {
        // Check if the index is within bounds before accessing the outputSignal vector
        if (p >= r)
        {
          tempOut[p] += coeff[r] * tempOut[p - r];
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
