#include "Fourier.hpp"
#include <omp.h>

std::vector<FourierSeries::FourierElements> FourierSeries::Data;

void FourierSeries::DFT(const std::vector<Complex> &Input, const size_t TiD)
{
if (TiD == 0)
std::cout << "Starting DFT computation" << std::endl;
for (size_t k = TiD; k < Input.size(); k += Params.Simulator.NumThreads)
{
Complex Sum(0, 0); 
for (size_t n = 0; n < Input.size(); n++)
{
double Phi = (2 * M_PI * k * n) / Input.size(); 
const Complex C(std::cos(Phi), -std::sin(Phi)); 
Sum += Input[n] * C;                            
}
Sum /= Input.size();
const double Freq = -double(k);
const double Phase = Sum.Angle();
const double Amp = Sum.Size();
#pragma omp critical
{
Data.push_back(FourierElements(Sum, Freq, Phase, Amp));
}
}
#pragma omp barrier
if (TiD == 0)
std::cout << "Finished DFT computation" << std::endl;
}

void FourierSeries::Sort(const size_t TiD)
{
if (TiD == 0)
{
std::sort(Data.begin(), Data.end(), FourierCmp());
std::reverse(Data.begin(), Data.end());
}
#pragma omp barrier
}

void FourierSeries::BubbleSort()
{
FourierElements Tmp;
for (int i = 0; i < Data.size(); i++)
{ 
for (int j = Data.size() - 1; j > i; j--)
{ 
if (Data[j].Amplitude > Data[j - 1].Amplitude)
{
Tmp = Data[j - 1];
Data[j - 1] = Data[j];
Data[j] = Tmp;
}
}
}
}