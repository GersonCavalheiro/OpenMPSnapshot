#include "Neighbourhood.hpp"
#include "Vec.hpp"
#include <omp.h>

NLayout::Layout NLayout::UsingLayout = NLayout::Invalid;
std::vector<Boid> NLayout::BoidsGlobal;
std::unordered_map<size_t, NLayout::FlockData> NLayout::BoidsGlobalData;

void NLayout::SetType(const Layout L)
{
assert(L == Local || L == Global);
UsingLayout = L;
BoidsGlobal.reserve(GlobalParams.SimulatorParams.NumBoids);
}

NLayout::Layout NLayout::GetType()
{
return UsingLayout;
}

bool NLayout::IsValid() const
{
if (UsingLayout == Invalid)
return false;
for (size_t i = 0; i < BoidsGlobal.size(); i++)
{
if (BoidsGlobal[i].BoidID != i)
return false;
}
if (BoidsGlobal.size() > 0)
{
for (size_t bID : BoidsGlobalData[FlockID].BoidIDs)
{
if (BoidsGlobal[bID].FlockID != FlockID)
return false;
}
}
if (FlockID == 0) 
{
size_t NumBoids = 0;
for (auto FD : BoidsGlobalData)
{
NumBoids += FD.second.Size();
}
if (NumBoids != BoidsGlobal.size())
return false;
}
for (const Boid &B : BoidsLocal)
{
if (B.FlockID != FlockID)
return false;
if (!B.IsValid())
return false;
}
return true;
}

void NLayout::NewBoid(Flock *FP, const size_t FID)
{
Boid NewBoidStruct(FID);
assert(NewBoidStruct.IsValid());
FlockID = FID;
if (UsingLayout == Local)
{
BoidsLocal.push_back(NewBoidStruct);
}
else
{
assert(UsingLayout == Global);
BoidsGlobal.push_back(NewBoidStruct);
BoidsGlobalData[FlockID].Add(NewBoidStruct);
}
assert(IsValid());
}

size_t NLayout::Size() const
{
if (UsingLayout == Local)
{
return BoidsLocal.size();
}
assert(UsingLayout == Global);
if (BoidsGlobal.size() == 0)
return 0;
return BoidsGlobalData[FlockID].Size();
}

std::vector<Boid *> NLayout::GetBoids() const
{
assert(IsValid());
if (UsingLayout == Local)
{
std::vector<Boid *> LocalFlock;
LocalFlock.reserve(BoidsLocal.size());
for (const Boid &B : BoidsLocal)
{
LocalFlock.push_back(const_cast<Boid *>(&B));
}
return LocalFlock;
}
assert(UsingLayout == Global);
std::vector<Boid *> GlobalFlock;
const FlockData &FD = BoidsGlobalData.at(FlockID);
for (auto It = FD.BoidIDs.begin(); It != FD.BoidIDs.end(); It++)
{
assert((*It) < BoidsGlobal.size());
const Boid &B = BoidsGlobal[*It];
GlobalFlock.push_back(const_cast<Boid *>(&B));
}
return GlobalFlock;
}

std::vector<Boid> *NLayout::GetAllBoidsPtr() const
{
if (UsingLayout == Local)
{
return const_cast<std::vector<Boid> *>(&BoidsLocal);
}
assert(UsingLayout == Global);
return const_cast<std::vector<Boid> *>(&BoidsGlobal);
}

Boid *NLayout::GetBoidF(const size_t Idx) const
{
if (UsingLayout == Global)
{
assert(Idx < BoidsGlobalData.at(FlockID).Size());
size_t GlobalIdx = BoidsGlobalData.at(FlockID).GetBoidIdx(Idx); 
assert(GlobalIdx < BoidsGlobal.size());
return (*this)[GlobalIdx];
}
return (*this)[Idx];
}

Boid *NLayout::operator[](const size_t Idx) const
{
if (UsingLayout == Local)
{
assert(Idx < BoidsLocal.size());
return const_cast<Boid *>(&(BoidsLocal[Idx]));
}
assert(UsingLayout == Global);
assert(Idx < BoidsGlobal.size());
return const_cast<Boid *>(&(BoidsGlobal[Idx]));
}

void NLayout::ClearLocal()
{
if (UsingLayout == Local)
{
assert(IsValid());
BoidsLocal.clear();
}
}

void NLayout::Destroy()
{
Boid::Destroy();
if (UsingLayout == Local)
{
ClearLocal();
}
assert(IsValid());
if (BoidsGlobal.size() > 0)
{
BoidsGlobal.clear();
for (auto It = BoidsGlobalData.begin(); It != BoidsGlobalData.end(); It++)
{
FlockData &F = It->second;
F.BoidIDs.clear();
}
BoidsGlobalData.clear();
}
}

void NLayout::Append(const std::vector<Boid> &Immigrants)
{
if (UsingLayout == Local)
{
BoidsLocal.insert(BoidsLocal.end(), Immigrants.begin(), Immigrants.end());
assert(IsValid());
}
else
{
assert(UsingLayout == Global);
for (const Boid &B : Immigrants)
{
const size_t Idx = B.BoidID;
if (BoidsGlobal[Idx].FlockID == FlockID)
{
continue; 
}
assert(Idx < BoidsGlobal.size());
assert(Idx < BoidsGlobalData.size());
#pragma omp critical
{
BoidsGlobalData.at(BoidsGlobal[Idx].FlockID).Remove(B); 
BoidsGlobal[Idx].FlockID = FlockID;                     
BoidsGlobalData.at(FlockID).Add(B);                     
assert(IsValid());
}
}
}
}