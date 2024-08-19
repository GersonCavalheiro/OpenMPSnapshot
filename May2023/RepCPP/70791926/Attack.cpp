#include "Attack.hpp"
#include "log.hpp"
#include "Crc32Tab.hpp"
#include "KeystreamTab.hpp"
#include "MultTab.hpp"

Attack::Attack(const Data& data, std::size_t index, std::vector<Keys>& solutions, bool exhaustive, Progress& progress)
: data(data), index(index + 1 - Attack::CONTIGUOUS_SIZE), solutions(solutions), exhaustive(exhaustive), progress(progress)
{}

void Attack::carryout(uint32 z7_2_32)
{
zlist[7] = z7_2_32;
exploreZlists(7);
}

void Attack::exploreZlists(int i)
{
if(i != 0) 
{
uint32 zim1_10_32 = Crc32Tab::getZim1_10_32(zlist[i]);

for(uint32 zim1_2_16 : KeystreamTab::getZi_2_16_vector(data.keystream[index+i-1], zim1_10_32))
{
zlist[i-1] = zim1_10_32 | zim1_2_16;

zlist[i] &= MASK_2_32; 
zlist[i] |= (Crc32Tab::crc32inv(zlist[i], 0) ^ zlist[i-1]) >> 8;

if(i < 7)
ylist[i+1] = Crc32Tab::getYi_24_32(zlist[i+1], zlist[i]);

exploreZlists(i-1);
}
}
else 
{
for(uint32 y7_8_24 = 0, prod = (MultTab::getMultinv(msb(ylist[7])) << 24) - MultTab::MULTINV;
y7_8_24 < 1 << 24;
y7_8_24 += 1 << 8, prod += MultTab::MULTINV << 8)
for(byte y7_0_8 : MultTab::getMsbProdFiber3(msb(ylist[6]) - msb(prod)))
if(prod + MultTab::getMultinv(y7_0_8) - (ylist[6] & MASK_24_32) <= MAXDIFF_0_24)
{
ylist[7] = y7_0_8 | y7_8_24 | (ylist[7] & MASK_24_32);
exploreYlists(7);
}
}
}

void Attack::exploreYlists(int i)
{
if(i != 3) 
{
uint32 fy = (ylist[i] - 1) * MultTab::MULTINV;
uint32 ffy = (fy - 1) * MultTab::MULTINV;

for(byte xi_0_8 : MultTab::getMsbProdFiber2(msb(ffy - (ylist[i-2] & MASK_24_32))))
{
uint32 yim1 = fy - xi_0_8;

if(ffy - MultTab::getMultinv(xi_0_8) - (ylist[i-2] & MASK_24_32) <= MAXDIFF_0_24
&& msb(yim1) == msb(ylist[i-1]))
{
ylist[i-1] = yim1;

xlist[i] = xi_0_8;

exploreYlists(i-1);
}
}
}
else 
testXlist();
}

void Attack::testXlist()
{
for(int i = 5; i <= 7; i++)
xlist[i] = (Crc32Tab::crc32(xlist[i-1], data.plaintext[index+i-1])
& MASK_8_32) 
| lsb(xlist[i]); 

uint32 x = xlist[7];
for(int i = 6; i >= 3; i--)
x = Crc32Tab::crc32inv(x, data.plaintext[index+i]);

uint32 y1_26_32 = Crc32Tab::getYi_24_32(zlist[1], zlist[0]) & MASK_26_32;
if(((ylist[3] - 1) * MultTab::MULTINV - lsb(x) - 1) * MultTab::MULTINV - y1_26_32 > MAXDIFF_0_26)
return;

Keys keysForward(xlist[7], ylist[7], zlist[7]);
keysForward.update(data.plaintext[index+7]);
for(bytevec::const_iterator p = data.plaintext.begin() + index + 8,
c = data.ciphertext.begin() + data.offset + index + 8;
p != data.plaintext.end();
++p, ++c)
{
if((*c ^ keysForward.getK()) != *p)
return;
keysForward.update(*p);
}

std::size_t indexForward = data.offset + data.plaintext.size();

Keys keysBackward(x, ylist[3], zlist[3]);
using rit = std::reverse_iterator<bytevec::const_iterator>;
for(rit p = rit(data.plaintext.begin() + index + 3),
c = rit(data.ciphertext.begin() + data.offset + index + 3);
p != data.plaintext.rend();
++p, ++c)
{
keysBackward.updateBackward(*c);
if((*c ^ keysBackward.getK()) != *p)
return;
}

std::size_t indexBackward = data.offset;

for(const std::pair<std::size_t, byte>& extra : data.extraPlaintext)
{
byte p;
if(extra.first < indexBackward)
{
keysBackward.updateBackward(data.ciphertext, indexBackward, extra.first);
indexBackward = extra.first;
p = data.ciphertext[indexBackward] ^ keysBackward.getK();
}
else
{
keysForward.update(data.ciphertext, indexForward, extra.first);
indexForward = extra.first;
p = data.ciphertext[indexForward] ^ keysForward.getK();
}

if(p != extra.second)
return;
}


keysBackward.updateBackward(data.ciphertext, indexBackward, 0);

#pragma omp critical
solutions.push_back(keysBackward);

progress.log([&keysBackward](std::ostream& os)
{
os << "Keys: " << keysBackward << std::endl;
});

if(!exhaustive)
progress.state = Progress::State::EarlyExit;
}

std::vector<Keys> attack(const Data& data, const u32vec& zi_2_32_vector, std::size_t index, const bool exhaustive, Progress& progress)
{
const uint32* candidates = zi_2_32_vector.data();
const std::int32_t size = zi_2_32_vector.size();

std::vector<Keys> solutions;
Attack worker(data, index, solutions, exhaustive, progress);

progress.done = 0;
progress.total = size;

#pragma omp parallel for firstprivate(worker) schedule(dynamic)
for(std::int32_t i = 0; i < size; ++i) 
{
if(progress.state != Progress::State::Normal)
continue; 

worker.carryout(candidates[i]);

progress.done++;
}

return solutions;
}
