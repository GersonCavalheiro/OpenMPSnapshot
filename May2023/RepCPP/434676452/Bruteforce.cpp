#include "Bruteforce.h"

Bruteforce::Bruteforce(std::string set_plainTextFilename, std::string set_encTextFilename)
{
m_plainTextFilename = set_plainTextFilename;
m_encTextFilename = set_encTextFilename;
cs = Cryptosystem();
}

void Bruteforce::brute(void)
{
std::vector<uint32_t> plaintTextBlocks = getBlocksText(this->m_plainTextFilename);
std::vector<uint32_t> enTextBlocks = getBlocksText(this->m_encTextFilename);

time_t g_timeElapsed;
g_timeElapsed = time(NULL);

omp_set_num_threads(8);

#pragma omp parallel for
for (int32_t key = std::numeric_limits<int32_t>::min(); key < std::numeric_limits<int32_t>::max(); ++key) {
bool found = true;
for (size_t i = 0; i < plaintTextBlocks.size(); ++i) {
uint32_t encBlock = this->cs.calculate(plaintTextBlocks[i], static_cast<uint32_t>(key));
if (encBlock != enTextBlocks[i]) {
found = false;
break;
}
}
if (found) {
#pragma omp critical
{
std::cout << "FOUND KEY: " << key << std::endl;
std::string filename = "probably_key" + std::to_string(key) + ".out";
std::ofstream file(filename, std::ios::binary);
unsigned char bytes[4] = { 0 };
bytes[0] = (key >> 24) & 0xFF;
bytes[1] = (key >> 16) & 0xFF;
bytes[2] = (key >> 8) & 0xFF;
bytes[3] = key & 0xFF;
file << bytes[0] << bytes[1] << bytes[2] << bytes[3];
file.close();
}
}
}
std::cout << "End of task. By!" << std::endl;
}

std::vector<uint32_t> Bruteforce::getBlocksText(std::string fileName)
{
std::ifstream file(fileName, std::ios::binary);
std::vector<uint32_t> blocks;

unsigned char buff[4] = { 0 };

while (file.read(reinterpret_cast<char*>(buff), 4) && !file.eof()) {
uint32_t block = buff[3] + (buff[2] << 8)
+ (buff[1] << 16) + (buff[0] << 24);
blocks.push_back(block);
std::fill(buff, buff + 4, 0);
}

file.close();
return blocks;
}
