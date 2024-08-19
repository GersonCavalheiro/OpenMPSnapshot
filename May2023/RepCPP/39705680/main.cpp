

#include <string>
#include <vector>
#include <iostream>
#include <cstdio>
#include <cmath>

#include <ImfRgbaFile.h>

#include "DepthImage.h"
#include "Common.h"

const std::string InputColorPattern = "out.RGB_color.%04d.exr";
const std::string InputDepthPattern = "out.VRayZDepth.%04d.exr";
const size_t MAX_PATTERN_LEN = 255;

std::string PatternToName(const std::string &pattern, int index) {
char fileName[255];
snprintf(fileName, 255, pattern.c_str(), index);
return std::string(fileName);
}

void PrintUsage() {
printf("Usage : dcompose <number of frames> <pattern c1> <d1> <c2> <d2> "
"<out>\n");
printf("\n");
}

void Test() {

int i = 0;
DepthImage im_1;
im_1.EmplaceData(PatternToName(InputColorPattern, i),
PatternToName(InputDepthPattern, i));

im_1.PrintInfo();
im_1.SaveToPNG("out.png");
}

int main(int argc, char *argv[]) {

printf("\n=============================\n");
printf("Randy depth composer utility");
printf("\n=============================\n");




if (argc != 7) {
PrintUsage();
return -1;
}

std::string n_frames_str(argv[1]);
std::string pat_c1(argv[2]);
std::string pat_d1(argv[3]);
std::string pat_c2(argv[4]);
std::string pat_d2(argv[5]);
std::string pat_out(argv[6]);
int n_frames = 0;

try {
n_frames = std::stoi(n_frames_str);
} catch (...) {
printf("Can't parse %s as a number\n", n_frames_str.c_str());
PrintUsage();
return -1;
}

DepthImage im_1, im_2;

for (int i = 0; i < n_frames; ++i) {


im_1.EmplaceData(PatternToName(pat_c1, i), PatternToName(pat_d1, i));

im_2.EmplaceData(PatternToName(pat_c2, i), PatternToName(pat_d2, i));

im_1.PrintInfo();
im_2.PrintInfo();

im_1 += im_2;


im_1.SaveToPNG(PatternToName(pat_out, i));
}

return 0;
}
