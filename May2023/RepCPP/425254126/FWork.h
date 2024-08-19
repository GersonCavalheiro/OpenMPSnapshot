#include <utility>
#include <fstream>
#include <iomanip>

#pragma once

class FWork {
private:
str fullPath;

protected:
bool write(std::ofstream& output, const vec2d<double>& u, const str& step) {
if (!output.is_open()) {
return false;
}

output << std::setprecision(8);
output << "Step s -------- " << step << "\n";

loop3(k) {
output << "component --- " << k << "\n";
for (size_t j = 0; j < u[k][0].size(); j++) {
for (size_t i = 0; i < u[k].size(); i++) {
output << std::setw(12) << u[k][i][j] << " ";
} output << "\n";
} output << "\n";
} output << "\n";

output.close();
return true;
}

public:
explicit FWork(bool clean) {
fullPath = AppConstansts::FULL_PATH;

this->clearFiles(clean);
}

void clearFiles(bool clean) {
if (clean) {
std::ofstream ofs;

for (const auto& layer : {AppConstansts::HALF_LAYER, AppConstansts::MAIN_LAYER}) {
ofs.open(fullPath + layer + ".txt", std::ofstream::out | std::ofstream::trunc);
ofs.close();
}
}
}

bool fwrite(const vec2d<double>& uPhase, size_t s, const str& layer) {
std::ofstream output(fullPath + layer + ".txt", std::ios::app);

str step = (layer == AppConstansts::MAIN_LAYER)
? std::to_string(s)
: std::to_string(s) + ".5";

this->write(output, uPhase, step);

return true;
}

bool fread(const str& layer) {
std::ifstream input(fullPath + layer + ".txt");

if (!input.is_open()) {
return false;
}

str line;
while (std::getline(input, line)) {
std::cout << line << "\n";
}

input.close();
return true;
}
};