#pragma once

#include <fstream>
#include <string>
#include <cstring>
#include <iostream>
#include <memory>


class CPU_profile{
public:
CPU_profile(){
_n_cores = n_cores();
_proc_freq = std::make_unique<uint32_t[]>(_n_cores);
}

inline uint16_t getCores() const { return _n_cores; }
void update_freq(){ parse_proc_freq("MHz"); }
inline uint32_t operator[](uint16_t idx) { return _proc_freq[idx]; }
inline const uint32_t operator[](uint16_t idx) const { return _proc_freq[idx]; }

inline void show_cpu_freq(char eol='\n') const {
for(int n = 0; n < _n_cores; ++n) std::cout << " | " << _proc_freq[n];
std::cout << " | " << eol << std::flush;
}

private:
uint16_t n_cores(){ return parse_value<uint16_t>("cores"); }


template<typename T>
T parse_value(const std::string& key){
std::ifstream read("/proc/cpuinfo");
for(std::string line; std::getline(read, line);){
char *token = std::strtok(&line[0], "	: ");
while(token != NULL){
std::string token_str{ token };
if(token_str == key){
token = std::strtok(NULL, "	: ");
std::string token_str_2{ token };
return static_cast<T>(std::stoul(token_str_2)); 
}
token = std::strtok(NULL, "	: ");
}
}
return 0;
}

void parse_proc_freq(const std::string& key){
std::ifstream read("/proc/cpuinfo");
uint32_t curr_proc_idx{ 0 };
for(std::string line; std::getline(read, line);){
char *token = std::strtok(&line[0], "	: ");
while(token != NULL){
std::string token_str{ token };
if(token_str == key){
token = std::strtok(NULL, "	: ");
std::string token_str_2{ token };
_proc_freq[curr_proc_idx++] = static_cast<uint32_t>(std::stoul(token_str_2));
}
token = std::strtok(NULL, "	: ");
}
}
}

uint16_t _n_cores;
std::unique_ptr<uint32_t[]> _proc_freq;
};