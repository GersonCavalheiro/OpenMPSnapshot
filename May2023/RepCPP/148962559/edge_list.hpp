
#pragma once

#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>

#include <unordered_map>

#include "../rabbit_order.hpp"

namespace edge_list {
namespace aux {

using rabbit_order::vint;
typedef std::tuple<vint, vint, float> edge;
typedef std::tuple<vint, vint> unweighted_edge;

off_t file_size(const int fd) {
struct stat st;
if (fstat(fd, &st) != 0)
RO_DIE << "stat(2): " << strerror(errno);
return st.st_size;
}

struct file_desc {
int fd;

file_desc(const std::string &path) {
fd = open(path.c_str(), O_RDONLY);
if (fd == -1)
RO_DIE << "open(2): " << strerror(errno);
}

~file_desc() {
if (close(fd) != 0)
RO_DIE << "close(2): " << strerror(errno);
}
};

struct mmapped_file {
size_t size;
const char *data;

mmapped_file(const file_desc &fd) {
size = file_size(fd.fd);
data = static_cast<char *>(
mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd.fd, 0));
if (data == NULL)
RO_DIE << "mmap(2): " << strerror(errno);
}

~mmapped_file() {
if (munmap(const_cast<char *>(data), size) != 0)
RO_DIE << "munmap(2): " << strerror(errno);
}
};

struct edge_parser {
const char *const strfirst;
const char *const strlast;
const char *crr;

edge_parser(const char *const first, const char *const last)
: strfirst(first), strlast(last), crr(first) {}

template<typename OutputIt>
void operator()(OutputIt dst) {
while (crr < strlast) {
eat_empty_lines();
if (crr < strlast)
*dst++ = eat_edge();
}
}

edge eat_edge() {
const vint s = eat_id();
eat_separator();
const vint t = eat_id();
return edge{s, t, 1.0};  
}

vint eat_id() {
const auto _crr = crr;
vint v = 0;
for (; crr < strlast && std::isdigit(*crr); ++crr) {
const vint _v = v * 10 + (*crr - '0');
if (_v < v)  
RO_DIE << "Too large vertex ID at line " << crr_line();
v = _v;
}
if (_crr == crr)  
RO_DIE << "Invalid vertex ID at line " << crr_line();
return v;
}

void eat_empty_lines() {
while (crr < strlast) {
if (*crr == '\r') ++crr;                                
else if (*crr == '\n') ++crr;                                
else if (*crr == '#') crr = std::find(crr, strlast, '\n');  
else break;
}
}

void eat_separator() {
while (crr < strlast && (*crr == '\t' || *crr == ',' || *crr == ' '))
++crr;
}

size_t crr_line() {
return std::count(strfirst, crr, '\n');
}
};

std::vector<edge> read(const std::string &path) {
const file_desc fd(path);
const mmapped_file mm(fd);
const int nthread = omp_get_max_threads();
const size_t zchunk = 1024 * 1024 * 64;  
const size_t nchunk = mm.size / zchunk + (mm.size % zchunk > 0);


std::vector<std::deque<edge> > eparts(nthread);
#pragma omp parallel for schedule(dynamic, 1)
for (size_t i = 0; i < nchunk; ++i) {
const char *p = mm.data + zchunk * i;
const char *q = mm.data + std::min(zchunk * (i + 1), mm.size);

if (i > 0) p = std::find(p, q, '\n');

if (p < q) {  
q = std::find(q, mm.data + mm.size, '\n');  
edge_parser(p, q)(std::back_inserter(eparts[omp_get_thread_num()]));
}
}

std::vector<size_t> eheads(nthread + 1);
for (int t = 0; t < nthread; ++t)
eheads[t + 1] = eheads[t] + eparts[t].size();

std::vector<edge> edges(eheads.back());
#pragma omp parallel for schedule(guided, 1)
for (int t = 0; t < nthread; ++t)
boost::copy(eparts[t], edges.begin() + eheads[t]);

return edges;
}

}  

using aux::edge;
using aux::read;

}  

