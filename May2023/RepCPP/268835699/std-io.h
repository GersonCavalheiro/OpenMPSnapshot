#ifndef __STD_IO_H__
#define __STD_IO_H__

#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <set>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../debug/debug.h"
#include "core-io.h"

namespace __core__ {
namespace __io__ {
namespace __std__ {
template <typename T,typename U=T> std::ostream& print(T *data,std::size_t size,std::ostream &ost=std::cerr,std::string separator=", ",std::string begin="{",std::string end="}") {
ost<<begin;
for(size_t i=0;i<size;++i) {
if(i<size-1)
ost<<static_cast<U>(data[i])<<separator;
else
ost<<static_cast<U>(data[i]);
}
ost<<end;
return ost;
}
template <typename T,typename U=T> std::ostream& print(T *data,std::size_t rows,std::size_t cols,std::ostream &ost=std::cerr,std::string separator=", ",
std::string rbegin="{",std::string rend="}",std::string begin="{",std::string end="}",std::string newRow="\n") {
ost<<begin;
for(std::size_t i=0;i<rows;++i) {
ost<<rbegin;
for(std::size_t j=0;j<cols;++j) {
ost<<static_cast<U>(data[i*cols+j]);
if(j!=cols-1)
ost<<separator;
}
if(i!=rows-1)
ost<<rend<<newRow;
else
ost<<rend<<end;
}
return ost;
}
template <typename T,typename U=T> std::ostream& print(const std::vector<T>& v,std::size_t size,std::ostream &ost=std::cerr,std::string separator=", ",std::string begin="{",std::string end="}") {
size=std::min(size,v.size());
ost<<begin;
for(size_t i=0;i<size;++i) {
if(i<size-1)
ost<<static_cast<U>(v[i])<<separator;
else
ost<<static_cast<U>(v[i]);
}
ost<<end;
return ost;
}
template <typename T,typename U=T> std::ostream& print(const std::vector<T>& v,std::size_t rows,std::size_t cols,std::ostream &ost=std::cerr,std::string separator=", ",
std::string rbegin="{",std::string rend="}",std::string begin="{",std::string end="}",std::string newRow="\n") {
ost<<begin;
for(std::size_t i=0;i<rows;++i) {
ost<<rbegin;
for(std::size_t j=0;j<cols;++j) {
ost<<static_cast<U>(v[i*cols+j]);
if(j!=cols-1)
ost<<separator;
}
ost<<rend<<newRow;
}
ost<<end;
return ost;
}
template <typename K,typename V,typename C> std::ostream & print(const std::map<K,V,C> &map,std::ostream &ost=std::cerr,std::size_t size=std::numeric_limits<size_t>::max(),std::string separator=", ",
std::string begin="{",std::string end="}",std::function<void(std::ostream &,K,V)> pprinter = [](std::ostream &__ost__,K key,V val) -> void { __ost__<<"["<<key<<"]->"<<val; }) {
size=std::min(size,map.size());
ost<<begin;
size_t c=0;
for(typename std::map<K,V,C>::const_iterator i=map.cbegin();i!=map.cend()&&(c<size);++i) {
pprinter(ost,i->first,i->second);
if((c++)!=(map.size()-1))
ost<<separator;
}
ost<<end;
return ost;
}
template <typename K,typename V,typename C> std::ostream & print(const std::unordered_map<K,V,C> &map,std::ostream &ost=std::cerr,std::size_t size=std::numeric_limits<size_t>::max(),std::string separator=", ",
std::string begin="{",std::string end="}",std::function<void(std::ostream &,K,V)> pprinter = [](std::ostream &__ost__,K key,V val) -> void { __ost__<<"["<<key<<"]->"<<val; }) {
size=std::min(size,map.size());
ost<<begin;
size_t c=0;
for(auto i=map.cbegin();(i!=map.cend())&&(c<size);++i) {
pprinter(ost,i->first,i->second);
if((c++)!=(map.size()-1))
ost<<separator;
}
ost<<end;
return ost;
}
template <typename T> std::ostream& print(const std::set<T>& set,std::ostream &ost=std::cerr,std::size_t size=std::numeric_limits<size_t>::max(),std::string separator=", ",std::string begin="{",std::string end="}") {
size=std::min(size,set.size());
size_t k=0;
ost<<begin;
for(auto i=set.cbegin();i!=set.cend()&&(k<size);++i) {
if(k==(size-1))
ost<<(*i)+1;
else
ost<<(*i)+1<<separator;
++k;
}
ost<<end;
return ost;
}
template <typename T> std::ostream& print(const std::unordered_set<T>& set,std::ostream &ost=std::cerr,std::size_t size=std::numeric_limits<size_t>::max(),std::string separator=", ",std::string begin="{",std::string end="}") {
size=std::min(size,set.size());
size_t k=0;
ost<<begin;
for(auto i=set.cbegin();i!=set.cend()&&(k<size);++i) {
if(k==(size-1))
ost<<(*i);
else
ost<<(*i)<<separator;
++k;
}
ost<<end;
return ost;
}

template <typename T> std::ostream & operator<<(std::ostream &ost,const std::vector<T> &vec) {
return print(vec,vec.size(),ost);
}
template <typename K,typename V,typename C> std::ostream & operator<<(std::ostream &ost,const std::map<K,V,C> &map) {
return print(map,ost);
}
template <typename K,typename V,typename C> std::ostream & operator<<(std::ostream &ost,const std::unordered_map<K,V,C> &map) {
return print(map,ost);
}
template <typename T> std::ostream& operator<<(std::ostream &ost,const std::set<T>& set) {
return print(set,ost);
}
template <typename T> std::ostream& operator<<(std::ostream &ost,const std::unordered_set<T>& set) {
return print(set,ost);
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
template <unsigned int mode=0,typename std::enable_if<mode==0,int>::type = 0> std::ifstream open_file(std::string file_name,std::ios_base::openmode openMode = std::ios_base::in) {
std::ifstream file;
try {
file.open(file_name,openMode);
if(file.is_open())
return std::move(file);
else {
error(true,"Couldn't open the file: "+file_name,RUNTIME_ERROR);
}
}
catch(std::ios_base::failure &exc) {
error(true,"Couldn't open the file: "+file_name+exc.what(),RUNTIME_ERROR);
}
file=std::ifstream();
file.setstate(std::ios_base::badbit);
return std::move(file);
}
template <unsigned int mode=0,typename std::enable_if<mode==1,int>::type = 0> std::ofstream open_file(std::string file_name,std::ios_base::openmode openMode = std::ios_base::out) {
std::ofstream file;
try {
file.open(file_name,openMode);
if(file.is_open())
return std::move(file);
else {
error(true,"Couldn't open the file: "+file_name,RUNTIME_ERROR);
}
}
catch(std::ios_base::failure &exc) {
error(true,"Couldn't open the file: "+file_name+exc.what(),RUNTIME_ERROR);
}
file=std::ofstream();
file.setstate(std::ios_base::badbit);
return std::move(file);
}
template <unsigned int mode=0,typename std::enable_if<(mode>=2),int>::type = 0> std::fstream open_file(std::string file_name,std::ios_base::openmode openMode=std::ios_base::in|std::ios_base::out) {
std::fstream file;
try {
file.open(file_name,openMode);
if(file.is_open())
return std::move(file);
else {
error(true,"Couldn't open the file: "+file_name,RUNTIME_ERROR);
}
}
catch(std::ios_base::failure &exc) {
error(true,"Couldn't open the file: "+file_name+", error given: "+exc.what(),RUNTIME_ERROR);
}
file=std::fstream();
file.setstate(std::ios_base::badbit);
return std::move(file);
}
void close_file(std::ifstream& file);
void close_file(std::ofstream& file);
void close_file(std::fstream& file);
std::vector<std::string>&& read_file(const std::string& file_name);
#pragma GCC diagnostic pop
}
}
}
#endif
