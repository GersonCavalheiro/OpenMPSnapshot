

#pragma once

#include <iostream>
#include <sstream>
#include <string.h>
#include <vector>



class rocalution_bench_cmdlines
{
private:
struct val
{
public:
int    argc{};
char** argv{};
~val()
{
if(this->argv != nullptr)
{
delete[] this->argv;
this->argv = nullptr;
}
}

val(){};
val(int n)
: argc(n)
{
this->argv = new char*[this->argc];
}

val& operator()(int n)
{
this->argc = n;
if(this->argv)
{
delete[] this->argv;
}
this->argv = new char*[this->argc];
return *this;
}
};

struct cmdline
{

public:
const char* get_ofilename() const
{
return this->m_ofilename;
};

int get_nplots() const
{
return this->get_nsamples() / this->m_options[this->m_option_index_x].args.size();
};

int get_noptions_x() const
{
return this->m_options[this->m_option_index_x].args.size();
};
int get_noptions() const
{
return this->m_options.size();
};
int get_option_nargs(int i)
{
return this->m_options[i].args.size();
}
const char* get_option_arg(int i, int j)
{
return this->m_options[i].args[j].name;
}
const char* get_option_name(int i)
{
return this->m_options[i].name;
}
int get_nsamples() const
{
return this->m_nsamples;
}

int get_option_index_x() const
{
return this->m_option_index_x;
}

int get_nruns() const
{
return this->m_bench_nruns;
}

bool is_stdout_disabled() const
{
return this->m_is_stdout_disabled;
}

cmdline(int argc, char** argv)
{

int detected_option_bench_n
= detect_option(argc, argv, "--bench-n", this->m_bench_nruns);

if(detected_option_bench_n == -1)
{
std::cerr << "missing parameter ?" << std::endl;
exit(1);
}

int detected_option_bench_o
= detect_option_string(argc, argv, "--bench-o", this->m_ofilename);
if(detected_option_bench_o == -1)
{
std::cerr << "missing parameter ?" << std::endl;
exit(1);
}

const char* option_x        = nullptr;
int detected_option_bench_x = detect_option_string(argc, argv, "--bench-x", option_x);
if(detected_option_bench_x == -1 || false == is_option(option_x))
{
std::cerr << "wrong position of option --bench-x  ?" << std::endl;
exit(1);
}

this->m_name = argv[0];
this->m_has_bench_option
= (detected_option_bench_x || detected_option_bench_o || detected_option_bench_n);

this->m_is_stdout_disabled = (false == detect_flag(argc, argv, "--bench-std"));

int jarg = -1;
for(int iarg = 1; iarg < argc; ++iarg)
{
if(argv[iarg] == option_x)
{
jarg = iarg;
break;
}
}

int iarg = 1;
while(iarg < argc)
{
if(is_option(argv[iarg]))
{
if(!strcmp(argv[iarg], "--bench-std"))
{
++iarg;
}
else if(!strcmp(argv[iarg], "--bench-o"))
{
iarg += 2;
}
else if(!strcmp(argv[iarg], "--bench-x"))
{
++iarg;
}
else if(!strcmp(argv[iarg], "--bench-n"))
{
iarg += 2;
}
else
{
cmdline_option option(argv[iarg]);

const int option_nargs      = count_option_nargs(iarg, argc, argv);
const int next_option_index = iarg + 1 + option_nargs;
for(int k = iarg + 1; k < next_option_index; ++k)
{
option.args.push_back(cmdline_arg(argv[k]));
}


if(jarg == iarg) 
{
this->m_option_index_x = this->m_options.size();
}

this->m_options.push_back(option);
iarg = next_option_index;
}
}
else
{
this->m_args.push_back(cmdline_arg(argv[iarg]));
++iarg;
}
}

this->m_nsamples = 1;
for(size_t ioption = 0; ioption < this->m_options.size(); ++ioption)
{
size_t n = this->m_options[ioption].args.size();
this->m_nsamples *= std::max(n, static_cast<size_t>(1));
}
}

void expand(val* p)
{
const int  num_options = this->m_options.size();
const auto num_samples = this->m_nsamples;
for(int i = 0; i < num_samples; ++i)
{
p[i](1 + this->m_args.size() + num_options * 2);
p[i].argc = 0;
}

for(int i = 0; i < num_samples; ++i)
{
p[i].argv[p[i].argc++] = this->m_name;
}

for(auto& arg : this->m_args)
{
for(int i = 0; i < num_samples; ++i)
p[i].argv[p[i].argc++] = arg.name;
}

const int option_x_nargs = this->m_options[this->m_option_index_x].args.size();
int       N              = option_x_nargs;
for(int iopt = 0; iopt < num_options; ++iopt)
{
cmdline_option& option = this->m_options[iopt];

for(int isample = 0; isample < num_samples; ++isample)
{
p[isample].argv[p[isample].argc++] = option.name;
}

if(iopt == this->m_option_index_x)
{
{
const int ngroups = num_samples / option_x_nargs;
for(int jgroup = 0; jgroup < ngroups; ++jgroup)
{
for(int ix = 0; ix < option_x_nargs; ++ix)
{
const int flat_index = jgroup * option_x_nargs + ix;
p[flat_index].argv[p[flat_index].argc++] = option.args[ix].name;
}
}
}

for(int isample = 0; isample < num_samples; ++isample)
{
if(p[isample].argc != p[0].argc)
{
std::cerr << "invalid struct line " << __LINE__ << std::endl;
}
}
}
else
{
const int option_narg = option.args.size();
if(option_narg > 1)
{
const int ngroups = num_samples / (N * option_narg);
for(int jgroup = 0; jgroup < ngroups; ++jgroup)
{
for(int option_iarg = 0; option_iarg < option_narg; ++option_iarg)
{
for(int i = 0; i < N; ++i)
{
const int flat_index
= N * (jgroup * option_narg + option_iarg) + i;
p[flat_index].argv[p[flat_index].argc++]
= option.args[option_iarg].name;
}
}
}
N *= std::max(option_narg, 1);
}
else
{
if(option_narg == 1)
{
for(int isample = 0; isample < num_samples; ++isample)
{
p[isample].argv[p[isample].argc++] = option.args[0].name;
}
}
}
}
}
}

private:
static inline int count_option_nargs(int iarg, int argc, char** argv)
{
int c = 0;
for(int j = iarg + 1; j < argc; ++j)
{
if(is_option(argv[j]))
{
return c;
}
++c;
}
return c;
}

static bool detect_flag(int argc, char** argv, const char* option_name)
{
for(int iarg = 1; iarg < argc; ++iarg)
{
if(!strcmp(argv[iarg], option_name))
{
return true;
}
}
return false;
}
template <typename T>
static int detect_option(int argc, char** argv, const char* option_name, T& value)
{
for(int iarg = 1; iarg < argc; ++iarg)
{
if(!strcmp(argv[iarg], option_name))
{
++iarg;
if(iarg < argc)
{
std::istringstream iss(argv[iarg]);
iss >> value;
return 1;
}
else
{
std::cerr << "missing value for option --bench-n " << std::endl;
return -1;
}
}
}
return 0;
}

static int
detect_option_string(int argc, char** argv, const char* option_name, const char*& value)
{
for(int iarg = 1; iarg < argc; ++iarg)
{
if(!strcmp(argv[iarg], option_name))
{
++iarg;
if(iarg < argc)
{
value = argv[iarg];
return 1;
}
else
{
std::cerr << "missing value for option " << option_name << std::endl;
return -1;
}
}
}
return 0;
}

struct cmdline_arg
{
char* name{};
cmdline_arg(char* name_)
: name(name_){};
};

struct cmdline_option
{
char*                    name{};
std::vector<cmdline_arg> args{};
cmdline_option(char* name_)
: name(name_){};
};

static inline bool is_option(const char* arg)
{
return arg[0] == '-';
}

char* m_name;

std::vector<cmdline_option> m_options;

std::vector<cmdline_arg> m_args;
bool                     m_has_bench_option{};
int                      m_bench_nruns{1};
int                      m_option_index_x;
int                      m_nsamples;
bool                     m_is_stdout_disabled{true};
const char*              m_ofilename{};
};

private:
cmdline m_cmd;
val*    m_cmdset{};

public:
const char* get_ofilename() const;

int         get_nsamples() const;
int         get_option_index_x() const;
int         get_option_nargs(int i);
const char* get_option_arg(int i, int j);
const char* get_option_name(int i);
int         get_noptions_x() const;
int         get_noptions() const;
bool        is_stdout_disabled() const;

int  get_nruns() const;
void get(int isample, int& argc, char** argv) const;

void get_argc(int isample, int& argc_) const;

rocalution_bench_cmdlines(int argc, char** argv);

static bool applies(int argc, char** argv);

void info() const;
};
