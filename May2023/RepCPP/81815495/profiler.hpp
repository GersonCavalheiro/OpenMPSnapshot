#ifndef VEXCL_PROFILER_HPP
#define VEXCL_PROFILER_HPP





#if defined(_MSC_VER) && ( defined(min) || defined(max) )
#  error Please define NOMINMAX macro globally in your project
#endif

#include <iostream>
#include <iomanip>
#include <string>
#include <map>
#include <memory>
#include <stack>
#include <vector>
#include <cassert>

#if defined(_MSC_VER) && (_MSC_VER < 1700)
#  define VEXCL_USE_BOOST_CHRONO
#  include <boost/chrono.hpp>
#else
#  include <chrono>
#endif

#include <boost/multi_index_container.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/sequenced_index.hpp>
#include <boost/multi_index/global_fun.hpp>
#include <boost/io/ios_state.hpp>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/count.hpp>
#include <boost/accumulators/statistics/sum.hpp>
#include <boost/accumulators/statistics/mean.hpp>

#ifdef _MSC_VER
#  pragma warning(push)
#  pragma warning(disable: 4244)
#endif
#include <boost/accumulators/statistics/median.hpp>
#ifdef _MSC_VER
#  pragma warning(pop)
#endif

#include <vexcl/backend.hpp>


namespace vex {


template <
#ifdef VEXCL_USE_BOOST_CHRONO
class Clock = boost::chrono::high_resolution_clock
#else
class Clock = std::chrono::high_resolution_clock
#endif
>
class stopwatch {
boost::accumulators::accumulator_set<
double,
boost::accumulators::stats<
boost::accumulators::tag::count,
boost::accumulators::tag::sum,
boost::accumulators::tag::mean,
boost::accumulators::tag::median(boost::accumulators::with_p_square_quantile)
>
> acc;

public:
stopwatch() { tic(); }

inline void tic() {
start = Clock::now();
}

inline double toc() {
const double delta = seconds(start, Clock::now());

acc(delta);

return delta;
}

inline double average() const {
namespace ba = boost::accumulators;

return ba::count(acc) >= 3 ? ba::median(acc) : ba::mean(acc);
}

inline double total() const {
return boost::accumulators::sum(acc);
}

inline size_t tics() const {
return boost::accumulators::count(acc);
}
private:
static double seconds(typename Clock::time_point begin, typename Clock::time_point end) {
return typename Clock::duration(end - begin).count() *
static_cast<double>(Clock::duration::period::num) /
Clock::duration::period::den;
}

typename Clock::time_point start;
};


template <
#ifdef VEXCL_USE_BOOST_CHRONO
class Clock = boost::chrono::high_resolution_clock
#else
class Clock = std::chrono::high_resolution_clock
#endif
>
class profiler {
private:
class profile_unit {
public:
profile_unit(std::string name) : watch(), name(name), children() {}
virtual ~profile_unit() {}

stopwatch<Clock> watch;
std::string name;

static std::string _name(const std::shared_ptr<profile_unit> &u) {
return u->name;
}

boost::multi_index_container<
std::shared_ptr<profile_unit>,
boost::multi_index::indexed_by<
boost::multi_index::sequenced<>,
boost::multi_index::ordered_unique<boost::multi_index::global_fun<
const std::shared_ptr<profile_unit> &, std::string, profiler::profile_unit::_name>>
>
> children;

virtual void tic() {
watch.tic();
}

virtual double toc() {
return watch.toc();
}

double children_time() const {
double tm = 0;

for(auto c = children.begin(); c != children.end(); c++)
tm += (*c)->watch.total();

return tm;
}

size_t max_line_width(size_t level) const {
size_t w = name.size() + level;

for(auto c = children.begin(); c != children.end(); c++)
w = std::max(w, (*c)->max_line_width(level + shift_width));

return w;
}

void print(std::ostream &out,
size_t level, double total, size_t width) const
{
using namespace std;
print_line(out, name, watch.total(), 100 * watch.total() / total, width, level);

if (watch.tics() > 1) {
out << " (" << setw(6) << watch.tics()
<< "x; avg: " << setprecision(6) << scientific
<< (watch.average() * 1e6) << " usec.)";
}

out << endl;

if (!children.empty()) {
double sec = watch.total() - children_time();
double perc = 100 * sec / total;
if(perc > 1e-1) {
print_line(out, "self", sec, perc, width, level + 1);
out << endl;
}
}

for(auto c = children.begin(); c != children.end(); c++)
(*c)->print(out, level + shift_width, total, width);
}

void print_line(std::ostream &out, const std::string &name,
double time, double perc, size_t width, size_t indent) const {
using namespace std;
out << "[" << setw(indent) << "" << name << ":"
<< setw(width - indent - name.size()) << ""
<< fixed << setw(10) << setprecision(3) << time << " sec."
<< "] (" << setprecision(2) << setw(6) << perc << "%)";
}

private:
static const size_t shift_width = 2U;
};

class cl_profile_unit : public profile_unit {
public:
cl_profile_unit(const std::string &name, std::vector<backend::command_queue> &queue)
: profile_unit(name), queue(queue) {}

void tic() override {
for(auto q = queue.begin(); q != queue.end(); ++q)
q->finish();

profile_unit::tic();
}

double toc() override {
for(auto q = queue.begin(); q != queue.end(); ++q)
q->finish();

return profile_unit::toc();
}
private:
std::vector<backend::command_queue> &queue;
};

public:

profiler(
const std::vector<backend::command_queue> &queue = std::vector<backend::command_queue>(),
const std::string &name = "Profile"
) : queue(queue)
{
auto root = std::shared_ptr<profile_unit>(new profile_unit(name));
root->tic();
stack.push_back(root);
}

void reset() {
auto root = std::shared_ptr<profile_unit>(new profile_unit(stack.front()->name));
stack.clear();
root->tic();
stack.push_back(root);
}

private:
void tic(profile_unit *u) {
assert(!stack.empty());
auto top = stack.back();
auto new_unit = std::shared_ptr<profile_unit>(u);
auto unit = *top->children.push_back(new_unit).first;
unit->tic();
stack.push_back(unit);
}

public:

void tic_cpu(const std::string &name) {
tic(new profile_unit(name));
}


void tic_cl(const std::string &name) {
assert(!queue.empty());
tic(new cl_profile_unit(name, queue));
}


double toc(const std::string &) {
assert(stack.size() > 1);

double delta = stack.back()->toc();
stack.pop_back();

return delta;
}

void print(std::ostream &out) {
boost::io::ios_all_saver stream_state(out);

if(stack.size() != 1)
out << "Warning! Profile is incomplete." << std::endl;

auto root = stack.front();
double length = root->toc();
out << std::endl;
root->print(out, 0, length, root->max_line_width(0));
}

private:
std::vector<backend::command_queue> queue;
std::deque<std::shared_ptr<profile_unit>> stack;
};

} 

namespace std {

template <class Clock>
inline std::ostream& operator<<(std::ostream &os, vex::profiler<Clock> &prof) {
prof.print(os);
return os;
}

}

#endif
