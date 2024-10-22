

#include "boost/fiber/numa/topology.hpp"

#include <exception>
#include <map>
#include <regex>
#include <set>
#include <string>
#include <utility>

#include <boost/algorithm/string.hpp>
#include <boost/assert.hpp>
#include <boost/config.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/format.hpp>

#include "boost/fiber/exceptions.hpp"

#ifdef BOOST_HAS_ABI_HEADERS
# include BOOST_ABI_PREFIX
#endif

#if !defined(BOOST_NO_CXX11_HDR_REGEX)

namespace al = boost::algorithm;
namespace fs = boost::filesystem;

namespace {

class directory_iterator {
private:
fs::directory_iterator          i_;
fs::directory_iterator          e_;
std::regex                      exp_;
std::pair< std::uint32_t, fs::path > idx_;

bool eval_( fs::directory_entry const& entry) {
std::string filename( entry.path().filename().string() );
std::smatch what;
if ( ! std::regex_search( filename, what, exp_) ) {
return false;
}
idx_ = std::make_pair( std::stoul( what[1].str() ), entry.path() );
return true;
}

public:
typedef std::input_iterator_tag iterator_category;
typedef const std::pair< std::uint32_t, fs::path > value_type;
typedef std::ptrdiff_t difference_type;
typedef value_type * pointer;
typedef value_type & reference;

directory_iterator() :
i_(), e_(), exp_(), idx_() {
}

directory_iterator( fs::path const& dir, std::string const& exp) :
i_( dir), e_(), exp_( exp), idx_() {
while ( i_ != e_ && ! eval_( * i_) ) {
++i_;
}
}

bool operator==( directory_iterator const& other) {
return i_ == other.i_;
}

bool operator!=( directory_iterator const& other) {
return i_ != other.i_;
}

directory_iterator & operator++() {
do {
++i_;
} while ( i_ != e_ && ! eval_( * i_) );
return * this;
}

directory_iterator operator++( int) {
directory_iterator tmp( * this);
++*this;
return tmp;
}

reference operator*() const {
return idx_;
}

pointer operator->() const {
return & idx_;
}
};

std::set< std::uint32_t > ids_from_line( std::string const& content) {
std::set< std::uint32_t > ids;
std::vector< std::string > v1;
al::split( v1, content, al::is_any_of(",") );
for ( std::string entry : v1) {
al::trim( entry);
std::vector< std::string > v2;
al::split( v2, entry, al::is_any_of("-") );
BOOST_ASSERT( ! v2.empty() );
if ( 1 == v2.size() ) {
ids.insert( std::stoul( v2[0]) );
} else {
std::uint32_t first = std::stoul( * v2.begin() );
std::uint32_t last = std::stoul( * v2.rbegin() );
for ( std::uint32_t i = first; i <= last; ++i) {
ids.insert( i);
}
}
}
return ids;
}

std::vector< std::uint32_t > distance_from_line( std::string const& content) {
std::vector< std::uint32_t > distance;
std::vector< std::string > v1;
al::split( v1, content, al::is_any_of(" ") );
for ( std::string entry : v1) {
al::trim( entry);
BOOST_ASSERT( ! entry.empty() );
distance.push_back( std::stoul( entry) );
}
return distance;
}

}

namespace boost {
namespace fibers {
namespace numa {

BOOST_FIBERS_DECL
std::vector< node > topology() {
std::vector< node > topo;
fs::ifstream fs_online{ fs::path("/sys/devices/system/cpu/online") };
std::string content;
std::getline( fs_online, content);
std::set< std::uint32_t > cpus = ids_from_line( content);
if ( cpus.empty() ) {
return topo;
}
std::map< std::uint32_t, node > map;
for ( std::uint32_t cpu_id : cpus) {
fs::path cpu_path{
boost::str(
boost::format("/sys/devices/system/cpu/cpu%d/") % cpu_id) };
BOOST_ASSERT( fs::exists( cpu_path) );
directory_iterator e;
for ( directory_iterator i{ cpu_path, "^node([0-9]+)$" };
i != e; ++i) {
std::uint32_t node_id = i->first;
map[node_id].id = node_id;
map[node_id].logical_cpus.insert( cpu_id);
break;
}
}
if ( map.empty() ) {
map[0].id = 0;
for ( std::uint32_t cpu_id : cpus) {
map[0].logical_cpus.insert( cpu_id);
}
}
for ( auto entry : map) {
fs::path distance_path{
boost::str(
boost::format("/sys/devices/system/node/node%d/distance") % entry.second.id) };
if ( fs::exists( distance_path) ) {
fs::ifstream fs_distance{ distance_path };
std::string content;
std::getline( fs_distance, content);
entry.second.distance = distance_from_line( content);
topo.push_back( entry.second);
} else {
entry.second.distance.push_back( 10);
topo.push_back( entry.second);
}
}
return topo;
}

}}}

#else

namespace boost {
namespace fibers {
namespace numa {

#if BOOST_COMP_CLANG || \
BOOST_COMP_GNUC || \
BOOST_COMP_INTEL ||  \
BOOST_COMP_MSVC
# pragma message "topology() not supported without <regex>"
#endif

BOOST_FIBERS_DECL
std::vector< node > topology() {
throw fiber_error{
std::make_error_code( std::errc::function_not_supported),
"boost fiber: topology() not supported without <regex>" };
return std::vector< node >{};
}

}}}

#endif

#ifdef BOOST_HAS_ABI_HEADERS
# include BOOST_ABI_SUFFIX
#endif
