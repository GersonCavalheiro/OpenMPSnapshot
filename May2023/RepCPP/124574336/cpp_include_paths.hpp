

#if !defined(BOOST_CPP_INCLUDE_PATHS_HPP_AF620DA4_B3D2_4221_AD91_8A1ABFFB6944_INCLUDED)
#define BOOST_CPP_INCLUDE_PATHS_HPP_AF620DA4_B3D2_4221_AD91_8A1ABFFB6944_INCLUDED

#include <string>
#include <list>
#include <utility>

#include <boost/assert.hpp>
#include <boost/wave/wave_config.hpp>
#include <boost/wave/util/filesystem_compatibility.hpp>

#if BOOST_WAVE_SUPPORT_PRAGMA_ONCE != 0
#include <boost/multi_index_container.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/ordered_index.hpp>
#endif

#if BOOST_WAVE_SERIALIZATION != 0
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/collections_save_imp.hpp>
#include <boost/serialization/collections_load_imp.hpp>
#include <boost/serialization/split_free.hpp>
#endif

#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>

#ifdef BOOST_HAS_ABI_HEADERS
#include BOOST_ABI_PREFIX
#endif

namespace boost { namespace wave { namespace util {

#if BOOST_WAVE_SUPPORT_PRAGMA_ONCE != 0
struct from {};
struct to {};

template<typename FromType, typename ToType>
struct bidirectional_map
{
typedef std::pair<FromType, ToType> value_type;

#if defined(BOOST_NO_POINTER_TO_MEMBER_TEMPLATE_PARAMETERS) || \
(defined(BOOST_MSVC) && (BOOST_MSVC == 1600) ) || \
(defined(BOOST_INTEL_CXX_VERSION) && \
(defined(_MSC_VER) && (BOOST_INTEL_CXX_VERSION <= 700)))

BOOST_STATIC_CONSTANT(unsigned, from_offset = offsetof(value_type, first));
BOOST_STATIC_CONSTANT(unsigned, to_offset   = offsetof(value_type, second));

typedef boost::multi_index::multi_index_container<
value_type,
boost::multi_index::indexed_by<
boost::multi_index::ordered_unique<
boost::multi_index::tag<from>,
boost::multi_index::member_offset<value_type, FromType, from_offset>
>,
boost::multi_index::ordered_non_unique<
boost::multi_index::tag<to>,
boost::multi_index::member_offset<value_type, ToType, to_offset>
>
>
> type;

#else

typedef boost::multi_index::multi_index_container<
value_type,
boost::multi_index::indexed_by<
boost::multi_index::ordered_unique<
boost::multi_index::tag<from>,
boost::multi_index::member<value_type, FromType, &value_type::first>
>,
boost::multi_index::ordered_non_unique<
boost::multi_index::tag<to>,
boost::multi_index::member<value_type, ToType, &value_type::second>
>
>
> type;

#endif
};
#endif 

#if BOOST_WAVE_SERIALIZATION != 0
struct load_filepos
{
static unsigned int get_line() { return 0; }
static unsigned int get_column() { return 0; }
static std::string get_file() { return "<loading-state>"; }
};
#endif

class include_paths
{
private:
typedef std::list<std::pair<boost::filesystem::path, std::string> >
include_list_type;
typedef include_list_type::value_type include_value_type;

#if BOOST_WAVE_SUPPORT_PRAGMA_ONCE != 0
typedef bidirectional_map<std::string, std::string>::type
pragma_once_set_type;
#endif

public:
include_paths()
:   was_sys_include_path(false),
current_dir(initial_path()),
current_rel_dir(initial_path())
{}

bool add_include_path(char const *path_, bool is_system = false)
{
return add_include_path(path_, (is_system || was_sys_include_path) ?
system_include_paths : user_include_paths);
}
void set_sys_include_delimiter() { was_sys_include_path = true; }
bool find_include_file (std::string &s, std::string &dir, bool is_system,
char const *current_file) const;
void set_current_directory(char const *path_);
boost::filesystem::path get_current_directory() const
{ return current_dir; }

protected:
bool find_include_file (std::string &s, std::string &dir,
include_list_type const &pathes, char const *) const;
bool add_include_path(char const *path_, include_list_type &pathes_);

private:
include_list_type user_include_paths;
include_list_type system_include_paths;
bool was_sys_include_path;          
boost::filesystem::path current_dir;
boost::filesystem::path current_rel_dir;

#if BOOST_WAVE_SUPPORT_PRAGMA_ONCE != 0
public:
bool has_pragma_once(std::string const &filename)
{
using boost::multi_index::get;
return get<from>(pragma_once_files).find(filename) != pragma_once_files.end();
}
bool add_pragma_once_header(std::string const &filename,
std::string const& guard_name)
{
typedef pragma_once_set_type::value_type value_type;
return pragma_once_files.insert(value_type(filename, guard_name)).second;
}
bool remove_pragma_once_header(std::string const& guard_name)
{
typedef pragma_once_set_type::index_iterator<to>::type to_iterator;
typedef std::pair<to_iterator, to_iterator> range_type;

range_type r = pragma_once_files.get<to>().equal_range(guard_name);
if (r.first != r.second) {
using boost::multi_index::get;
get<to>(pragma_once_files).erase(r.first, r.second);
return true;
}
return false;
}

private:
pragma_once_set_type pragma_once_files;
#endif

#if BOOST_WAVE_SERIALIZATION != 0
public:
BOOST_STATIC_CONSTANT(unsigned int, version = 0x10);
BOOST_STATIC_CONSTANT(unsigned int, version_mask = 0x0f);

private:
friend class boost::serialization::access;
template<typename Archive>
void save(Archive & ar, const unsigned int version) const
{
using namespace boost::serialization;
#if BOOST_WAVE_SUPPORT_PRAGMA_ONCE != 0
ar & make_nvp("pragma_once_files", pragma_once_files);
#endif
ar & make_nvp("user_include_paths", user_include_paths);
ar & make_nvp("system_include_paths", system_include_paths);
ar & make_nvp("was_sys_include_path", was_sys_include_path);
}
template<typename Archive>
void load(Archive & ar, const unsigned int loaded_version)
{
using namespace boost::serialization;
if (version != (loaded_version & ~version_mask)) {
BOOST_WAVE_THROW(preprocess_exception, incompatible_config,
"cpp_include_path state version", load_filepos());
return;
}

#if BOOST_WAVE_SUPPORT_PRAGMA_ONCE != 0
ar & make_nvp("pragma_once_files", pragma_once_files);
#endif
include_list_type user_paths, system_paths;
ar & make_nvp("user_include_paths", user_paths);
ar & make_nvp("system_include_paths", system_paths);

if (user_paths != user_include_paths)
{
BOOST_WAVE_THROW(preprocess_exception, incompatible_config,
"user include paths", load_filepos());
return;
}
if (system_paths != system_include_paths)
{
BOOST_WAVE_THROW(preprocess_exception, incompatible_config,
"system include paths", load_filepos());
return;
}

ar & make_nvp("was_sys_include_path", was_sys_include_path);
}
BOOST_SERIALIZATION_SPLIT_MEMBER()
#endif
};

inline
bool include_paths::add_include_path (
char const *path_, include_list_type &pathes_)
{
namespace fs = boost::filesystem;
if (path_) {
fs::path newpath = util::complete_path(create_path(path_), current_dir);

if (!fs::exists(newpath) || !fs::is_directory(newpath)) {
return false;
}

pathes_.push_back (include_value_type(newpath, path_));
return true;
}
return false;
}

inline
bool include_paths::find_include_file (std::string &s, std::string &dir,
include_list_type const &pathes, char const *current_file) const
{
namespace fs = boost::filesystem;
typedef include_list_type::const_iterator const_include_list_iter_t;

const_include_list_iter_t it = pathes.begin();
const_include_list_iter_t include_paths_end = pathes.end();

#if BOOST_WAVE_SUPPORT_INCLUDE_NEXT != 0
if (0 != current_file) {


fs::path file_path (create_path(current_file));
for (; it != include_paths_end; ++it) {
fs::path currpath (create_path((*it).first.string()));
if (std::equal(currpath.begin(), currpath.end(), file_path.begin()))
{
++it;     
break;
}
}
}
#endif

for (; it != include_paths_end; ++it) {
fs::path currpath (create_path(s));
if (!currpath.has_root_directory()) {
currpath = create_path((*it).first.string());
currpath /= create_path(s);      
}

if (fs::exists(currpath)) {
fs::path dirpath (create_path(s));
if (!dirpath.has_root_directory()) {
dirpath = create_path((*it).second);
dirpath /= create_path(s);
}

dir = dirpath.string();
s = normalize(currpath).string();    
return true;
}
}
return false;
}

inline bool
include_paths::find_include_file (std::string &s, std::string &dir,
bool is_system, char const *current_file) const
{
namespace fs = boost::filesystem;

if (!is_system) {
if (!was_sys_include_path) { 
fs::path currpath(create_path(s));
if (!currpath.has_root_directory()) {
currpath = create_path(current_dir.string());
currpath /= create_path(s);
}

if (fs::exists(currpath) && 0 == current_file) {
fs::path dirpath(create_path(s));
if (!dirpath.has_root_directory()) {
dirpath = create_path(current_rel_dir.string());
dirpath /= create_path(s);
}

dir = dirpath.string();
s = normalize(currpath).string();    
return true;
}

if (find_include_file(s, dir, user_include_paths, current_file))
return true;

}
else {
return find_include_file(s, dir, user_include_paths, current_file);
}

}

return find_include_file (s, dir, system_include_paths, current_file);
}


inline bool
as_relative_to(boost::filesystem::path const& path,
boost::filesystem::path const& base, boost::filesystem::path& result)
{
if (path.has_root_path()) {
if (path.root_path() == base.root_path())
return as_relative_to(path.relative_path(), base.relative_path(), result);

result = path;    
}
else {
if (base.has_root_path()) {
return false;
}
else {
typedef boost::filesystem::path::const_iterator path_iterator;
path_iterator path_it = path.begin();
path_iterator base_it = base.begin();
while (path_it != path.end() && base_it != base.end() ) {
if (*path_it != *base_it)
break;
++path_it; ++base_it;
}

for (; base_it != base.end(); ++base_it)
result /= "..";

for (; path_it != path.end(); ++path_it)
result /= *path_it;
}
}
return true;
}

inline
void include_paths::set_current_directory(char const *path_)
{
namespace fs = boost::filesystem;

fs::path filepath (create_path(path_));
fs::path filename = util::complete_path(filepath, current_dir);

BOOST_ASSERT(!(fs::exists(filename) && fs::is_directory(filename)));

current_rel_dir.clear();
if (!as_relative_to(branch_path(filepath), current_dir, current_rel_dir))
current_rel_dir = branch_path(filepath);
current_dir = branch_path(filename);
}

}}}   

#if BOOST_WAVE_SERIALIZATION != 0
namespace boost { namespace serialization {

template<class Archive>
inline void save (Archive & ar, boost::filesystem::path const& p,
const unsigned int )
{
using namespace boost::serialization;
std::string path_str(p.native_file_string());
ar & make_nvp("filepath", path_str);
}

template<class Archive>
inline void load (Archive & ar, boost::filesystem::path &p,
const unsigned int )
{
using namespace boost::serialization;
std::string path_str;
ar & make_nvp("filepath", path_str);
p = wave::util::create_path(path_str);
}

template<class Archive>
inline void serialize (Archive & ar, boost::filesystem::path &p,
const unsigned int file_version)
{
boost::serialization::split_free(ar, p, file_version);
}

template<class Archive>
inline void save (Archive & ar,
const typename boost::wave::util::bidirectional_map<
std::string, std::string
>::type &t,
const unsigned int )
{
boost::serialization::stl::save_collection<
Archive,
typename boost::wave::util::bidirectional_map<
std::string, std::string
>::type
>(ar, t);
}

template<class Archive>
inline void load (Archive & ar,
typename boost::wave::util::bidirectional_map<std::string, std::string>::type &t,
const unsigned int )
{
typedef typename boost::wave::util::bidirectional_map<
std::string, std::string
>::type map_type;
boost::serialization::stl::load_collection<
Archive, map_type,
boost::serialization::stl::archive_input_unique<Archive, map_type>,
boost::serialization::stl::no_reserve_imp<map_type>
>(ar, t);
}

template<class Archive>
inline void serialize (Archive & ar,
typename boost::wave::util::bidirectional_map<
std::string, std::string
>::type &t,
const unsigned int file_version)
{
boost::serialization::split_free(ar, t, file_version);
}

}}  

BOOST_CLASS_VERSION(boost::wave::util::include_paths,
boost::wave::util::include_paths::version);

#endif  

#ifdef BOOST_HAS_ABI_HEADERS
#include BOOST_ABI_SUFFIX
#endif

#endif 
