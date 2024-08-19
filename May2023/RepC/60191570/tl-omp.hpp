#ifndef TL_OMP_HPP
#define TL_OMP_HPP
#include "tl-common.hpp"
#include "cxx-utils.h"
#include "tl-scope.hpp"
#include "tl-handler.hpp"
#include "tl-dto.hpp"
#include "tl-datareference.hpp"
#include "tl-nodecl-utils.hpp"
#include "tl-pragmasupport.hpp"
#include "tl-omp-deps.hpp"
#include "tl-omp-reduction.hpp"
#include <map>
#include <set>
#include <stack>
#include <utility>
#include "tl-ompss.hpp"
namespace TL
{
namespace OpenMP
{
#define BITMAP(x) (1<<x)
enum DataSharingAttribute
{
DS_UNDEFINED = 0,
DS_SHARED = BITMAP(0),
DS_PRIVATE = BITMAP(1),
DS_FIRSTPRIVATE = BITMAP(2) | DS_PRIVATE,
DS_LASTPRIVATE = BITMAP(3) | DS_PRIVATE,
DS_FIRSTLASTPRIVATE = DS_FIRSTPRIVATE | DS_LASTPRIVATE,
DS_REDUCTION = BITMAP(4),
DS_THREADPRIVATE = BITMAP(5),
DS_COPYIN = BITMAP(6),
DS_COPYPRIVATE = BITMAP(7),
DS_AUTO = BITMAP(8),
DS_SIMD_REDUCTION = BITMAP(9),
DS_WEAKREDUCTION = BITMAP(10),
DS_TASK_REDUCTION = BITMAP(11),
DS_IN_REDUCTION = BITMAP(12),
DS_NONE = BITMAP(16),
};
enum DataSharingKind
{
DSK_NONE = 0,
DSK_EXPLICIT,
DSK_IMPLICIT,
DSK_PREDETERMINED_INDUCTION_VAR,
};
struct DataSharingValue
{
DataSharingAttribute attr;
DataSharingKind kind;
DataSharingValue()
: attr(DS_UNDEFINED), kind(DSK_NONE) { }
DataSharingValue(DataSharingAttribute a, DataSharingKind k)
: attr(a), kind(k) { }
};
enum MapDirection
{
MAP_DIR_UNDEFINED = 0,
MAP_DIR_TO = BITMAP(0),
MAP_DIR_FROM = BITMAP(1),
MAP_DIR_TOFROM = MAP_DIR_TO | MAP_DIR_FROM,
MAP_DIR_ALLOC = BITMAP(2)
};
enum MapKind
{
MAP_KIND_UNDEFINED = 0,
MAP_KIND_EXPLICIT,
MAP_KIND_IMPLICIT,
};
struct MappingValue
{
MapDirection direction;
MapKind kind;
Nodecl::NodeclBase map_expr;
MappingValue()
: direction(MAP_DIR_UNDEFINED), kind(MAP_KIND_UNDEFINED), map_expr() { }
MappingValue(MapDirection d, MapKind k)
: direction(d), kind(k), map_expr() { }
MappingValue(MapDirection d, MapKind k, Nodecl::NodeclBase e)
: direction(d), kind(k), map_expr(e) { }
};
#undef BITMAP
class LIBTL_CLASS ReductionSymbol
{
private:
Symbol _base_symbol;
Nodecl::NodeclBase _reduction_expr;
TL::Type _reduction_type;
Reduction *_reduction;
public:
ReductionSymbol(Symbol s, Nodecl::NodeclBase _expr,
TL::Type t, Reduction *reduction) :
_base_symbol(s),
_reduction_expr(_expr),
_reduction_type(t),
_reduction(reduction)
{
}
ReductionSymbol(const ReductionSymbol& red_sym) :
_base_symbol(red_sym._base_symbol),
_reduction_expr(red_sym._reduction_expr),
_reduction_type(red_sym._reduction_type),
_reduction(red_sym._reduction)
{
}
Symbol get_symbol() const
{
return _base_symbol;
}
Reduction* get_reduction() const
{
return _reduction;
}
Type get_reduction_type() const
{
return _reduction_type;
}
Nodecl::NodeclBase get_reduction_expression() const
{
return _reduction_expr;
}
};
template <typename Key, typename Value>
struct map_plus_insertion_order
{
typedef TL::ObjectList<Key> seq_t;
typedef std::map<Key, Value> map_t;
seq_t i;
map_t m;
Value &operator[](const Key &sym)
{
i.insert(sym);
return m[sym];
}
void remove(const Key &sym)
{
auto it = std::find(i.begin(), i.end(), sym);
if (it != i.end())
i.erase(it);
m.erase(sym);
}
};
class LIBTL_CLASS DataEnvironment
{
private:
int *_num_refs;
struct DataSharingAttributeInfo
{
DataSharingValue data_sharing;
std::string reason;
DataSharingAttributeInfo()
: data_sharing(), reason("(symbol has undefined data-sharing)") { }
DataSharingAttributeInfo(DataSharingValue ds,
const std::string &r)
: data_sharing(ds), reason(r) { }
};
typedef map_plus_insertion_order<TL::Symbol, DataSharingAttributeInfo> data_sharing_map_t;
data_sharing_map_t *_data_sharing;
struct MappingAttributeInfo
{
MappingValue mapping;
std::string reason;
MappingAttributeInfo()
: mapping(), reason("(symbol has undefined mapping)") { }
MappingAttributeInfo(MappingValue mv,
const std::string &r)
: mapping(mv), reason(r) { }
};
typedef map_plus_insertion_order<TL::Symbol, MappingAttributeInfo> device_mapping_map_t;
device_mapping_map_t *_device_mapping;
DataEnvironment *_enclosing;
ObjectList<ReductionSymbol> _reduction_symbols;
ObjectList<ReductionSymbol> _in_reduction_symbols;
ObjectList<ReductionSymbol> _task_reduction_symbols;
ObjectList<ReductionSymbol> _simd_reduction_symbols;
ObjectList<ReductionSymbol> _weakreduction_symbols;
ObjectList<DependencyItem> _dependency_items;
Nodecl::NodeclBase _cost_expr;
Nodecl::NodeclBase _priority_expr;
Nodecl::NodeclBase _onready_expr;
TL::OmpSs::TargetInfo _target_info;
bool _is_parallel;
bool _is_teams;
DataSharingAttributeInfo get_data_sharing_internal(Symbol sym);
DataSharingAttributeInfo get_data_sharing_info(Symbol sym, bool check_enclosing);
MappingAttributeInfo get_device_mapping_internal(Symbol sym);
MappingAttributeInfo get_device_mapping_info(Symbol sym, bool check_enclosing);
public:
DataEnvironment(DataEnvironment *enclosing);
~DataEnvironment();
DataEnvironment(const DataEnvironment& ds);
void set_data_sharing(Symbol sym,
DataSharingAttribute data_attr,
DataSharingKind kind,
const std::string& reason);
void set_data_sharing(Symbol sym,
DataSharingAttribute data_attr,
DataSharingKind kind,
DataReference data_ref,
const std::string& reason);
void remove_data_sharing(Symbol sym);
void set_reduction(const ReductionSymbol& reduction_symbol, const std::string& reason);
void set_task_reduction(const ReductionSymbol& reduction_symbol, const std::string& reason);
void set_in_reduction(const ReductionSymbol& reduction_symbol, const std::string& reason);
void set_simd_reduction(const ReductionSymbol &reduction_symbol);
void set_weakreduction(const ReductionSymbol &reduction_symbol, const std::string& reason);
DataSharingValue get_data_sharing(Symbol sym, bool check_enclosing = true);
std::string get_data_sharing_reason(Symbol sym, bool check_enclosing = true);
DataEnvironment* get_enclosing();
void get_all_symbols(DataSharingAttribute data_attr, ObjectList<Symbol> &symbols);
void get_all_symbols(ObjectList<Symbol> &symbols);
typedef std::pair<Symbol, std::string> DataSharingInfoPair;
void get_all_symbols_info(DataSharingAttribute data_attr, ObjectList<DataSharingInfoPair> &symbols);
void get_all_reduction_symbols(ObjectList<ReductionSymbol> &symbols);
void get_all_task_reduction_symbols(ObjectList<ReductionSymbol> &symbols);
void get_all_in_reduction_symbols(ObjectList<ReductionSymbol> &symbols);
void get_all_simd_reduction_symbols(ObjectList<ReductionSymbol> &symbols);
void get_all_weakreduction_symbols(ObjectList<ReductionSymbol> &symbols);
TL::OmpSs::TargetInfo& get_target_info();
void set_target_info(const TL::OmpSs::TargetInfo &target_info);
DataEnvironment& set_is_parallel(bool b);
bool get_is_parallel();
DataEnvironment& set_is_teams(bool b);
bool get_is_teams();
void add_dependence(const DependencyItem &dependency_item);
void get_all_dependences(ObjectList<DependencyItem>& dependency_items);
void set_cost_expr(const Nodecl::NodeclBase &node);
Nodecl::NodeclBase get_cost_expr();
void set_priority_expr(const Nodecl::NodeclBase &node);
Nodecl::NodeclBase get_priority_expr();
void set_onready_expr(const Nodecl::NodeclBase &node);
Nodecl::NodeclBase get_onready_expr();
void set_device_mapping(Symbol sym,
MappingValue map_value,
const std::string& reason);
MappingValue get_device_mapping(Symbol sym,
bool check_enclosing = true);
TL::ObjectList<MappingValue> get_all_device_mappings();
};
class LIBTL_CLASS Info : public Object
{
private:
DataEnvironment* _root_data_environment;
DataEnvironment* _current_data_environment;
std::map<Nodecl::NodeclBase, DataEnvironment*> _map_data_environment;
std::stack<DataEnvironment*> _stack_data_environment;
public:
Info(DataEnvironment* root_data_environment)
: _root_data_environment(root_data_environment),
_current_data_environment(root_data_environment) { }
DataEnvironment& get_new_data_environment(Nodecl::NodeclBase);
DataEnvironment& get_data_environment(Nodecl::NodeclBase);
DataEnvironment& get_root_data_environment();
DataEnvironment& get_current_data_environment();
void push_current_data_environment(DataEnvironment&);
void pop_current_data_environment();
void reset();
};
class LIBTL_CLASS OpenMPPhase : public PragmaCustomCompilerPhase
{
protected:
Nodecl::NodeclBase translation_unit;
Scope global_scope;
bool _disable_clause_warnings;
std::shared_ptr<OpenMP::Info> openmp_info;
std::shared_ptr<OmpSs::FunctionTaskSet> function_task_set;
public:
virtual void pre_run(DTO& data_flow);
virtual void run(DTO& data_flow);
virtual void init(DTO& data_flow);
OpenMPPhase()
: PragmaCustomCompilerPhase(),
_disable_clause_warnings(false)
{
}
void disable_clause_warnings(bool b);
virtual ~OpenMPPhase() { }
};
void add_extra_symbols(Nodecl::NodeclBase data_ref,
DataEnvironment& ds,
ObjectList<Symbol>& extra_symbols);
std::string data_sharing_to_string(DataSharingAttribute data_attr);
}
}
extern "C"
{
TL::CompilerPhase* give_compiler_phase_object(void);
}
#endif 
