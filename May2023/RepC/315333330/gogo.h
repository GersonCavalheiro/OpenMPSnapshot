#ifndef GO_GOGO_H
#define GO_GOGO_H
#include "go-linemap.h"
class Traverse;
class Statement_inserter;
class Type;
class Type_hash_identical;
class Type_equal;
class Type_identical;
class Typed_identifier;
class Typed_identifier_list;
class Function_type;
class Expression;
class Expression_list;
class Statement;
class Temporary_statement;
class Block;
class Function;
class Bindings;
class Bindings_snapshot;
class Package;
class Variable;
class Pointer_type;
class Struct_type;
class Struct_field;
class Struct_field_list;
class Array_type;
class Map_type;
class Channel_type;
class Interface_type;
class Named_type;
class Forward_declaration_type;
class Named_object;
class Label;
class Translate_context;
class Backend;
class Export;
class Import;
class Bexpression;
class Btype;
class Bstatement;
class Bblock;
class Bvariable;
class Blabel;
class Bfunction;
class Escape_context;
class Node;
class Import_init
{
public:
Import_init(const std::string& package_name, const std::string& init_name,
int priority)
: package_name_(package_name), init_name_(init_name), priority_(priority)
{ }
const std::string&
package_name() const
{ return this->package_name_; }
const std::string&
init_name() const
{ return this->init_name_; }
int
priority() const
{ return this->priority_; }
void
set_priority(int new_priority)
{ this->priority_ = new_priority; }
void
record_precursor_fcn(std::string init_fcn_name)
{ this->precursor_functions_.insert(init_fcn_name); }
const std::set<std::string>&
precursors() const
{ return this->precursor_functions_; }
private:
std::string package_name_;
std::string init_name_;
std::set<std::string> precursor_functions_;
int priority_;
};
struct Import_init_lt {
bool operator()(const Import_init* i1, const Import_init* i2) const
{
return i1->init_name() < i2->init_name();
}
};
class Import_init_set : public std::set<Import_init*, Import_init_lt> {
};
inline bool
priority_compare(const Import_init* i1, const Import_init* i2)
{
if (i1->priority() < i2->priority())
return true;
if (i1->priority() > i2->priority())
return false;
if (i1->package_name() != i2->package_name())
return i1->package_name() < i2->package_name();
return i1->init_name() < i2->init_name();
}
class Gogo
{
public:
Gogo(Backend* backend, Linemap *linemap, int int_type_size, int pointer_size);
Backend*
backend()
{ return this->backend_; }
Linemap*
linemap()
{ return this->linemap_; }
const std::string&
package_name() const;
void
set_package_name(const std::string&, Location);
bool
is_main_package() const;
std::string
pack_hidden_name(const std::string& name, bool is_exported) const
{
return (is_exported
? name
: '.' + this->pkgpath() + '.' + name);
}
static std::string
unpack_hidden_name(const std::string& name)
{ return name[0] != '.' ? name : name.substr(name.rfind('.') + 1); }
static bool
is_hidden_name(const std::string& name)
{ return name[0] == '.'; }
static std::string
hidden_name_pkgpath(const std::string& name)
{
go_assert(Gogo::is_hidden_name(name));
return name.substr(1, name.rfind('.') - 1);
}
static std::string
mangle_possibly_hidden_name(const std::string& name)
{ 
std::string n;
if (!Gogo::is_hidden_name(name))
n = name;
else
{
n = ".";
std::string pkgpath = Gogo::hidden_name_pkgpath(name);
n.append(Gogo::pkgpath_for_symbol(pkgpath));
n.append(1, '.');
n.append(Gogo::unpack_hidden_name(name));
}
return n;
}
static std::string
message_name(const std::string& name);
static bool
is_sink_name(const std::string& name)
{
return (name[0] == '.'
&& name[name.length() - 1] == '_'
&& name[name.length() - 2] == '.');
}
static std::string
pkgpath_for_symbol(const std::string& pkgpath);
const std::string&
pkgpath() const;
const std::string&
pkgpath_symbol() const;
void
set_pkgpath(const std::string&);
void
set_prefix(const std::string&);
bool
pkgpath_from_option() const
{ return this->pkgpath_from_option_; }
const std::string&
relative_import_path() const
{ return this->relative_import_path_; }
void
set_relative_import_path(const std::string& s)
{ this->relative_import_path_ = s; }
void
set_c_header(const std::string& s)
{ this->c_header_ = s; }
bool
check_divide_by_zero() const
{ return this->check_divide_by_zero_; }
void
set_check_divide_by_zero(bool b)
{ this->check_divide_by_zero_ = b; }
bool
check_divide_overflow() const
{ return this->check_divide_overflow_; }
void
set_check_divide_overflow(bool b)
{ this->check_divide_overflow_ = b; }
bool
compiling_runtime() const
{ return this->compiling_runtime_; }
void
set_compiling_runtime(bool b)
{ this->compiling_runtime_ = b; }
int
debug_escape_level() const
{ return this->debug_escape_level_; }
void
set_debug_escape_level(int level)
{ this->debug_escape_level_ = level; }
std::string
debug_escape_hash() const
{ return this->debug_escape_hash_; }
void
set_debug_escape_hash(const std::string& s)
{ this->debug_escape_hash_ = s; }
int64_t
nil_check_size_threshold() const
{ return this->nil_check_size_threshold_; }
void
set_nil_check_size_threshold(int64_t bytes)
{ this->nil_check_size_threshold_ = bytes; }
void
import_package(const std::string& filename, const std::string& local_name,
bool is_local_name_exported, bool must_exist, Location);
bool
in_global_scope() const;
Named_object*
lookup(const std::string&, Named_object** pfunction) const;
Named_object*
lookup_in_block(const std::string&) const;
Named_object*
lookup_global(const char*) const;
Package*
add_imported_package(const std::string& real_name, const std::string& alias,
bool is_alias_exported,
const std::string& pkgpath,
const std::string& pkgpath_symbol,
Location location,
bool* padd_to_globals);
Package*
register_package(const std::string& pkgpath,
const std::string& pkgpath_symbol, Location);
std::string
pkgpath_symbol_for_package(const std::string&);
Named_object*
start_function(const std::string& name, Function_type* type,
bool add_method_to_type, Location);
void
finish_function(Location);
Named_object*
current_function() const;
Block*
current_block();
void
start_block(Location);
Block*
finish_block(Location);
Named_object*
add_erroneous_name(const std::string& name);
Named_object*
add_unknown_name(const std::string& name, Location);
Named_object*
declare_function(const std::string&, Function_type*, Location);
Named_object*
declare_package_function(const std::string&, Function_type*, Location);
Label*
add_label_definition(const std::string&, Location);
Label*
add_label_reference(const std::string&, Location,
bool issue_goto_errors);
typedef std::pair<std::vector<Named_object*>, bool> Analysis_set;
void
add_analysis_set(const std::vector<Named_object*>& group, bool recursive)
{ this->analysis_sets_.push_back(std::make_pair(group, recursive)); }
Bindings_snapshot*
bindings_snapshot(Location);
void
add_statement(Statement*);
void
add_block(Block*, Location);
Named_object*
add_constant(const Typed_identifier&, Expression*, int iota_value);
void
add_type(const std::string&, Type*, Location);
void
add_named_type(Named_type*);
Named_object*
declare_type(const std::string&, Location);
Named_object*
declare_package_type(const std::string&, Location);
void
define_type(Named_object*, Named_type*);
Named_object*
add_variable(const std::string&, Variable*);
Named_object*
add_sink();
void
add_type_to_verify(Type* type);
void
add_dot_import_object(Named_object*);
void
add_file_block_name(const std::string& name, Location location)
{ this->file_block_names_[name] = location; }
void
add_linkname(const std::string& go_name, bool is_exported,
const std::string& ext_name, Location location);
void
mark_locals_used();
void
record_interface_type(Interface_type*);
void
set_need_init_fn()
{ this->need_init_fn_ = true; }
bool
current_file_imported_unsafe() const
{ return this->current_file_imported_unsafe_; }
void
clear_file_scope();
void
record_var_depends_on(Variable* var1, Named_object* var2)
{
go_assert(this->var_deps_.find(var1) == this->var_deps_.end());
this->var_deps_[var1] = var2;
}
Named_object*
var_depends_on(Variable* var) const
{
Var_deps::const_iterator p = this->var_deps_.find(var);
return p != this->var_deps_.end() ? p->second : NULL;
}
void
queue_specific_type_function(Type* type, Named_type* name, int64_t size,
const std::string& hash_name,
Function_type* hash_fntype,
const std::string& equal_name,
Function_type* equal_fntype);
void
write_specific_type_functions();
bool
specific_type_functions_are_written() const
{ return this->specific_type_functions_are_written_; }
void
add_gc_root(Expression* expr)
{
this->set_need_init_fn();
this->gc_roots_.push_back(expr);
}
void
traverse(Traverse*);
void
define_global_names();
void
verify_types();
void
lower_parse_tree();
void
lower_block(Named_object* function, Block*);
void
lower_expression(Named_object* function, Statement_inserter*, Expression**);
void
lower_constant(Named_object*);
void
flatten_block(Named_object* function, Block*);
void
flatten_expression(Named_object* function, Statement_inserter*, Expression**);
void
create_function_descriptors();
void
finalize_methods();
void
determine_types();
void
check_types();
void
check_types_in_block(Block*);
void
check_return_statements();
void
analyze_escape();
void
discover_analysis_sets();
void
assign_connectivity(Escape_context*, Named_object*);
void
propagate_escape(Escape_context*, Node*);
void
tag_function(Escape_context*, Named_object*);
void
reclaim_escape_nodes();
void
do_exports();
void
add_import_init_fn(const std::string& package_name,
const std::string& init_name, int prio);
Import_init*
lookup_init(const std::string& init_name);
void
remove_shortcuts();
void
order_evaluations();
void
add_write_barriers();
bool
assign_needs_write_barrier(Expression* lhs);
Statement*
assign_with_write_barrier(Function*, Block*, Statement_inserter*,
Expression* lhs, Expression* rhs, Location);
void
flatten();
void
build_recover_thunks();
static Named_object*
declare_builtin_rf_address(const char* name);
void
simplify_thunk_statements();
void
dump_ast(const char* basename);
void
dump_call_graph(const char* basename);
void
dump_connection_graphs(const char* basename);
void
convert_named_types();
void
convert_named_types_in_bindings(Bindings*);
bool
named_types_are_converted() const
{ return this->named_types_are_converted_; }
void
check_self_dep(Named_object*);
void
write_globals();
Expression*
runtime_error(int code, Location);
void
build_interface_method_tables();
Expression*
allocate_memory(Type *type, Location);
std::string
function_asm_name(const std::string& go_name, const Package*,
const Type* receiver);
std::string
function_descriptor_name(Named_object*);
std::string
stub_method_name(const Package*, const std::string& method_name);
void
specific_type_function_names(const Type*, const Named_type*,
std::string* hash_name,
std::string* equal_name);
std::string
global_var_asm_name(const std::string& go_name, const Package*);
static std::string
erroneous_name();
static bool
is_erroneous_name(const std::string&);
std::string
thunk_name();
static bool
is_thunk(const Named_object*);
std::string
init_function_name();
std::string
nested_function_name(Named_object* enclosing);
std::string
sink_function_name();
std::string
redefined_function_name();
std::string
recover_thunk_name(const std::string& name, const Type* rtype);
std::string
gc_root_name();
std::string
initializer_name();
std::string
map_zero_value_name();
const std::string&
get_init_fn_name();
std::string
type_descriptor_name(Type*, Named_type*);
std::string
gc_symbol_name(Type*);
std::string
ptrmask_symbol_name(const std::string& ptrmask_sym_name);
std::string
interface_method_table_name(Interface_type*, Type*, bool is_pointer);
static bool
is_special_name(const std::string& name);
private:
struct Open_function
{
Named_object* function;
std::vector<Block*> blocks;
};
typedef std::vector<Open_function> Open_functions;
void
import_unsafe(const std::string&, bool is_exported, Location);
Bindings*
current_bindings();
const Bindings*
current_bindings() const;
void
write_c_header();
Named_object*
initialization_function_decl();
Named_object*
create_initialization_function(Named_object* fndecl, Bstatement* code_stmt);
void
init_imports(std::vector<Bstatement*>&, Bfunction* bfunction);
void
register_gc_vars(const std::vector<Named_object*>&,
std::vector<Bstatement*>&,
Bfunction* init_bfunction);
Named_object*
write_barrier_variable();
Statement*
check_write_barrier(Block*, Statement*, Statement*);
typedef std::map<std::string, Package*> Imports;
typedef std::map<std::string, Package*> Packages;
typedef std::map<Variable*, Named_object*> Var_deps;
typedef Unordered_map(std::string, Location) File_block_names;
struct Specific_type_function
{
Type* type;
Named_type* name;
int64_t size;
std::string hash_name;
Function_type* hash_fntype;
std::string equal_name;
Function_type* equal_fntype;
Specific_type_function(Type* atype, Named_type* aname, int64_t asize,
const std::string& ahash_name,
Function_type* ahash_fntype,
const std::string& aequal_name,
Function_type* aequal_fntype)
: type(atype), name(aname), size(asize), hash_name(ahash_name),
hash_fntype(ahash_fntype), equal_name(aequal_name),
equal_fntype(aequal_fntype)
{ }
};
void
recompute_init_priorities();
void
update_init_priority(Import_init* ii,
std::set<const Import_init *>* visited);
Backend* backend_;
Linemap* linemap_;
Package* package_;
Open_functions functions_;
Bindings* globals_;
File_block_names file_block_names_;
Imports imports_;
bool imported_unsafe_;
bool current_file_imported_unsafe_;
Packages packages_;
std::vector<Named_object*> init_functions_;
Var_deps var_deps_;
bool need_init_fn_;
std::string init_fn_name_;
Import_init_set imported_init_fns_;
std::string pkgpath_;
std::string pkgpath_symbol_;
std::string prefix_;
bool pkgpath_set_;
bool pkgpath_from_option_;
bool prefix_from_option_;
std::string relative_import_path_;
std::string c_header_;
bool check_divide_by_zero_;
bool check_divide_overflow_;
bool compiling_runtime_;
int debug_escape_level_;
std::string debug_escape_hash_;
int64_t nil_check_size_threshold_;
std::vector<Type*> verify_types_;
std::vector<Interface_type*> interface_types_;
std::vector<Specific_type_function*> specific_type_functions_;
bool specific_type_functions_are_written_;
bool named_types_are_converted_;
std::vector<Analysis_set> analysis_sets_;
std::vector<Expression*> gc_roots_;
};
class Block
{
public:
Block(Block* enclosing, Location);
const Block*
enclosing() const
{ return this->enclosing_; }
Bindings*
bindings()
{ return this->bindings_; }
const Bindings*
bindings() const
{ return this->bindings_; }
const std::vector<Statement*>*
statements() const
{ return &this->statements_; }
Location
start_location() const
{ return this->start_location_; }
Location
end_location() const
{ return this->end_location_; }
void
add_statement(Statement*);
void
add_statement_at_front(Statement*);
void
replace_statement(size_t index, Statement*);
void
insert_statement_before(size_t index, Statement*);
void
insert_statement_after(size_t index, Statement*);
void
set_end_location(Location location)
{ this->end_location_ = location; }
int
traverse(Traverse*);
void
determine_types();
bool
may_fall_through() const;
Bblock*
get_backend(Translate_context*);
typedef std::vector<Statement*>::iterator iterator;
iterator
begin()
{ return this->statements_.begin(); }
iterator
end()
{ return this->statements_.end(); }
private:
Block* enclosing_;
std::vector<Statement*> statements_;
Bindings* bindings_;
Location start_location_;
Location end_location_;
};
class Function
{
public:
Function(Function_type* type, Named_object*, Block*, Location);
Function_type*
type() const
{ return this->type_; }
Named_object*
enclosing() const
{ return this->enclosing_; }
void
set_enclosing(Named_object* enclosing)
{
go_assert(this->enclosing_ == NULL);
this->enclosing_ = enclosing;
}
typedef std::vector<Named_object*> Results;
void
create_result_variables(Gogo*);
void
update_result_variables();
Results*
result_variables()
{ return this->results_; }
bool
is_sink() const
{ return this->is_sink_; }
void
set_is_sink()
{ this->is_sink_ = true; }
bool
results_are_named() const
{ return this->results_are_named_; }
const std::string&
asm_name() const
{ return this->asm_name_; }
void
set_asm_name(const std::string& asm_name)
{ this->asm_name_ = asm_name; }
unsigned int
pragmas() const
{ return this->pragmas_; }
void
set_pragmas(unsigned int pragmas)
{
this->pragmas_ = pragmas;
}
unsigned int
next_nested_function_index()
{
++this->nested_functions_;
return this->nested_functions_;
}
bool
nointerface() const;
void
set_nointerface();
void
set_is_unnamed_type_stub_method()
{
go_assert(this->is_method());
this->is_unnamed_type_stub_method_ = true;
}
size_t
closure_field_count() const
{ return this->closure_fields_.size(); }
void
add_closure_field(Named_object* var, Location loc)
{ this->closure_fields_.push_back(std::make_pair(var, loc)); }
bool
needs_closure() const
{ return !this->closure_fields_.empty(); }
Named_object*
closure_var();
void
set_closure_var(Named_object* v)
{
go_assert(this->closure_var_ == NULL);
this->closure_var_ = v;
}
Named_object*
enclosing_var(unsigned int index)
{
go_assert(index < this->closure_fields_.size());
return closure_fields_[index].first;
}
void
set_closure_type();
Block*
block() const
{ return this->block_; }
Location
location() const
{ return this->location_; }
bool
is_method() const;
Label*
add_label_definition(Gogo*, const std::string& label_name, Location);
Label*
add_label_reference(Gogo*, const std::string& label_name,
Location, bool issue_goto_errors);
void
check_labels() const;
unsigned int
new_local_type_index()
{ return this->local_type_count_++; }
bool
calls_recover() const
{ return this->calls_recover_; }
void
set_calls_recover()
{ this->calls_recover_ = true; }
bool
is_recover_thunk() const
{ return this->is_recover_thunk_; }
void
set_is_recover_thunk()
{ this->is_recover_thunk_ = true; }
bool
has_recover_thunk() const
{ return this->has_recover_thunk_; }
void
set_has_recover_thunk()
{ this->has_recover_thunk_ = true; }
void
set_calls_defer_retaddr()
{ this->calls_defer_retaddr_ = true; }
bool
is_type_specific_function()
{ return this->is_type_specific_function_; }
void
set_is_type_specific_function()
{ this->is_type_specific_function_ = true; }
void
set_in_unique_section()
{ this->in_unique_section_ = true; }
void
swap_for_recover(Function *);
int
traverse(Traverse*);
void
determine_types();
Expression*
descriptor(Gogo*, Named_object*);
void
set_descriptor(Expression* descriptor)
{
go_assert(this->descriptor_ == NULL);
this->descriptor_ = descriptor;
}
Bfunction*
get_or_make_decl(Gogo*, Named_object*);
Bfunction*
get_decl() const;
void
build(Gogo*, Named_object*);
Bstatement*
return_value(Gogo*, Named_object*, Location) const;
Expression*
defer_stack(Location);
void
export_func(Export*, const std::string& name) const;
static void
export_func_with_type(Export*, const std::string& name,
const Function_type*, bool nointerface);
static void
import_func(Import*, std::string* pname, Typed_identifier** receiver,
Typed_identifier_list** pparameters,
Typed_identifier_list** presults, bool* is_varargs,
bool* nointerface);
private:
typedef Unordered_map(std::string, Label*) Labels;
void
build_defer_wrapper(Gogo*, Named_object*, Bstatement**, Bstatement**);
typedef std::vector<std::pair<Named_object*,
Location> > Closure_fields;
Function_type* type_;
Named_object* enclosing_;
Results* results_;
Closure_fields closure_fields_;
Named_object* closure_var_;
Block* block_;
Location location_;
Labels labels_;
unsigned int local_type_count_;
std::string asm_name_;
Expression* descriptor_;
Bfunction* fndecl_;
Temporary_statement* defer_stack_;
unsigned int pragmas_;
unsigned int nested_functions_;
bool is_sink_ : 1;
bool results_are_named_ : 1;
bool is_unnamed_type_stub_method_ : 1;
bool calls_recover_ : 1;
bool is_recover_thunk_ : 1;
bool has_recover_thunk_ : 1;
bool calls_defer_retaddr_ : 1;
bool is_type_specific_function_ : 1;
bool in_unique_section_ : 1;
};
class Bindings_snapshot
{
public:
Bindings_snapshot(const Block*, Location);
void
check_goto_from(const Block* b, Location);
void
check_goto_to(const Block* b);
private:
bool
check_goto_block(Location, const Block*, const Block*, size_t*);
void
check_goto_defs(Location, const Block*, size_t, size_t);
const Block* block_;
std::vector<size_t> counts_;
Location location_;
};
class Function_declaration
{
public:
Function_declaration(Function_type* fntype, Location location)
: fntype_(fntype), location_(location), asm_name_(), descriptor_(NULL),
fndecl_(NULL), pragmas_(0)
{ }
Function_type*
type() const
{ return this->fntype_; }
Location
location() const
{ return this->location_; }
bool
is_method() const;
const std::string&
asm_name() const
{ return this->asm_name_; }
void
set_asm_name(const std::string& asm_name)
{ this->asm_name_ = asm_name; }
unsigned int
pragmas() const
{ return this->pragmas_; }
void
set_pragmas(unsigned int pragmas)
{
this->pragmas_ = pragmas;
}
bool
nointerface() const;
void
set_nointerface();
Expression*
descriptor(Gogo*, Named_object*);
bool
has_descriptor() const
{ return this->descriptor_ != NULL; }
Bfunction*
get_or_make_decl(Gogo*, Named_object*);
void
build_backend_descriptor(Gogo*);
void
export_func(Export* exp, const std::string& name) const
{
Function::export_func_with_type(exp, name, this->fntype_,
this->is_method() && this->nointerface());
}
void
check_types() const;
private:
Function_type* fntype_;
Location location_;
std::string asm_name_;
Expression* descriptor_;
Bfunction* fndecl_;
unsigned int pragmas_;
};
class Variable
{
public:
Variable(Type*, Expression*, bool is_global, bool is_parameter,
bool is_receiver, Location);
Type*
type();
Type*
type() const;
bool
has_type() const;
Expression*
init() const
{ return this->init_; }
bool
has_pre_init() const
{ return this->preinit_ != NULL; }
Block*
preinit() const
{ return this->preinit_; }
bool
is_global() const
{ return this->is_global_; }
bool
is_parameter() const
{ return this->is_parameter_; }
bool
is_closure() const
{ return this->is_closure_; }
void
set_is_closure()
{
this->is_closure_ = true;
}
bool
is_receiver() const
{ return this->is_receiver_; }
void
set_is_receiver()
{
go_assert(this->is_parameter_);
this->is_receiver_ = true;
}
void
set_is_not_receiver()
{
go_assert(this->is_parameter_);
this->is_receiver_ = false;
}
bool
is_varargs_parameter() const
{ return this->is_varargs_parameter_; }
bool
is_address_taken() const
{ return this->is_address_taken_; }
bool
is_in_heap() const
{
return this->is_address_taken_ 
&& this->escapes_
&& !this->is_global_;
}
void
set_address_taken()
{ this->is_address_taken_ = true; }
bool
is_non_escaping_address_taken() const
{ return this->is_non_escaping_address_taken_; }
void
set_non_escaping_address_taken()
{ this->is_non_escaping_address_taken_ = true; }
bool
escapes()
{ return this->escapes_; }
void
set_does_not_escape()
{ this->escapes_ = false; }
Location
location() const
{ return this->location_; }
void
set_is_varargs_parameter()
{
go_assert(this->is_parameter_);
this->is_varargs_parameter_ = true;
}
bool
is_used() const
{ return this->is_used_; }
void
set_is_used()
{ this->is_used_ = true; }
void
clear_init()
{ this->init_ = NULL; }
void
set_init(Expression* init)
{ this->init_ = init; }
Block*
preinit_block(Gogo*);
void
add_preinit_statement(Gogo*, Statement*);
void
lower_init_expression(Gogo*, Named_object*, Statement_inserter*);
void
flatten_init_expression(Gogo*, Named_object*, Statement_inserter*);
void
set_type_from_init_tuple()
{ this->type_from_init_tuple_ = true; }
void
set_type_from_range_index()
{ this->type_from_range_index_ = true; }
void
set_type_from_range_value()
{ this->type_from_range_value_ = true; }
void
set_type_from_chan_element()
{ this->type_from_chan_element_ = true; }
void
clear_type_from_chan_element()
{
go_assert(this->type_from_chan_element_);
this->type_from_chan_element_ = false;
}
bool
is_type_switch_var() const
{ return this->is_type_switch_var_; }
void
set_is_type_switch_var()
{ this->is_type_switch_var_ = true; }
void
set_in_unique_section()
{
go_assert(this->is_global_);
this->in_unique_section_ = true;
}
Statement*
toplevel_decl()
{ return this->toplevel_decl_; }
void
set_toplevel_decl(Statement* s)
{
go_assert(!this->is_global_ && !this->is_parameter_ && !this->is_receiver_);
this->toplevel_decl_ = s;
}
int
traverse_expression(Traverse*, unsigned int traverse_mask);
void
determine_type();
Bvariable*
get_backend_variable(Gogo*, Named_object*, const Package*,
const std::string&);
Bexpression*
get_init(Gogo*, Named_object* function);
Bstatement*
get_init_block(Gogo*, Named_object* function, Bvariable* decl);
void
export_var(Export*, const std::string& name) const;
static void
import_var(Import*, std::string* pname, Type** ptype);
private:
Type*
type_from_tuple(Expression*, bool) const;
Type*
type_from_range(Expression*, bool, bool) const;
Type*
type_from_chan_element(Expression*, bool) const;
Type* type_;
Expression* init_;
Block* preinit_;
Location location_;
Bvariable* backend_;
bool is_global_ : 1;
bool is_parameter_ : 1;
bool is_closure_ : 1;
bool is_receiver_ : 1;
bool is_varargs_parameter_ : 1;
bool is_used_ : 1;
bool is_address_taken_ : 1;
bool is_non_escaping_address_taken_ : 1;
bool seen_ : 1;
bool init_is_lowered_ : 1;
bool init_is_flattened_ : 1;
bool type_from_init_tuple_ : 1;
bool type_from_range_index_ : 1;
bool type_from_range_value_ : 1;
bool type_from_chan_element_ : 1;
bool is_type_switch_var_ : 1;
bool determined_type_ : 1;
bool in_unique_section_ : 1;
bool escapes_ : 1;
Statement* toplevel_decl_;
};
class Result_variable
{
public:
Result_variable(Type* type, Function* function, int index,
Location location)
: type_(type), function_(function), index_(index), location_(location),
backend_(NULL), is_address_taken_(false),
is_non_escaping_address_taken_(false), escapes_(true)
{ }
Type*
type() const
{ return this->type_; }
Function*
function() const
{ return this->function_; }
int
index() const
{ return this->index_; }
Location
location() const
{ return this->location_; }
bool
is_address_taken() const
{ return this->is_address_taken_; }
void
set_address_taken()
{ this->is_address_taken_ = true; }
bool
is_non_escaping_address_taken() const
{ return this->is_non_escaping_address_taken_; }
void
set_non_escaping_address_taken()
{ this->is_non_escaping_address_taken_ = true; }
bool
escapes()
{ return this->escapes_; }
void
set_does_not_escape()
{ this->escapes_ = false; }
bool
is_in_heap() const
{
return this->is_address_taken_
&& this->escapes_;
}
void
set_function(Function* function)
{ this->function_ = function; }
Bvariable*
get_backend_variable(Gogo*, Named_object*, const std::string&);
private:
Type* type_;
Function* function_;
int index_;
Location location_;
Bvariable* backend_;
bool is_address_taken_;
bool is_non_escaping_address_taken_;
bool escapes_;
};
class Named_constant
{
public:
Named_constant(Type* type, Expression* expr, int iota_value,
Location location)
: type_(type), expr_(expr), iota_value_(iota_value), location_(location),
lowering_(false), is_sink_(false), bconst_(NULL)
{ }
Type*
type() const
{ return this->type_; }
void
set_type(Type* t);
Expression*
expr() const
{ return this->expr_; }
int
iota_value() const
{ return this->iota_value_; }
Location
location() const
{ return this->location_; }
bool
lowering() const
{ return this->lowering_; }
void
set_lowering()
{ this->lowering_ = true; }
void
clear_lowering()
{ this->lowering_ = false; }
bool
is_sink() const
{ return this->is_sink_; }
void
set_is_sink()
{ this->is_sink_ = true; }
int
traverse_expression(Traverse*);
void
determine_type();
void
set_error();
void
export_const(Export*, const std::string& name) const;
static void
import_const(Import*, std::string*, Type**, Expression**);
Bexpression*
get_backend(Gogo*, Named_object*);
private:
Type* type_;
Expression* expr_;
int iota_value_;
Location location_;
bool lowering_;
bool is_sink_;
Bexpression* bconst_;
};
class Type_declaration
{
public:
Type_declaration(Location location)
: location_(location), in_function_(NULL), in_function_index_(0),
methods_(), issued_warning_(false)
{ }
Location
location() const
{ return this->location_; }
Named_object*
in_function(unsigned int* pindex)
{
*pindex = this->in_function_index_;
return this->in_function_;
}
void
set_in_function(Named_object* f, unsigned int index)
{
this->in_function_ = f;
this->in_function_index_ = index;
}
Named_object*
add_method(const std::string& name, Function* function);
Named_object*
add_method_declaration(const std::string& name, Package*,
Function_type* type, Location location);
void
add_existing_method(Named_object* no)
{ this->methods_.push_back(no); }
bool
has_methods() const;
const std::vector<Named_object*>*
methods() const
{ return &this->methods_; }
void
define_methods(Named_type*);
bool
using_type();
private:
Location location_;
Named_object* in_function_;
unsigned int in_function_index_;
std::vector<Named_object*> methods_;
bool issued_warning_;
};
class Unknown_name
{
public:
Unknown_name(Location location)
: location_(location), real_named_object_(NULL)
{ }
Location
location() const
{ return this->location_; }
Named_object*
real_named_object() const
{ return this->real_named_object_; }
void
set_real_named_object(Named_object* no);
private:
Location location_;
Named_object*
real_named_object_;
};
class Named_object
{
public:
enum Classification
{
NAMED_OBJECT_UNINITIALIZED,
NAMED_OBJECT_ERRONEOUS,
NAMED_OBJECT_UNKNOWN,
NAMED_OBJECT_CONST,
NAMED_OBJECT_TYPE,
NAMED_OBJECT_TYPE_DECLARATION,
NAMED_OBJECT_VAR,
NAMED_OBJECT_RESULT_VAR,
NAMED_OBJECT_SINK,
NAMED_OBJECT_FUNC,
NAMED_OBJECT_FUNC_DECLARATION,
NAMED_OBJECT_PACKAGE
};
Classification
classification() const
{ return this->classification_; }
bool
is_erroneous() const
{ return this->classification_ == NAMED_OBJECT_ERRONEOUS; }
bool
is_unknown() const
{ return this->classification_ == NAMED_OBJECT_UNKNOWN; }
bool
is_const() const
{ return this->classification_ == NAMED_OBJECT_CONST; }
bool
is_type() const
{ return this->classification_ == NAMED_OBJECT_TYPE; }
bool
is_type_declaration() const
{ return this->classification_ == NAMED_OBJECT_TYPE_DECLARATION; }
bool
is_variable() const
{ return this->classification_ == NAMED_OBJECT_VAR; }
bool
is_result_variable() const
{ return this->classification_ == NAMED_OBJECT_RESULT_VAR; }
bool
is_sink() const
{ return this->classification_ == NAMED_OBJECT_SINK; }
bool
is_function() const
{ return this->classification_ == NAMED_OBJECT_FUNC; }
bool
is_function_declaration() const
{ return this->classification_ == NAMED_OBJECT_FUNC_DECLARATION; }
bool
is_package() const
{ return this->classification_ == NAMED_OBJECT_PACKAGE; }
static Named_object*
make_erroneous_name(const std::string& name)
{ return new Named_object(name, NULL, NAMED_OBJECT_ERRONEOUS); }
static Named_object*
make_unknown_name(const std::string& name, Location);
static Named_object*
make_constant(const Typed_identifier&, const Package*, Expression*,
int iota_value);
static Named_object*
make_type(const std::string&, const Package*, Type*, Location);
static Named_object*
make_type_declaration(const std::string&, const Package*, Location);
static Named_object*
make_variable(const std::string&, const Package*, Variable*);
static Named_object*
make_result_variable(const std::string&, Result_variable*);
static Named_object*
make_sink();
static Named_object*
make_function(const std::string&, const Package*, Function*);
static Named_object*
make_function_declaration(const std::string&, const Package*, Function_type*,
Location);
static Named_object*
make_package(const std::string& alias, Package* package);
Unknown_name*
unknown_value()
{
go_assert(this->classification_ == NAMED_OBJECT_UNKNOWN);
return this->u_.unknown_value;
}
const Unknown_name*
unknown_value() const
{
go_assert(this->classification_ == NAMED_OBJECT_UNKNOWN);
return this->u_.unknown_value;
}
Named_constant*
const_value()
{
go_assert(this->classification_ == NAMED_OBJECT_CONST);
return this->u_.const_value;
}
const Named_constant*
const_value() const
{
go_assert(this->classification_ == NAMED_OBJECT_CONST);
return this->u_.const_value;
}
Named_type*
type_value()
{
go_assert(this->classification_ == NAMED_OBJECT_TYPE);
return this->u_.type_value;
}
const Named_type*
type_value() const
{
go_assert(this->classification_ == NAMED_OBJECT_TYPE);
return this->u_.type_value;
}
Type_declaration*
type_declaration_value()
{
go_assert(this->classification_ == NAMED_OBJECT_TYPE_DECLARATION);
return this->u_.type_declaration;
}
const Type_declaration*
type_declaration_value() const
{
go_assert(this->classification_ == NAMED_OBJECT_TYPE_DECLARATION);
return this->u_.type_declaration;
}
Variable*
var_value()
{
go_assert(this->classification_ == NAMED_OBJECT_VAR);
return this->u_.var_value;
}
const Variable*
var_value() const
{
go_assert(this->classification_ == NAMED_OBJECT_VAR);
return this->u_.var_value;
}
Result_variable*
result_var_value()
{
go_assert(this->classification_ == NAMED_OBJECT_RESULT_VAR);
return this->u_.result_var_value;
}
const Result_variable*
result_var_value() const
{
go_assert(this->classification_ == NAMED_OBJECT_RESULT_VAR);
return this->u_.result_var_value;
}
Function*
func_value()
{
go_assert(this->classification_ == NAMED_OBJECT_FUNC);
return this->u_.func_value;
}
const Function*
func_value() const
{
go_assert(this->classification_ == NAMED_OBJECT_FUNC);
return this->u_.func_value;
}
Function_declaration*
func_declaration_value()
{
go_assert(this->classification_ == NAMED_OBJECT_FUNC_DECLARATION);
return this->u_.func_declaration_value;
}
const Function_declaration*
func_declaration_value() const
{
go_assert(this->classification_ == NAMED_OBJECT_FUNC_DECLARATION);
return this->u_.func_declaration_value;
}
Package*
package_value()
{
go_assert(this->classification_ == NAMED_OBJECT_PACKAGE);
return this->u_.package_value;
}
const Package*
package_value() const
{
go_assert(this->classification_ == NAMED_OBJECT_PACKAGE);
return this->u_.package_value;
}
const std::string&
name() const
{ return this->name_; }
std::string
message_name() const;
const Package*
package() const
{ return this->package_; }
Named_object*
resolve()
{
Named_object* ret = this;
if (this->is_unknown())
{
Named_object* r = this->unknown_value()->real_named_object();
if (r != NULL)
ret = r;
}
return ret;
}
const Named_object*
resolve() const
{
const Named_object* ret = this;
if (this->is_unknown())
{
const Named_object* r = this->unknown_value()->real_named_object();
if (r != NULL)
ret = r;
}
return ret;
}
Location
location() const;
Bvariable*
get_backend_variable(Gogo*, Named_object* function);
std::string
get_id(Gogo*);
void
get_backend(Gogo*, std::vector<Bexpression*>&, std::vector<Btype*>&,
std::vector<Bfunction*>&);
void
set_type_value(Named_type*);
void
set_function_value(Function*);
void
declare_as_type();
void
export_named_object(Export*) const;
void
set_is_redefinition()
{ this->is_redefinition_ = true; }
bool
is_redefinition() const
{ return this->is_redefinition_; }
private:
Named_object(const std::string&, const Package*, Classification);
std::string name_;
const Package* package_;
Classification classification_;
union
{
Unknown_name* unknown_value;
Named_constant* const_value;
Named_type* type_value;
Type_declaration* type_declaration;
Variable* var_value;
Result_variable* result_var_value;
Function* func_value;
Function_declaration* func_declaration_value;
Package* package_value;
} u_;
bool is_redefinition_;
};
class Bindings
{
public:
typedef Unordered_map(std::string, Named_object*) Contour;
Bindings(Bindings* enclosing);
Named_object*
add_erroneous_name(const std::string& name)
{ return this->add_named_object(Named_object::make_erroneous_name(name)); }
Named_object*
add_unknown_name(const std::string& name, Location location)
{
return this->add_named_object(Named_object::make_unknown_name(name,
location));
}
Named_object*
add_constant(const Typed_identifier& tid, const Package* package,
Expression* expr, int iota_value)
{
return this->add_named_object(Named_object::make_constant(tid, package,
expr,
iota_value));
}
Named_object*
add_type(const std::string& name, const Package* package, Type* type,
Location location)
{
return this->add_named_object(Named_object::make_type(name, package, type,
location));
}
Named_object*
add_named_type(Named_type* named_type);
Named_object*
add_type_declaration(const std::string& name, const Package* package,
Location location)
{
Named_object* no = Named_object::make_type_declaration(name, package,
location);
return this->add_named_object(no);
}
Named_object*
add_variable(const std::string& name, const Package* package,
Variable* variable)
{
return this->add_named_object(Named_object::make_variable(name, package,
variable));
}
Named_object*
add_result_variable(const std::string& name, Result_variable* result)
{
return this->add_named_object(Named_object::make_result_variable(name,
result));
}
Named_object*
add_function(const std::string& name, const Package*, Function* function);
Named_object*
add_function_declaration(const std::string& name, const Package* package,
Function_type* type, Location location);
Named_object*
add_package(const std::string& alias, Package* package)
{
Named_object* no = Named_object::make_package(alias, package);
return this->add_named_object(no);
}
void
define_type(Named_object*, Named_type*);
void
add_method(Named_object*);
Named_object*
add_named_object(Named_object* no)
{ return this->add_named_object_to_contour(&this->bindings_, no); }
void
clear_file_scope(Gogo*);
Named_object*
lookup(const std::string&) const;
Named_object*
lookup_local(const std::string&) const;
void
remove_binding(Named_object*);
void
mark_locals_used();
int
traverse(Traverse*, bool is_global);
typedef std::vector<Named_object*>::const_iterator
const_definitions_iterator;
const_definitions_iterator
begin_definitions() const
{ return this->named_objects_.begin(); }
const_definitions_iterator
end_definitions() const
{ return this->named_objects_.end(); }
size_t
size_definitions() const
{ return this->named_objects_.size(); }
bool
empty_definitions() const
{ return this->named_objects_.empty(); }
typedef Contour::const_iterator const_declarations_iterator;
const_declarations_iterator
begin_declarations() const
{ return this->bindings_.begin(); }
const_declarations_iterator
end_declarations() const
{ return this->bindings_.end(); }
size_t
size_declarations() const
{ return this->bindings_.size(); }
bool
empty_declarations() const
{ return this->bindings_.empty(); }
Named_object*
first_declaration()
{ return this->bindings_.empty() ? NULL : this->bindings_.begin()->second; }
private:
Named_object*
add_named_object_to_contour(Contour*, Named_object*);
Named_object*
new_definition(Named_object*, Named_object*);
Bindings* enclosing_;
std::vector<Named_object*> named_objects_;
Contour bindings_;
};
class Label
{
public:
Label(const std::string& name)
: name_(name), location_(Linemap::unknown_location()), snapshot_(NULL),
refs_(), is_used_(false), blabel_(NULL), depth_(DEPTH_UNKNOWN)
{ }
const std::string&
name() const
{ return this->name_; }
bool
is_defined() const
{ return !Linemap::is_unknown_location(this->location_); }
bool
is_used() const
{ return this->is_used_; }
void
set_is_used()
{ this->is_used_ = true; }
bool
looping() const
{ return this->depth_ == DEPTH_LOOPING; }
void
set_looping()
{ this->depth_ = DEPTH_LOOPING; }
bool
nonlooping() const
{ return this->depth_ == DEPTH_NONLOOPING; }
void
set_nonlooping()
{ this->depth_ = DEPTH_NONLOOPING; }
Location
location() const
{ return this->location_; }
Bindings_snapshot*
snapshot() const
{ return this->snapshot_; }
void
add_snapshot_ref(Bindings_snapshot* snapshot)
{
go_assert(Linemap::is_unknown_location(this->location_));
this->refs_.push_back(snapshot);
}
const std::vector<Bindings_snapshot*>&
refs() const
{ return this->refs_; }
void
clear_refs();
void
define(Location location, Bindings_snapshot* snapshot)
{
if (this->is_dummy_label())
return;
go_assert(Linemap::is_unknown_location(this->location_)
&& this->snapshot_ == NULL);
this->location_ = location;
this->snapshot_ = snapshot;
}
Blabel*
get_backend_label(Translate_context*);
Bexpression*
get_addr(Translate_context*, Location location);
static Label*
create_dummy_label();
bool
is_dummy_label() const
{ return this->name_ == "_"; }
enum Loop_depth
{
DEPTH_UNKNOWN,
DEPTH_NONLOOPING,
DEPTH_LOOPING
};
private:
std::string name_;
Location location_;
Bindings_snapshot* snapshot_;
std::vector<Bindings_snapshot*> refs_;
bool is_used_;
Blabel* blabel_;
Loop_depth depth_;
};
class Unnamed_label
{
public:
Unnamed_label(Location location)
: location_(location), derived_from_(NULL), blabel_(NULL)
{ }
Location
location() const
{ return this->location_; }
void
set_location(Location location)
{ this->location_ = location; }
Statement*
derived_from() const
{ return this->derived_from_; }
void
set_derived_from(Statement* s)
{ this->derived_from_ = s; }
Bstatement*
get_definition(Translate_context*);
Bstatement*
get_goto(Translate_context*, Location location);
private:
Blabel*
get_blabel(Translate_context*);
Location location_;
Statement* derived_from_;
Blabel* blabel_;
};
class Package_alias
{
public:
Package_alias(Location location)
: location_(location), used_(0)
{ }
Location
location()
{ return this->location_; }
size_t
used() const
{ return this->used_; }
void
note_usage()
{ this->used_++; }
private:
Location location_;
size_t used_;
};
class Package
{
public:
Package(const std::string& pkgpath, const std::string& pkgpath_symbol,
Location location);
const std::string&
pkgpath() const
{ return this->pkgpath_; }
std::string
pkgpath_symbol() const;
void
set_pkgpath_symbol(const std::string&);
Location
location() const
{ return this->location_; }
bool
has_package_name() const
{ return !this->package_name_.empty(); }
const std::string&
package_name() const
{
go_assert(!this->package_name_.empty());
return this->package_name_;
}
Bindings*
bindings()
{ return this->bindings_; }
typedef std::map<std::string, Package_alias*> Aliases;
const Aliases&
aliases() const
{ return this->aliases_; }
void
note_usage(const std::string& alias) const;
void
note_fake_usage(Expression* usage) const
{ this->fake_uses_.insert(usage); }
void
forget_usage(Expression* usage) const;
void
clear_used();
Named_object*
lookup(const std::string& name) const
{ return this->bindings_->lookup(name); }
void
set_package_name(const std::string& name, Location);
void
set_location(Location location)
{ this->location_ = location; }
Package_alias*
add_alias(const std::string& alias, Location);
Named_object*
add_constant(const Typed_identifier& tid, Expression* expr)
{ return this->bindings_->add_constant(tid, this, expr, 0); }
Named_object*
add_type(const std::string& name, Type* type, Location location)
{ return this->bindings_->add_type(name, this, type, location); }
Named_object*
add_type_declaration(const std::string& name, Location location)
{ return this->bindings_->add_type_declaration(name, this, location); }
Named_object*
add_variable(const std::string& name, Variable* variable)
{ return this->bindings_->add_variable(name, this, variable); }
Named_object*
add_function_declaration(const std::string& name, Function_type* type,
Location loc)
{ return this->bindings_->add_function_declaration(name, this, type, loc); }
void
determine_types();
private:
std::string pkgpath_;
std::string pkgpath_symbol_;
std::string package_name_;
Bindings* bindings_;
Location location_;
Aliases aliases_;
mutable std::set<Expression*> fake_uses_;
};
const int TRAVERSE_CONTINUE = -1;
const int TRAVERSE_EXIT = 0;
const int TRAVERSE_SKIP_COMPONENTS = 1;
class Traverse
{
public:
static const unsigned int traverse_variables =          0x1;
static const unsigned int traverse_constants =          0x2;
static const unsigned int traverse_functions =          0x4;
static const unsigned int traverse_blocks =             0x8;
static const unsigned int traverse_statements =        0x10;
static const unsigned int traverse_expressions =       0x20;
static const unsigned int traverse_types =             0x40;
static const unsigned int traverse_func_declarations = 0x80;
Traverse(unsigned int traverse_mask)
: traverse_mask_(traverse_mask), types_seen_(NULL), expressions_seen_(NULL)
{ }
virtual ~Traverse();
unsigned int
traverse_mask() const
{ return this->traverse_mask_; }
bool
remember_type(const Type*);
bool
remember_expression(const Expression*);
virtual int
variable(Named_object*);
virtual int
constant(Named_object*, bool);
virtual int
function(Named_object*);
virtual int
block(Block*);
virtual int
statement(Block*, size_t* index, Statement*);
virtual int
expression(Expression**);
virtual int
type(Type*);
virtual int
function_declaration(Named_object*);
private:
typedef Unordered_set(const Type*) Types_seen;
typedef Unordered_set(const Expression*) Expressions_seen;
unsigned int traverse_mask_;
Types_seen* types_seen_;
Expressions_seen* expressions_seen_;
};
class Statement_inserter
{
public:
Statement_inserter()
: block_(NULL), pindex_(NULL), gogo_(NULL), var_(NULL)
{ }
Statement_inserter(Block* block, size_t *pindex)
: block_(block), pindex_(pindex), gogo_(NULL), var_(NULL)
{ }
Statement_inserter(Gogo* gogo, Variable* var)
: block_(NULL), pindex_(NULL), gogo_(gogo), var_(var)
{ go_assert(var->is_global()); }
void
insert(Statement* s);
private:
Block* block_;
size_t* pindex_;
Gogo* gogo_;
Variable* var_;
};
class Translate_context
{
public:
Translate_context(Gogo* gogo, Named_object* function, Block* block,
Bblock* bblock)
: gogo_(gogo), backend_(gogo->backend()), function_(function),
block_(block), bblock_(bblock), is_const_(false)
{ }
Gogo*
gogo()
{ return this->gogo_; }
Backend*
backend()
{ return this->backend_; }
Named_object*
function()
{ return this->function_; }
Block*
block()
{ return this->block_; }
Bblock*
bblock()
{ return this->bblock_; }
bool
is_const()
{ return this->is_const_; }
void
set_is_const()
{ this->is_const_ = true; }
private:
Gogo* gogo_;
Backend* backend_;
Named_object* function_;
Block *block_;
Bblock* bblock_;
bool is_const_;
};
static const int RUNTIME_ERROR_SLICE_INDEX_OUT_OF_BOUNDS = 0;
static const int RUNTIME_ERROR_ARRAY_INDEX_OUT_OF_BOUNDS = 1;
static const int RUNTIME_ERROR_STRING_INDEX_OUT_OF_BOUNDS = 2;
static const int RUNTIME_ERROR_SLICE_SLICE_OUT_OF_BOUNDS = 3;
static const int RUNTIME_ERROR_ARRAY_SLICE_OUT_OF_BOUNDS = 4;
static const int RUNTIME_ERROR_STRING_SLICE_OUT_OF_BOUNDS = 5;
static const int RUNTIME_ERROR_NIL_DEREFERENCE = 6;
static const int RUNTIME_ERROR_MAKE_SLICE_OUT_OF_BOUNDS = 7;
static const int RUNTIME_ERROR_MAKE_MAP_OUT_OF_BOUNDS = 8;
static const int RUNTIME_ERROR_MAKE_CHAN_OUT_OF_BOUNDS = 9;
static const int RUNTIME_ERROR_DIVISION_BY_ZERO = 10;
static const int RUNTIME_ERROR_GO_NIL = 11;
extern Gogo* go_get_gogo();
extern bool saw_errors();
#endif 
