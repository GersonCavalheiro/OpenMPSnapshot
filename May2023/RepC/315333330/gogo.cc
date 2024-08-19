#include "go-system.h"
#include <fstream>
#include "filenames.h"
#include "go-c.h"
#include "go-diagnostics.h"
#include "go-encode-id.h"
#include "go-dump.h"
#include "go-optimize.h"
#include "lex.h"
#include "types.h"
#include "statements.h"
#include "expressions.h"
#include "runtime.h"
#include "import.h"
#include "export.h"
#include "backend.h"
#include "gogo.h"
Gogo::Gogo(Backend* backend, Linemap* linemap, int, int pointer_size)
: backend_(backend),
linemap_(linemap),
package_(NULL),
functions_(),
globals_(new Bindings(NULL)),
file_block_names_(),
imports_(),
imported_unsafe_(false),
current_file_imported_unsafe_(false),
packages_(),
init_functions_(),
var_deps_(),
need_init_fn_(false),
init_fn_name_(),
imported_init_fns_(),
pkgpath_(),
pkgpath_symbol_(),
prefix_(),
pkgpath_set_(false),
pkgpath_from_option_(false),
prefix_from_option_(false),
relative_import_path_(),
c_header_(),
check_divide_by_zero_(true),
check_divide_overflow_(true),
compiling_runtime_(false),
debug_escape_level_(0),
nil_check_size_threshold_(4096),
verify_types_(),
interface_types_(),
specific_type_functions_(),
specific_type_functions_are_written_(false),
named_types_are_converted_(false),
analysis_sets_(),
gc_roots_()
{
const Location loc = Linemap::predeclared_location();
Named_type* uint8_type = Type::make_integer_type("uint8", true, 8,
RUNTIME_TYPE_KIND_UINT8);
this->add_named_type(uint8_type);
this->add_named_type(Type::make_integer_type("uint16", true,  16,
RUNTIME_TYPE_KIND_UINT16));
this->add_named_type(Type::make_integer_type("uint32", true,  32,
RUNTIME_TYPE_KIND_UINT32));
this->add_named_type(Type::make_integer_type("uint64", true,  64,
RUNTIME_TYPE_KIND_UINT64));
this->add_named_type(Type::make_integer_type("int8",  false,   8,
RUNTIME_TYPE_KIND_INT8));
this->add_named_type(Type::make_integer_type("int16", false,  16,
RUNTIME_TYPE_KIND_INT16));
Named_type* int32_type = Type::make_integer_type("int32", false,  32,
RUNTIME_TYPE_KIND_INT32);
this->add_named_type(int32_type);
this->add_named_type(Type::make_integer_type("int64", false,  64,
RUNTIME_TYPE_KIND_INT64));
this->add_named_type(Type::make_float_type("float32", 32,
RUNTIME_TYPE_KIND_FLOAT32));
this->add_named_type(Type::make_float_type("float64", 64,
RUNTIME_TYPE_KIND_FLOAT64));
this->add_named_type(Type::make_complex_type("complex64", 64,
RUNTIME_TYPE_KIND_COMPLEX64));
this->add_named_type(Type::make_complex_type("complex128", 128,
RUNTIME_TYPE_KIND_COMPLEX128));
int int_type_size = pointer_size;
if (int_type_size < 32)
int_type_size = 32;
this->add_named_type(Type::make_integer_type("uint", true,
int_type_size,
RUNTIME_TYPE_KIND_UINT));
Named_type* int_type = Type::make_integer_type("int", false, int_type_size,
RUNTIME_TYPE_KIND_INT);
this->add_named_type(int_type);
this->add_named_type(Type::make_integer_type("uintptr", true,
pointer_size,
RUNTIME_TYPE_KIND_UINTPTR));
uint8_type->integer_type()->set_is_byte();
Named_object* byte_type = Named_object::make_type("byte", NULL, uint8_type,
loc);
byte_type->type_value()->set_is_alias();
this->add_named_type(byte_type->type_value());
int32_type->integer_type()->set_is_rune();
Named_object* rune_type = Named_object::make_type("rune", NULL, int32_type,
loc);
rune_type->type_value()->set_is_alias();
this->add_named_type(rune_type->type_value());
this->add_named_type(Type::make_named_bool_type());
this->add_named_type(Type::make_named_string_type());
{
Typed_identifier_list *methods = new Typed_identifier_list;
Typed_identifier_list *results = new Typed_identifier_list;
results->push_back(Typed_identifier("", Type::lookup_string_type(), loc));
Type *method_type = Type::make_function_type(NULL, NULL, results, loc);
methods->push_back(Typed_identifier("Error", method_type, loc));
Interface_type *error_iface = Type::make_interface_type(methods, loc);
error_iface->finalize_methods();
Named_type *error_type = Named_object::make_type("error", NULL, error_iface, loc)->type_value();
this->add_named_type(error_type);
}
this->globals_->add_constant(Typed_identifier("true",
Type::make_boolean_type(),
loc),
NULL,
Expression::make_boolean(true, loc),
0);
this->globals_->add_constant(Typed_identifier("false",
Type::make_boolean_type(),
loc),
NULL,
Expression::make_boolean(false, loc),
0);
this->globals_->add_constant(Typed_identifier("nil", Type::make_nil_type(),
loc),
NULL,
Expression::make_nil(loc),
0);
Type* abstract_int_type = Type::make_abstract_integer_type();
this->globals_->add_constant(Typed_identifier("iota", abstract_int_type,
loc),
NULL,
Expression::make_iota(),
0);
Function_type* new_type = Type::make_function_type(NULL, NULL, NULL, loc);
new_type->set_is_varargs();
new_type->set_is_builtin();
this->globals_->add_function_declaration("new", NULL, new_type, loc);
Function_type* make_type = Type::make_function_type(NULL, NULL, NULL, loc);
make_type->set_is_varargs();
make_type->set_is_builtin();
this->globals_->add_function_declaration("make", NULL, make_type, loc);
Typed_identifier_list* len_result = new Typed_identifier_list();
len_result->push_back(Typed_identifier("", int_type, loc));
Function_type* len_type = Type::make_function_type(NULL, NULL, len_result,
loc);
len_type->set_is_builtin();
this->globals_->add_function_declaration("len", NULL, len_type, loc);
Typed_identifier_list* cap_result = new Typed_identifier_list();
cap_result->push_back(Typed_identifier("", int_type, loc));
Function_type* cap_type = Type::make_function_type(NULL, NULL, len_result,
loc);
cap_type->set_is_builtin();
this->globals_->add_function_declaration("cap", NULL, cap_type, loc);
Function_type* print_type = Type::make_function_type(NULL, NULL, NULL, loc);
print_type->set_is_varargs();
print_type->set_is_builtin();
this->globals_->add_function_declaration("print", NULL, print_type, loc);
print_type = Type::make_function_type(NULL, NULL, NULL, loc);
print_type->set_is_varargs();
print_type->set_is_builtin();
this->globals_->add_function_declaration("println", NULL, print_type, loc);
Type *empty = Type::make_empty_interface_type(loc);
Typed_identifier_list* panic_parms = new Typed_identifier_list();
panic_parms->push_back(Typed_identifier("e", empty, loc));
Function_type *panic_type = Type::make_function_type(NULL, panic_parms,
NULL, loc);
panic_type->set_is_builtin();
this->globals_->add_function_declaration("panic", NULL, panic_type, loc);
Typed_identifier_list* recover_result = new Typed_identifier_list();
recover_result->push_back(Typed_identifier("", empty, loc));
Function_type* recover_type = Type::make_function_type(NULL, NULL,
recover_result,
loc);
recover_type->set_is_builtin();
this->globals_->add_function_declaration("recover", NULL, recover_type, loc);
Function_type* close_type = Type::make_function_type(NULL, NULL, NULL, loc);
close_type->set_is_varargs();
close_type->set_is_builtin();
this->globals_->add_function_declaration("close", NULL, close_type, loc);
Typed_identifier_list* copy_result = new Typed_identifier_list();
copy_result->push_back(Typed_identifier("", int_type, loc));
Function_type* copy_type = Type::make_function_type(NULL, NULL,
copy_result, loc);
copy_type->set_is_varargs();
copy_type->set_is_builtin();
this->globals_->add_function_declaration("copy", NULL, copy_type, loc);
Function_type* append_type = Type::make_function_type(NULL, NULL, NULL, loc);
append_type->set_is_varargs();
append_type->set_is_builtin();
this->globals_->add_function_declaration("append", NULL, append_type, loc);
Function_type* complex_type = Type::make_function_type(NULL, NULL, NULL, loc);
complex_type->set_is_varargs();
complex_type->set_is_builtin();
this->globals_->add_function_declaration("complex", NULL, complex_type, loc);
Function_type* real_type = Type::make_function_type(NULL, NULL, NULL, loc);
real_type->set_is_varargs();
real_type->set_is_builtin();
this->globals_->add_function_declaration("real", NULL, real_type, loc);
Function_type* imag_type = Type::make_function_type(NULL, NULL, NULL, loc);
imag_type->set_is_varargs();
imag_type->set_is_builtin();
this->globals_->add_function_declaration("imag", NULL, imag_type, loc);
Function_type* delete_type = Type::make_function_type(NULL, NULL, NULL, loc);
delete_type->set_is_varargs();
delete_type->set_is_builtin();
this->globals_->add_function_declaration("delete", NULL, delete_type, loc);
}
std::string
Gogo::pkgpath_for_symbol(const std::string& pkgpath)
{
std::string s = pkgpath;
for (size_t i = 0; i < s.length(); ++i)
{
char c = s[i];
if ((c >= 'a' && c <= 'z')
|| (c >= 'A' && c <= 'Z')
|| (c >= '0' && c <= '9'))
;
else
s[i] = '_';
}
return s;
}
const std::string&
Gogo::pkgpath() const
{
go_assert(this->pkgpath_set_);
return this->pkgpath_;
}
void
Gogo::set_pkgpath(const std::string& arg)
{
go_assert(!this->pkgpath_set_);
this->pkgpath_ = arg;
this->pkgpath_set_ = true;
this->pkgpath_from_option_ = true;
}
const std::string&
Gogo::pkgpath_symbol() const
{
go_assert(this->pkgpath_set_);
return this->pkgpath_symbol_;
}
void
Gogo::set_prefix(const std::string& arg)
{
go_assert(!this->prefix_from_option_);
this->prefix_ = arg;
this->prefix_from_option_ = true;
}
std::string
Gogo::message_name(const std::string& name)
{
return go_localize_identifier(Gogo::unpack_hidden_name(name).c_str());
}
const std::string&
Gogo::package_name() const
{
go_assert(this->package_ != NULL);
return this->package_->package_name();
}
void
Gogo::set_package_name(const std::string& package_name,
Location location)
{
if (this->package_ != NULL)
{
if (this->package_->package_name() != package_name)
go_error_at(location, "expected package %<%s%>",
Gogo::message_name(this->package_->package_name()).c_str());
return;
}
if (this->pkgpath_set_)
this->pkgpath_symbol_ = Gogo::pkgpath_for_symbol(this->pkgpath_);
else
{
if (!this->prefix_from_option_ && package_name == "main")
{
this->pkgpath_ = package_name;
this->pkgpath_symbol_ = Gogo::pkgpath_for_symbol(package_name);
}
else
{
if (!this->prefix_from_option_)
this->prefix_ = "go";
this->pkgpath_ = this->prefix_ + '.' + package_name;
this->pkgpath_symbol_ = (Gogo::pkgpath_for_symbol(this->prefix_) + '.'
+ Gogo::pkgpath_for_symbol(package_name));
}
this->pkgpath_set_ = true;
}
this->package_ = this->register_package(this->pkgpath_,
this->pkgpath_symbol_, location);
this->package_->set_package_name(package_name, location);
if (this->is_main_package())
{
Location uloc = Linemap::unknown_location();
this->declare_function(Gogo::pack_hidden_name("main", false),
Type::make_function_type (NULL, NULL, NULL, uloc),
uloc);
}
}
bool
Gogo::is_main_package() const
{
return (this->package_name() == "main"
&& !this->pkgpath_from_option_
&& !this->prefix_from_option_);
}
void
Gogo::import_package(const std::string& filename,
const std::string& local_name,
bool is_local_name_exported,
bool must_exist,
Location location)
{
if (filename.empty())
{
go_error_at(location, "import path is empty");
return;
}
const char *pf = filename.data();
const char *pend = pf + filename.length();
while (pf < pend)
{
unsigned int c;
int adv = Lex::fetch_char(pf, &c);
if (adv == 0)
{
go_error_at(location, "import path contains invalid UTF-8 sequence");
return;
}
if (c == '\0')
{
go_error_at(location, "import path contains NUL");
return;
}
if (c < 0x20 || c == 0x7f)
{
go_error_at(location, "import path contains control character");
return;
}
if (c == '\\')
{
go_error_at(location, "import path contains backslash; use slash");
return;
}
if (Lex::is_unicode_space(c))
{
go_error_at(location, "import path contains space character");
return;
}
if (c < 0x7f && strchr("!\"#$%&'()*,:;<=>?[]^`{|}", c) != NULL)
{
go_error_at(location,
"import path contains invalid character '%c'", c);
return;
}
pf += adv;
}
if (IS_ABSOLUTE_PATH(filename.c_str()))
{
go_error_at(location, "import path cannot be absolute path");
return;
}
if (local_name == "init")
go_error_at(location, "cannot import package as init");
if (filename == "unsafe")
{
this->import_unsafe(local_name, is_local_name_exported, location);
this->current_file_imported_unsafe_ = true;
return;
}
Imports::const_iterator p = this->imports_.find(filename);
if (p != this->imports_.end())
{
Package* package = p->second;
package->set_location(location);
std::string ln = local_name;
bool is_ln_exported = is_local_name_exported;
if (ln.empty())
{
ln = package->package_name();
go_assert(!ln.empty());
is_ln_exported = Lex::is_exported_name(ln);
}
if (ln == "_")
;
else if (ln == ".")
{
Bindings* bindings = package->bindings();
for (Bindings::const_declarations_iterator p =
bindings->begin_declarations();
p != bindings->end_declarations();
++p)
this->add_dot_import_object(p->second);
std::string dot_alias = "." + package->package_name();
package->add_alias(dot_alias, location);
}
else
{
package->add_alias(ln, location);
ln = this->pack_hidden_name(ln, is_ln_exported);
this->package_->bindings()->add_package(ln, package);
}
return;
}
Import::Stream* stream = Import::open_package(filename, location,
this->relative_import_path_);
if (stream == NULL)
{
if (must_exist)
go_error_at(location, "import file %qs not found", filename.c_str());
return;
}
Import imp(stream, location);
imp.register_builtin_types(this);
Package* package = imp.import(this, local_name, is_local_name_exported);
if (package != NULL)
{
if (package->pkgpath() == this->pkgpath())
go_error_at(location,
("imported package uses same package path as package "
"being compiled (see -fgo-pkgpath option)"));
this->imports_.insert(std::make_pair(filename, package));
}
delete stream;
}
Import_init *
Gogo::lookup_init(const std::string& init_name)
{
Import_init tmp("", init_name, -1);
Import_init_set::iterator it = this->imported_init_fns_.find(&tmp);
return (it != this->imported_init_fns_.end()) ? *it : NULL;
}
void
Gogo::add_import_init_fn(const std::string& package_name,
const std::string& init_name, int prio)
{
for (Import_init_set::iterator p =
this->imported_init_fns_.begin();
p != this->imported_init_fns_.end();
++p)
{
Import_init *ii = (*p);
if (ii->init_name() == init_name)
{
if (ii->package_name() != package_name)
{
go_error_at(Linemap::unknown_location(),
"duplicate package initialization name %qs",
Gogo::message_name(init_name).c_str());
go_inform(Linemap::unknown_location(), "used by package %qs",
Gogo::message_name(ii->package_name()).c_str());
go_inform(Linemap::unknown_location(), " and by package %qs",
Gogo::message_name(package_name).c_str());
}
ii->set_priority(prio);
return;
}
}
Import_init* nii = new Import_init(package_name, init_name, prio);
this->imported_init_fns_.insert(nii);
}
bool
Gogo::in_global_scope() const
{
return this->functions_.empty();
}
Bindings*
Gogo::current_bindings()
{
if (!this->functions_.empty())
return this->functions_.back().blocks.back()->bindings();
else if (this->package_ != NULL)
return this->package_->bindings();
else
return this->globals_;
}
const Bindings*
Gogo::current_bindings() const
{
if (!this->functions_.empty())
return this->functions_.back().blocks.back()->bindings();
else if (this->package_ != NULL)
return this->package_->bindings();
else
return this->globals_;
}
void
Gogo::update_init_priority(Import_init* ii,
std::set<const Import_init *>* visited)
{
visited->insert(ii);
int succ_prior = -1;
for (std::set<std::string>::const_iterator pci =
ii->precursors().begin();
pci != ii->precursors().end();
++pci)
{
Import_init* succ = this->lookup_init(*pci);
if (visited->find(succ) == visited->end())
update_init_priority(succ, visited);
succ_prior = std::max(succ_prior, succ->priority());
}
if (ii->priority() <= succ_prior)
ii->set_priority(succ_prior + 1);
}
void
Gogo::recompute_init_priorities()
{
std::set<Import_init *> nonroots;
for (Import_init_set::const_iterator p =
this->imported_init_fns_.begin();
p != this->imported_init_fns_.end();
++p)
{
const Import_init *ii = *p;
for (std::set<std::string>::const_iterator pci =
ii->precursors().begin();
pci != ii->precursors().end();
++pci)
{
Import_init* ii = this->lookup_init(*pci);
nonroots.insert(ii);
}
}
std::set<const Import_init*> visited;
for (Import_init_set::iterator p =
this->imported_init_fns_.begin();
p != this->imported_init_fns_.end();
++p)
{
Import_init* ii = *p;
if (nonroots.find(ii) != nonroots.end())
continue;
update_init_priority(ii, &visited);
}
}
void
Gogo::init_imports(std::vector<Bstatement*>& init_stmts, Bfunction *bfunction)
{
go_assert(this->is_main_package());
if (this->imported_init_fns_.empty())
return;
Location unknown_loc = Linemap::unknown_location();
Function_type* func_type =
Type::make_function_type(NULL, NULL, NULL, unknown_loc);
Btype* fntype = func_type->get_backend_fntype(this);
recompute_init_priorities();
std::vector<const Import_init*> v;
for (Import_init_set::const_iterator p =
this->imported_init_fns_.begin();
p != this->imported_init_fns_.end();
++p)
{
if ((*p)->priority() < 0)
go_error_at(Linemap::unknown_location(),
"internal error: failed to set init priority for %s",
(*p)->package_name().c_str());
v.push_back(*p);
}
std::sort(v.begin(), v.end(), priority_compare);
std::vector<Bexpression*> empty_args;
for (std::vector<const Import_init*>::const_iterator p = v.begin();
p != v.end();
++p)
{
const Import_init* ii = *p;
std::string user_name = ii->package_name() + ".init";
const std::string& init_name(ii->init_name());
Bfunction* pfunc = this->backend()->function(fntype, user_name, init_name,
true, true, true, false,
false, false, unknown_loc);
Bexpression* pfunc_code =
this->backend()->function_code_expression(pfunc, unknown_loc);
Bexpression* pfunc_call =
this->backend()->call_expression(bfunction, pfunc_code, empty_args,
NULL, unknown_loc);
init_stmts.push_back(this->backend()->expression_statement(bfunction,
pfunc_call));
}
}
void
Gogo::register_gc_vars(const std::vector<Named_object*>& var_gc,
std::vector<Bstatement*>& init_stmts,
Bfunction* init_bfn)
{
if (var_gc.empty() && this->gc_roots_.empty())
return;
Type* pvt = Type::make_pointer_type(Type::make_void_type());
Type* uintptr_type = Type::lookup_integer_type("uintptr");
Type* byte_type = this->lookup_global("byte")->type_value();
Type* pointer_byte_type = Type::make_pointer_type(byte_type);
Struct_type* root_type =
Type::make_builtin_struct_type(4,
"decl", pvt,
"size", uintptr_type,
"ptrdata", uintptr_type,
"gcdata", pointer_byte_type);
Location builtin_loc = Linemap::predeclared_location();
unsigned long roots_len = var_gc.size() + this->gc_roots_.size();
Expression* length = Expression::make_integer_ul(roots_len, NULL,
builtin_loc);
Array_type* root_array_type = Type::make_array_type(root_type, length);
root_array_type->set_is_array_incomparable();
Type* int_type = Type::lookup_integer_type("int");
Struct_type* root_list_type =
Type::make_builtin_struct_type(3,
"next", pvt,
"count", int_type,
"roots", root_array_type);
Expression_list* roots_init = new Expression_list();
for (std::vector<Named_object*>::const_iterator p = var_gc.begin();
p != var_gc.end();
++p)
{
Expression_list* init = new Expression_list();
Location no_loc = (*p)->location();
Expression* decl = Expression::make_var_reference(*p, no_loc);
Expression* decl_addr =
Expression::make_unary(OPERATOR_AND, decl, no_loc);
decl_addr->unary_expression()->set_does_not_escape();
decl_addr = Expression::make_cast(pvt, decl_addr, no_loc);
init->push_back(decl_addr);
Expression* size =
Expression::make_type_info(decl->type(),
Expression::TYPE_INFO_SIZE);
init->push_back(size);
Expression* ptrdata =
Expression::make_type_info(decl->type(),
Expression::TYPE_INFO_BACKEND_PTRDATA);
init->push_back(ptrdata);
Expression* gcdata = Expression::make_ptrmask_symbol(decl->type());
init->push_back(gcdata);
Expression* root_ctor =
Expression::make_struct_composite_literal(root_type, init, no_loc);
roots_init->push_back(root_ctor);
}
for (std::vector<Expression*>::const_iterator p = this->gc_roots_.begin();
p != this->gc_roots_.end();
++p)
{
Expression_list *init = new Expression_list();
Expression* expr = *p;
Location eloc = expr->location();
init->push_back(Expression::make_cast(pvt, expr, eloc));
Type* type = expr->type()->points_to();
go_assert(type != NULL);
Expression* size =
Expression::make_type_info(type,
Expression::TYPE_INFO_SIZE);
init->push_back(size);
Expression* ptrdata =
Expression::make_type_info(type,
Expression::TYPE_INFO_BACKEND_PTRDATA);
init->push_back(ptrdata);
Expression* gcdata = Expression::make_ptrmask_symbol(type);
init->push_back(gcdata);
Expression* root_ctor =
Expression::make_struct_composite_literal(root_type, init, eloc);
roots_init->push_back(root_ctor);
}
Expression_list* root_list_init = new Expression_list();
root_list_init->push_back(Expression::make_nil(builtin_loc));
root_list_init->push_back(Expression::make_integer_ul(roots_len, int_type,
builtin_loc));
Expression* roots_ctor =
Expression::make_array_composite_literal(root_array_type, roots_init,
builtin_loc);
root_list_init->push_back(roots_ctor);
Expression* root_list_ctor =
Expression::make_struct_composite_literal(root_list_type, root_list_init,
builtin_loc);
Expression* root_addr = Expression::make_unary(OPERATOR_AND, root_list_ctor,
builtin_loc);
root_addr->unary_expression()->set_is_gc_root();
Expression* register_roots = Runtime::make_call(Runtime::REGISTER_GC_ROOTS,
builtin_loc, 1, root_addr);
Translate_context context(this, NULL, NULL, NULL);
Bexpression* bcall = register_roots->get_backend(&context);
init_stmts.push_back(this->backend()->expression_statement(init_bfn, bcall));
}
Named_object*
Gogo::initialization_function_decl()
{
std::string name = this->get_init_fn_name();
Location loc = this->package_->location();
Function_type* fntype = Type::make_function_type(NULL, NULL, NULL, loc);
Function* initfn = new Function(fntype, NULL, NULL, loc);
return Named_object::make_function(name, NULL, initfn);
}
Named_object*
Gogo::create_initialization_function(Named_object* initfn,
Bstatement* code_stmt)
{
go_assert(this->is_main_package() || this->need_init_fn_);
if (initfn == NULL)
initfn = this->initialization_function_decl();
Bfunction* fndecl = initfn->func_value()->get_or_make_decl(this, initfn);
Location pkg_loc = this->package_->location();
std::vector<Bvariable*> vars;
this->backend()->block(fndecl, NULL, vars, pkg_loc, pkg_loc);
if (!this->backend()->function_set_body(fndecl, code_stmt))
{
go_assert(saw_errors());
return NULL;
}
return initfn;
}
class Find_var : public Traverse
{
public:
typedef Unordered_set(const void*) Seen_objects;
Find_var(Named_object* var, Seen_objects* seen_objects)
: Traverse(traverse_expressions),
var_(var), seen_objects_(seen_objects), found_(false)
{ }
bool
found() const
{ return this->found_; }
int
expression(Expression**);
private:
Named_object* var_;
Seen_objects* seen_objects_;
bool found_;
};
int
Find_var::expression(Expression** pexpr)
{
Expression* e = *pexpr;
Var_expression* ve = e->var_expression();
if (ve != NULL)
{
Named_object* v = ve->named_object();
if (v == this->var_)
{
this->found_ = true;
return TRAVERSE_EXIT;
}
if (v->is_variable() && v->package() == NULL)
{
Expression* init = v->var_value()->init();
if (init != NULL)
{
std::pair<Seen_objects::iterator, bool> ins =
this->seen_objects_->insert(v);
if (ins.second)
{
if (Expression::traverse(&init, this) == TRAVERSE_EXIT)
return TRAVERSE_EXIT;
}
}
}
}
Func_expression* fe = e->func_expression();
Bound_method_expression* bme = e->bound_method_expression();
if (fe != NULL || bme != NULL)
{
const Named_object* f = fe != NULL ? fe->named_object() : bme->function();
if (f->is_function() && f->package() == NULL)
{
std::pair<Seen_objects::iterator, bool> ins =
this->seen_objects_->insert(f);
if (ins.second)
{
if (f->func_value()->block()->traverse(this) == TRAVERSE_EXIT)
return TRAVERSE_EXIT;
}
}
}
Temporary_reference_expression* tre = e->temporary_reference_expression();
if (tre != NULL)
{
Temporary_statement* ts = tre->statement();
Expression* init = ts->init();
if (init != NULL)
{
std::pair<Seen_objects::iterator, bool> ins =
this->seen_objects_->insert(ts);
if (ins.second)
{
if (Expression::traverse(&init, this) == TRAVERSE_EXIT)
return TRAVERSE_EXIT;
}
}
}
return TRAVERSE_CONTINUE;
}
static bool
expression_requires(Expression* expr, Block* preinit, Named_object* dep,
Named_object* var)
{
Find_var::Seen_objects seen_objects;
Find_var find_var(var, &seen_objects);
if (expr != NULL)
Expression::traverse(&expr, &find_var);
if (preinit != NULL)
preinit->traverse(&find_var);
if (dep != NULL)
{
Expression* init = dep->var_value()->init();
if (init != NULL)
Expression::traverse(&init, &find_var);
if (dep->var_value()->has_pre_init())
dep->var_value()->preinit()->traverse(&find_var);
}
return find_var.found();
}
class Var_init
{
public:
Var_init()
: var_(NULL), init_(NULL), dep_count_(0)
{ }
Var_init(Named_object* var, Bstatement* init)
: var_(var), init_(init), dep_count_(0)
{ }
Named_object*
var() const
{ return this->var_; }
Bstatement*
init() const
{ return this->init_; }
size_t
dep_count() const
{ return this->dep_count_; }
void
add_dependency()
{ ++this->dep_count_; }
void
remove_dependency()
{ --this->dep_count_; }
private:
Named_object* var_;
Bstatement* init_;
size_t dep_count_;
};
inline bool
operator<(const Var_init& v1, const Var_init& v2)
{ return v1.var()->name() < v2.var()->name(); }
typedef std::list<Var_init> Var_inits;
static void
sort_var_inits(Gogo* gogo, Var_inits* var_inits)
{
if (var_inits->empty())
return;
typedef std::pair<Named_object*, Named_object*> No_no;
typedef std::map<No_no, bool> Cache;
Cache cache;
typedef std::map<Var_init, std::set<Var_init*> > Init_deps;
Init_deps init_deps;
bool init_loop = false;
for (Var_inits::iterator p1 = var_inits->begin();
p1 != var_inits->end();
++p1)
{
Named_object* var = p1->var();
Expression* init = var->var_value()->init();
Block* preinit = var->var_value()->preinit();
Named_object* dep = gogo->var_depends_on(var->var_value());
for (Var_inits::iterator p2 = var_inits->begin();
p2 != var_inits->end();
++p2)
{
if (var == p2->var())
continue;
Named_object* p2var = p2->var();
No_no key(var, p2var);
std::pair<Cache::iterator, bool> ins =
cache.insert(std::make_pair(key, false));
if (ins.second)
ins.first->second = expression_requires(init, preinit, dep, p2var);
if (ins.first->second)
{
init_deps[*p2].insert(&(*p1));
p1->add_dependency();
key = std::make_pair(p2var, var);
ins = cache.insert(std::make_pair(key, false));
if (ins.second)
ins.first->second =
expression_requires(p2var->var_value()->init(),
p2var->var_value()->preinit(),
gogo->var_depends_on(p2var->var_value()),
var);
if (ins.first->second)
{
go_error_at(var->location(),
("initialization expressions for %qs and "
"%qs depend upon each other"),
var->message_name().c_str(),
p2var->message_name().c_str());
go_inform(p2->var()->location(), "%qs defined here",
p2var->message_name().c_str());
init_loop = true;
break;
}
}
}
}
if (!init_deps.empty() && !init_loop)
{
Var_inits ready;
while (!var_inits->empty())
{
Var_inits::iterator v1;;
for (v1 = var_inits->begin(); v1 != var_inits->end(); ++v1)
{
if (v1->dep_count() == 0)
break;
}
go_assert(v1 != var_inits->end());
ready.splice(ready.end(), *var_inits, v1);
Init_deps::iterator p1 = init_deps.find(*v1);
if (p1 != init_deps.end())
{
std::set<Var_init*> resolved = p1->second;
for (std::set<Var_init*>::iterator pv = resolved.begin();
pv != resolved.end();
++pv)
(*pv)->remove_dependency();
init_deps.erase(p1);
}
}
var_inits->swap(ready);
go_assert(init_deps.empty());
}
for (Var_inits::const_iterator p = var_inits->begin();
p != var_inits->end();
++p)
gogo->check_self_dep(p->var());
}
void
Gogo::check_self_dep(Named_object* var)
{
Expression* init = var->var_value()->init();
Block* preinit = var->var_value()->preinit();
Named_object* dep = this->var_depends_on(var->var_value());
if (init != NULL
&& dep == NULL
&& expression_requires(init, preinit, NULL, var))
go_error_at(var->location(),
"initialization expression for %qs depends upon itself",
var->message_name().c_str());
}
void
Gogo::write_globals()
{
this->build_interface_method_tables();
Bindings* bindings = this->current_bindings();
for (Bindings::const_declarations_iterator p = bindings->begin_declarations();
p != bindings->end_declarations();
++p)
{
Named_object* no = p->second;
if (no->is_function_declaration())
no->func_declaration_value()->build_backend_descriptor(this);
}
std::vector<Btype*> type_decls;
std::vector<Bvariable*> var_decls;
std::vector<Bexpression*> const_decls;
std::vector<Bfunction*> func_decls;
Named_object* init_fndecl = NULL;
Bfunction* init_bfn = NULL;
std::vector<Bstatement*> init_stmts;
std::vector<Bstatement*> var_init_stmts;
if (this->is_main_package())
{
init_fndecl = this->initialization_function_decl();
init_bfn = init_fndecl->func_value()->get_or_make_decl(this, init_fndecl);
this->init_imports(init_stmts, init_bfn);
}
Var_inits var_inits;
size_t count_definitions = bindings->size_definitions();
std::vector<Named_object*> var_gc;
var_gc.reserve(count_definitions);
for (Bindings::const_definitions_iterator p = bindings->begin_definitions();
p != bindings->end_definitions();
++p)
{
Named_object* no = *p;
go_assert(!no->is_type_declaration() && !no->is_function_declaration());
if (no->is_package())
continue;
if (no->package() != NULL)
continue;
if ((no->is_function() && no->func_value()->is_sink())
|| (no->is_const() && no->const_value()->is_sink()))
continue;
if (no->is_const())
{
Type* type = no->const_value()->type();
if (type == NULL)
type = no->const_value()->expr()->type();
if (type->is_abstract() || !type->is_numeric_type())
continue;
}
if (!no->is_variable())
no->get_backend(this, const_decls, type_decls, func_decls);
else
{
Variable* var = no->var_value();
Bvariable* bvar = no->get_backend_variable(this, NULL);
var_decls.push_back(bvar);
bool is_sink = no->name()[0] == '_' && no->name()[1] == '.';
Bstatement* var_init_stmt = NULL;
if (!var->has_pre_init())
{
bool is_static_initializer = false;
if (var->init() == NULL)
is_static_initializer = true;
else
{
Type* var_type = var->type();
Expression* init = var->init();
Expression* init_cast =
Expression::make_cast(var_type, init, var->location());
is_static_initializer = init_cast->is_static_initializer();
}
Named_object* var_init_fn;
if (is_static_initializer)
var_init_fn = NULL;
else
{
if (init_fndecl == NULL)
{
init_fndecl = this->initialization_function_decl();
Function* func = init_fndecl->func_value();
init_bfn = func->get_or_make_decl(this, init_fndecl);
}
var_init_fn = init_fndecl;
}
Bexpression* var_binit = var->get_init(this, var_init_fn);
if (var_binit == NULL)
;
else if (is_static_initializer)
{
if (expression_requires(var->init(), NULL,
this->var_depends_on(var), no))
go_error_at(no->location(),
"initialization expression for %qs depends "
"upon itself",
no->message_name().c_str());
this->backend()->global_variable_set_init(bvar, var_binit);
}
else if (is_sink)
var_init_stmt =
this->backend()->expression_statement(init_bfn, var_binit);
else
{
Location loc = var->location();
Bexpression* var_expr =
this->backend()->var_expression(bvar, loc);
var_init_stmt =
this->backend()->assignment_statement(init_bfn, var_expr,
var_binit, loc);
}
}
else
{
if (init_fndecl == NULL)
init_fndecl = this->initialization_function_decl();
Bvariable* var_decl = is_sink ? NULL : bvar;
var_init_stmt = var->get_init_block(this, init_fndecl, var_decl);
}
if (var_init_stmt != NULL)
{
if (var->init() == NULL && !var->has_pre_init())
var_init_stmts.push_back(var_init_stmt);
else
var_inits.push_back(Var_init(no, var_init_stmt));
}
else if (this->var_depends_on(var) != NULL)
{
Btype* btype = no->var_value()->type()->get_backend(this);
Bexpression* zero = this->backend()->zero_expression(btype);
Bstatement* zero_stmt =
this->backend()->expression_statement(init_bfn, zero);
var_inits.push_back(Var_init(no, zero_stmt));
}
if (!is_sink && var->type()->has_pointer())
{
if (this->compiling_runtime()
&& this->package_name() == "runtime"
&& Gogo::unpack_hidden_name(no->name()) == "gcRoots")
;
else
var_gc.push_back(no);
}
}
}
this->register_gc_vars(var_gc, init_stmts, init_bfn);
init_stmts.push_back(this->backend()->statement_list(var_init_stmts));
if (!var_inits.empty())
{
sort_var_inits(this, &var_inits);
for (Var_inits::const_iterator p = var_inits.begin();
p != var_inits.end();
++p)
init_stmts.push_back(p->init());
}
std::vector<Bexpression*> empty_args;
for (std::vector<Named_object*>::const_iterator p =
this->init_functions_.begin();
p != this->init_functions_.end();
++p)
{
Location func_loc = (*p)->location();
Function* func = (*p)->func_value();
Bfunction* initfn = func->get_or_make_decl(this, *p);
Bexpression* func_code =
this->backend()->function_code_expression(initfn, func_loc);
Bexpression* call = this->backend()->call_expression(init_bfn, func_code,
empty_args,
NULL, func_loc);
Bstatement* ist = this->backend()->expression_statement(init_bfn, call);
init_stmts.push_back(ist);
}
Bstatement* init_fncode = this->backend()->statement_list(init_stmts);
if (this->need_init_fn_ || this->is_main_package())
{
init_fndecl =
this->create_initialization_function(init_fndecl, init_fncode);
if (init_fndecl != NULL)
func_decls.push_back(init_fndecl->func_value()->get_decl());
}
go_assert(count_definitions == this->current_bindings()->size_definitions());
if (!saw_errors())
this->backend()->write_global_definitions(type_decls, const_decls,
func_decls, var_decls);
}
Block*
Gogo::current_block()
{
if (this->functions_.empty())
return NULL;
else
return this->functions_.back().blocks.back();
}
Named_object*
Gogo::lookup(const std::string& name, Named_object** pfunction) const
{
if (pfunction != NULL)
*pfunction = NULL;
if (Gogo::is_sink_name(name))
return Named_object::make_sink();
for (Open_functions::const_reverse_iterator p = this->functions_.rbegin();
p != this->functions_.rend();
++p)
{
Named_object* ret = p->blocks.back()->bindings()->lookup(name);
if (ret != NULL)
{
if (pfunction != NULL)
*pfunction = p->function;
return ret;
}
}
if (this->package_ != NULL)
{
Named_object* ret = this->package_->bindings()->lookup(name);
if (ret != NULL)
{
if (ret->package() != NULL)
{
std::string dot_alias = "." + ret->package()->package_name();
ret->package()->note_usage(dot_alias);
}
return ret;
}
}
return NULL;
}
Named_object*
Gogo::lookup_in_block(const std::string& name) const
{
go_assert(!this->functions_.empty());
go_assert(!this->functions_.back().blocks.empty());
return this->functions_.back().blocks.back()->bindings()->lookup_local(name);
}
Named_object*
Gogo::lookup_global(const char* name) const
{
return this->globals_->lookup(name);
}
Package*
Gogo::add_imported_package(const std::string& real_name,
const std::string& alias_arg,
bool is_alias_exported,
const std::string& pkgpath,
const std::string& pkgpath_symbol,
Location location,
bool* padd_to_globals)
{
Package* ret = this->register_package(pkgpath, pkgpath_symbol, location);
ret->set_package_name(real_name, location);
*padd_to_globals = false;
if (alias_arg == "_")
;
else if (alias_arg == ".")
{
*padd_to_globals = true;
std::string dot_alias = "." + real_name;
ret->add_alias(dot_alias, location);
}
else
{
std::string alias = alias_arg;
if (alias.empty())
{
alias = real_name;
is_alias_exported = Lex::is_exported_name(alias);
}
ret->add_alias(alias, location);
alias = this->pack_hidden_name(alias, is_alias_exported);
Named_object* no = this->package_->bindings()->add_package(alias, ret);
if (!no->is_package())
return NULL;
}
return ret;
}
Package*
Gogo::register_package(const std::string& pkgpath,
const std::string& pkgpath_symbol, Location location)
{
Package* package = NULL;
std::pair<Packages::iterator, bool> ins =
this->packages_.insert(std::make_pair(pkgpath, package));
if (!ins.second)
{
package = ins.first->second;
go_assert(package != NULL && package->pkgpath() == pkgpath);
if (!pkgpath_symbol.empty())
package->set_pkgpath_symbol(pkgpath_symbol);
if (Linemap::is_unknown_location(package->location()))
package->set_location(location);
}
else
{
package = new Package(pkgpath, pkgpath_symbol, location);
go_assert(ins.first->second == NULL);
ins.first->second = package;
}
return package;
}
std::string
Gogo::pkgpath_symbol_for_package(const std::string& pkgpath)
{
Packages::iterator p = this->packages_.find(pkgpath);
go_assert(p != this->packages_.end());
return p->second->pkgpath_symbol();
}
Named_object*
Gogo::start_function(const std::string& name, Function_type* type,
bool add_method_to_type, Location location)
{
bool at_top_level = this->functions_.empty();
Block* block = new Block(NULL, location);
Named_object* enclosing = (at_top_level
? NULL
: this->functions_.back().function);
Function* function = new Function(type, enclosing, block, location);
if (type->is_method())
{
const Typed_identifier* receiver = type->receiver();
Variable* this_param = new Variable(receiver->type(), NULL, false,
true, true, location);
std::string rname = receiver->name();
if (rname.empty() || Gogo::is_sink_name(rname))
{
static unsigned int count;
char buf[50];
snprintf(buf, sizeof buf, "r.%u", count);
++count;
rname = buf;
}
block->bindings()->add_variable(rname, NULL, this_param);
}
const Typed_identifier_list* parameters = type->parameters();
bool is_varargs = type->is_varargs();
if (parameters != NULL)
{
for (Typed_identifier_list::const_iterator p = parameters->begin();
p != parameters->end();
++p)
{
Variable* param = new Variable(p->type(), NULL, false, true, false,
p->location());
if (is_varargs && p + 1 == parameters->end())
param->set_is_varargs_parameter();
std::string pname = p->name();
if (pname.empty() || Gogo::is_sink_name(pname))
{
static unsigned int count;
char buf[50];
snprintf(buf, sizeof buf, "p.%u", count);
++count;
pname = buf;
}
block->bindings()->add_variable(pname, NULL, param);
}
}
function->create_result_variables(this);
const std::string* pname;
std::string nested_name;
bool is_init = false;
if (Gogo::unpack_hidden_name(name) == "init" && !type->is_method())
{
if ((type->parameters() != NULL && !type->parameters()->empty())
|| (type->results() != NULL && !type->results()->empty()))
go_error_at(location,
"func init must have no arguments and no return values");
nested_name = this->init_function_name();
pname = &nested_name;
is_init = true;
}
else if (!name.empty())
pname = &name;
else
{
nested_name = this->nested_function_name(enclosing);
pname = &nested_name;
}
Named_object* ret;
if (Gogo::is_sink_name(*pname))
{
std::string sname(this->sink_function_name());
ret = Named_object::make_function(sname, NULL, function);
ret->func_value()->set_is_sink();
if (!type->is_method())
ret = this->package_->bindings()->add_named_object(ret);
else if (add_method_to_type)
{
Type* rtype = type->receiver()->type();
if (rtype->classification() == Type::TYPE_POINTER)
rtype = rtype->points_to();
while (rtype->named_type() != NULL
&& rtype->named_type()->is_alias())
rtype = rtype->named_type()->real_type()->forwarded();
if (rtype->is_error_type())
;
else if (rtype->named_type() != NULL)
{
if (rtype->named_type()->named_object()->package() != NULL)
go_error_at(type->receiver()->location(),
"may not define methods on non-local type");
}
else if (rtype->forward_declaration_type() != NULL)
{
rtype->forward_declaration_type()->add_existing_method(ret);
}
else
go_error_at(type->receiver()->location(),
("invalid receiver type "
"(receiver must be a named type)"));
}
}
else if (!type->is_method())
{
ret = this->package_->bindings()->add_function(*pname, NULL, function);
if (!ret->is_function() || ret->func_value() != function)
{
std::string rname(this->redefined_function_name());
ret = this->package_->bindings()->add_function(rname, NULL, function);
}
}
else
{
if (!add_method_to_type)
ret = Named_object::make_function(name, NULL, function);
else
{
go_assert(at_top_level);
Type* rtype = type->receiver()->type();
if (rtype->classification() == Type::TYPE_POINTER)
rtype = rtype->points_to();
while (rtype->named_type() != NULL
&& rtype->named_type()->is_alias())
rtype = rtype->named_type()->real_type()->forwarded();
if (rtype->is_error_type())
ret = Named_object::make_function(name, NULL, function);
else if (rtype->named_type() != NULL)
{
if (rtype->named_type()->named_object()->package() != NULL)
{
go_error_at(type->receiver()->location(),
"may not define methods on non-local type");
ret = Named_object::make_function(name, NULL, function);
}
else
{
ret = rtype->named_type()->add_method(name, function);
if (!ret->is_function())
{
ret = Named_object::make_function(name, NULL, function);
}
}
}
else if (rtype->forward_declaration_type() != NULL)
{
Named_object* type_no =
rtype->forward_declaration_type()->named_object();
if (type_no->is_unknown())
{
Named_object* declared =
this->declare_package_type(type_no->name(),
type_no->location());
go_assert(declared
== type_no->unknown_value()->real_named_object());
}
ret = rtype->forward_declaration_type()->add_method(name,
function);
}
else
{
go_error_at(type->receiver()->location(),
("invalid receiver type (receiver must "
"be a named type)"));
ret = Named_object::make_function(name, NULL, function);
}
}
this->package_->bindings()->add_method(ret);
}
this->functions_.resize(this->functions_.size() + 1);
Open_function& of(this->functions_.back());
of.function = ret;
of.blocks.push_back(block);
if (is_init)
{
this->init_functions_.push_back(ret);
this->need_init_fn_ = true;
}
return ret;
}
void
Gogo::finish_function(Location location)
{
this->finish_block(location);
go_assert(this->functions_.back().blocks.empty());
this->functions_.pop_back();
}
Named_object*
Gogo::current_function() const
{
go_assert(!this->functions_.empty());
return this->functions_.back().function;
}
void
Gogo::start_block(Location location)
{
go_assert(!this->functions_.empty());
Block* block = new Block(this->current_block(), location);
this->functions_.back().blocks.push_back(block);
}
Block*
Gogo::finish_block(Location location)
{
go_assert(!this->functions_.empty());
go_assert(!this->functions_.back().blocks.empty());
Block* block = this->functions_.back().blocks.back();
this->functions_.back().blocks.pop_back();
block->set_end_location(location);
return block;
}
Named_object*
Gogo::add_erroneous_name(const std::string& name)
{
return this->package_->bindings()->add_erroneous_name(name);
}
Named_object*
Gogo::add_unknown_name(const std::string& name, Location location)
{
return this->package_->bindings()->add_unknown_name(name, location);
}
Named_object*
Gogo::declare_function(const std::string& name, Function_type* type,
Location location)
{
if (!type->is_method())
return this->current_bindings()->add_function_declaration(name, NULL, type,
location);
else
{
Type* rtype = type->receiver()->type();
if (rtype->classification() == Type::TYPE_POINTER)
rtype = rtype->points_to();
if (rtype->is_error_type())
return NULL;
else if (rtype->named_type() != NULL)
return rtype->named_type()->add_method_declaration(name, NULL, type,
location);
else if (rtype->forward_declaration_type() != NULL)
{
Forward_declaration_type* ftype = rtype->forward_declaration_type();
return ftype->add_method_declaration(name, NULL, type, location);
}
else
{
go_error_at(type->receiver()->location(),
"invalid receiver type (receiver must be a named type)");
return Named_object::make_erroneous_name(name);
}
}
}
Label*
Gogo::add_label_definition(const std::string& label_name,
Location location)
{
go_assert(!this->functions_.empty());
Function* func = this->functions_.back().function->func_value();
Label* label = func->add_label_definition(this, label_name, location);
this->add_statement(Statement::make_label_statement(label, location));
return label;
}
Label*
Gogo::add_label_reference(const std::string& label_name,
Location location, bool issue_goto_errors)
{
go_assert(!this->functions_.empty());
Function* func = this->functions_.back().function->func_value();
return func->add_label_reference(this, label_name, location,
issue_goto_errors);
}
Bindings_snapshot*
Gogo::bindings_snapshot(Location location)
{
return new Bindings_snapshot(this->current_block(), location);
}
void
Gogo::add_statement(Statement* statement)
{
go_assert(!this->functions_.empty()
&& !this->functions_.back().blocks.empty());
this->functions_.back().blocks.back()->add_statement(statement);
}
void
Gogo::add_block(Block* block, Location location)
{
go_assert(!this->functions_.empty()
&& !this->functions_.back().blocks.empty());
Statement* statement = Statement::make_block_statement(block, location);
this->functions_.back().blocks.back()->add_statement(statement);
}
Named_object*
Gogo::add_constant(const Typed_identifier& tid, Expression* expr,
int iota_value)
{
return this->current_bindings()->add_constant(tid, NULL, expr, iota_value);
}
void
Gogo::add_type(const std::string& name, Type* type, Location location)
{
Named_object* no = this->current_bindings()->add_type(name, NULL, type,
location);
if (!this->in_global_scope() && no->is_type())
{
Named_object* f = this->functions_.back().function;
unsigned int index;
if (f->is_function())
index = f->func_value()->new_local_type_index();
else
index = 0;
no->type_value()->set_in_function(f, index);
}
}
void
Gogo::add_named_type(Named_type* type)
{
go_assert(this->in_global_scope());
this->current_bindings()->add_named_type(type);
}
Named_object*
Gogo::declare_type(const std::string& name, Location location)
{
Bindings* bindings = this->current_bindings();
Named_object* no = bindings->add_type_declaration(name, NULL, location);
if (!this->in_global_scope() && no->is_type_declaration())
{
Named_object* f = this->functions_.back().function;
unsigned int index;
if (f->is_function())
index = f->func_value()->new_local_type_index();
else
index = 0;
no->type_declaration_value()->set_in_function(f, index);
}
return no;
}
Named_object*
Gogo::declare_package_type(const std::string& name, Location location)
{
return this->package_->bindings()->add_type_declaration(name, NULL, location);
}
Named_object*
Gogo::declare_package_function(const std::string& name, Function_type* type,
Location location)
{
return this->package_->bindings()->add_function_declaration(name, NULL, type,
location);
}
void
Gogo::define_type(Named_object* no, Named_type* type)
{
this->current_bindings()->define_type(no, type);
}
Named_object*
Gogo::add_variable(const std::string& name, Variable* variable)
{
Named_object* no = this->current_bindings()->add_variable(name, NULL,
variable);
if (no != NULL
&& no->is_variable()
&& !no->var_value()->is_parameter()
&& !this->functions_.empty())
this->add_statement(Statement::make_variable_declaration(no));
return no;
}
Named_object*
Gogo::add_sink()
{
return Named_object::make_sink();
}
void
Gogo::add_dot_import_object(Named_object* no)
{
Named_object* e = this->package_->bindings()->lookup(no->name());
if (e != NULL && e->package() == NULL)
{
if (e->is_unknown())
e = e->resolve();
if (e->package() == NULL
&& (e->is_type_declaration()
|| e->is_function_declaration()
|| e->is_unknown()))
{
this->add_file_block_name(no->name(), no->location());
return;
}
}
this->current_bindings()->add_named_object(no);
}
void
Gogo::add_linkname(const std::string& go_name, bool is_exported,
const std::string& ext_name, Location loc)
{
Named_object* no =
this->package_->bindings()->lookup(this->pack_hidden_name(go_name,
is_exported));
if (no == NULL)
go_error_at(loc, "%s is not defined", go_name.c_str());
else if (no->is_function())
no->func_value()->set_asm_name(ext_name);
else if (no->is_function_declaration())
no->func_declaration_value()->set_asm_name(ext_name);
else
go_error_at(loc,
("%s is not a function; "
"
go_name.c_str());
}
void
Gogo::mark_locals_used()
{
for (Open_functions::iterator pf = this->functions_.begin();
pf != this->functions_.end();
++pf)
{
for (std::vector<Block*>::iterator pb = pf->blocks.begin();
pb != pf->blocks.end();
++pb)
(*pb)->bindings()->mark_locals_used();
}
}
void
Gogo::record_interface_type(Interface_type* itype)
{
this->interface_types_.push_back(itype);
}
void
Gogo::define_global_names()
{
if (this->is_main_package())
{
this->import_package("runtime", "_", false, false,
Linemap::predeclared_location());
}
for (Bindings::const_declarations_iterator p =
this->globals_->begin_declarations();
p != this->globals_->end_declarations();
++p)
{
Named_object* global_no = p->second;
std::string name(Gogo::pack_hidden_name(global_no->name(), false));
Named_object* no = this->package_->bindings()->lookup(name);
if (no == NULL)
continue;
no = no->resolve();
if (no->is_type_declaration())
{
if (global_no->is_type())
{
if (no->type_declaration_value()->has_methods())
{
for (std::vector<Named_object*>::const_iterator p =
no->type_declaration_value()->methods()->begin();
p != no->type_declaration_value()->methods()->end();
p++)
go_error_at((*p)->location(),
"may not define methods on non-local type");
}
no->set_type_value(global_no->type_value());
}
else
{
go_error_at(no->location(), "expected type");
Type* errtype = Type::make_error_type();
Named_object* err =
Named_object::make_type("erroneous_type", NULL, errtype,
Linemap::predeclared_location());
no->set_type_value(err->type_value());
}
}
else if (no->is_unknown())
no->unknown_value()->set_real_named_object(global_no);
}
for (Bindings::const_declarations_iterator p =
this->package_->bindings()->begin_declarations();
p != this->package_->bindings()->end_declarations();
++p)
{
if (p->second->is_unknown()
&& p->second->unknown_value()->real_named_object() == NULL)
{
continue;
}
File_block_names::const_iterator pf =
this->file_block_names_.find(p->second->name());
if (pf != this->file_block_names_.end())
{
std::string n = p->second->message_name();
go_error_at(p->second->location(),
"%qs defined as both imported name and global name",
n.c_str());
go_inform(pf->second, "%qs imported here", n.c_str());
}
if (!p->second->is_function()
&& Gogo::unpack_hidden_name(p->second->name()) == "init")
{
go_error_at(p->second->location(),
"cannot declare init - must be func");
}
}
}
void
Gogo::clear_file_scope()
{
this->package_->bindings()->clear_file_scope(this);
bool quiet = saw_errors();
for (Packages::iterator p = this->packages_.begin();
p != this->packages_.end();
++p)
{
Package* package = p->second;
if (package != this->package_ && !quiet)
{
for (Package::Aliases::const_iterator p1 = package->aliases().begin();
p1 != package->aliases().end();
++p1)
{
if (!p1->second->used())
{
std::string pkg_name = package->package_name();
if (p1->first != pkg_name && p1->first[0] != '.')
{
go_error_at(p1->second->location(),
"imported and not used: %s as %s",
Gogo::message_name(pkg_name).c_str(),
Gogo::message_name(p1->first).c_str());
}
else
go_error_at(p1->second->location(),
"imported and not used: %s",
Gogo::message_name(pkg_name).c_str());
}
}
}
package->clear_used();
}
this->current_file_imported_unsafe_ = false;
}
void
Gogo::queue_specific_type_function(Type* type, Named_type* name, int64_t size,
const std::string& hash_name,
Function_type* hash_fntype,
const std::string& equal_name,
Function_type* equal_fntype)
{
go_assert(!this->specific_type_functions_are_written_);
go_assert(!this->in_global_scope());
Specific_type_function* tsf = new Specific_type_function(type, name, size,
hash_name,
hash_fntype,
equal_name,
equal_fntype);
this->specific_type_functions_.push_back(tsf);
}
class Specific_type_functions : public Traverse
{
public:
Specific_type_functions(Gogo* gogo)
: Traverse(traverse_types),
gogo_(gogo)
{ }
int
type(Type*);
private:
Gogo* gogo_;
};
int
Specific_type_functions::type(Type* t)
{
Named_object* hash_fn;
Named_object* equal_fn;
switch (t->classification())
{
case Type::TYPE_NAMED:
{
Named_type* nt = t->named_type();
if (nt->is_alias())
return TRAVERSE_CONTINUE;
if (t->needs_specific_type_functions(this->gogo_))
t->type_functions(this->gogo_, nt, NULL, NULL, &hash_fn, &equal_fn);
Type* rt = nt->real_type();
if (rt->struct_type() == NULL)
{
if (Type::traverse(rt, this) == TRAVERSE_EXIT)
return TRAVERSE_EXIT;
}
else
{
bool is_defined_elsewhere = nt->named_object()->package() != NULL;
const Struct_field_list* fields = rt->struct_type()->fields();
for (Struct_field_list::const_iterator p = fields->begin();
p != fields->end();
++p)
{
if (is_defined_elsewhere
&& Gogo::is_hidden_name(p->field_name()))
continue;
if (Type::traverse(p->type(), this) == TRAVERSE_EXIT)
return TRAVERSE_EXIT;
}
}
return TRAVERSE_SKIP_COMPONENTS;
}
case Type::TYPE_STRUCT:
case Type::TYPE_ARRAY:
if (t->needs_specific_type_functions(this->gogo_))
t->type_functions(this->gogo_, NULL, NULL, NULL, &hash_fn, &equal_fn);
break;
default:
break;
}
return TRAVERSE_CONTINUE;
}
void
Gogo::write_specific_type_functions()
{
Specific_type_functions stf(this);
this->traverse(&stf);
while (!this->specific_type_functions_.empty())
{
Specific_type_function* tsf = this->specific_type_functions_.back();
this->specific_type_functions_.pop_back();
tsf->type->write_specific_type_functions(this, tsf->name, tsf->size,
tsf->hash_name,
tsf->hash_fntype,
tsf->equal_name,
tsf->equal_fntype);
delete tsf;
}
this->specific_type_functions_are_written_ = true;
}
void
Gogo::traverse(Traverse* traverse)
{
if (this->package_->bindings()->traverse(traverse, true) == TRAVERSE_EXIT)
return;
for (Packages::const_iterator p = this->packages_.begin();
p != this->packages_.end();
++p)
{
if (p->second != this->package_)
{
if (p->second->bindings()->traverse(traverse, true) == TRAVERSE_EXIT)
break;
}
}
}
void
Gogo::add_type_to_verify(Type* type)
{
this->verify_types_.push_back(type);
}
class Verify_types : public Traverse
{
public:
Verify_types()
: Traverse(traverse_types)
{ }
int
type(Type*);
};
int
Verify_types::type(Type* t)
{
if (!t->verify())
return TRAVERSE_SKIP_COMPONENTS;
return TRAVERSE_CONTINUE;
}
void
Gogo::verify_types()
{
Verify_types traverse;
this->traverse(&traverse);
for (std::vector<Type*>::iterator p = this->verify_types_.begin();
p != this->verify_types_.end();
++p)
(*p)->verify();
this->verify_types_.clear();
}
class Lower_parse_tree : public Traverse
{
public:
Lower_parse_tree(Gogo* gogo, Named_object* function)
: Traverse(traverse_variables
| traverse_constants
| traverse_functions
| traverse_statements
| traverse_expressions),
gogo_(gogo), function_(function), iota_value_(-1), inserter_()
{ }
void
set_inserter(const Statement_inserter* inserter)
{ this->inserter_ = *inserter; }
int
variable(Named_object*);
int
constant(Named_object*, bool);
int
function(Named_object*);
int
statement(Block*, size_t* pindex, Statement*);
int
expression(Expression**);
private:
Gogo* gogo_;
Named_object* function_;
int iota_value_;
Statement_inserter inserter_;
};
int
Lower_parse_tree::variable(Named_object* no)
{
if (!no->is_variable())
return TRAVERSE_CONTINUE;
if (no->is_variable() && no->var_value()->is_global())
{
no->var_value()->lower_init_expression(this->gogo_, this->function_,
&this->inserter_);
return TRAVERSE_CONTINUE;
}
if (no->var_value()->has_type())
{
Type* type = no->var_value()->type();
if (type != NULL)
{
if (Type::traverse(type, this) == TRAVERSE_EXIT)
return TRAVERSE_EXIT;
}
}
go_assert(!no->var_value()->has_pre_init());
return TRAVERSE_SKIP_COMPONENTS;
}
int
Lower_parse_tree::constant(Named_object* no, bool)
{
Named_constant* nc = no->const_value();
if (nc->lowering())
return TRAVERSE_CONTINUE;
nc->set_lowering();
go_assert(this->iota_value_ == -1);
this->iota_value_ = nc->iota_value();
nc->traverse_expression(this);
this->iota_value_ = -1;
nc->clear_lowering();
return TRAVERSE_CONTINUE;
}
int
Lower_parse_tree::function(Named_object* no)
{
no->func_value()->set_closure_type();
go_assert(this->function_ == NULL);
this->function_ = no;
int t = no->func_value()->traverse(this);
this->function_ = NULL;
if (t == TRAVERSE_EXIT)
return t;
return TRAVERSE_SKIP_COMPONENTS;
}
int
Lower_parse_tree::statement(Block* block, size_t* pindex, Statement* sorig)
{
if (sorig->is_block_statement())
return TRAVERSE_CONTINUE;
Statement_inserter hold_inserter(this->inserter_);
this->inserter_ = Statement_inserter(block, pindex);
int t = sorig->traverse_contents(this);
if (t == TRAVERSE_EXIT)
{
this->inserter_ = hold_inserter;
return t;
}
Statement* s = sorig;
while (true)
{
Statement* snew = s->lower(this->gogo_, this->function_, block,
&this->inserter_);
if (snew == s)
break;
s = snew;
t = s->traverse_contents(this);
if (t == TRAVERSE_EXIT)
{
this->inserter_ = hold_inserter;
return t;
}
}
if (s != sorig)
block->replace_statement(*pindex, s);
this->inserter_ = hold_inserter;
return TRAVERSE_SKIP_COMPONENTS;
}
int
Lower_parse_tree::expression(Expression** pexpr)
{
if ((*pexpr)->traverse_subexpressions(this) == TRAVERSE_EXIT)
return TRAVERSE_EXIT;
while (true)
{
Expression* e = *pexpr;
Expression* enew = e->lower(this->gogo_, this->function_,
&this->inserter_, this->iota_value_);
if (enew == e)
break;
if (enew->traverse_subexpressions(this) == TRAVERSE_EXIT)
return TRAVERSE_EXIT;
*pexpr = enew;
}
if ((*pexpr)->unknown_expression() == NULL)
Type::traverse((*pexpr)->type(), this);
return TRAVERSE_SKIP_COMPONENTS;
}
void
Gogo::lower_parse_tree()
{
Lower_parse_tree lower_parse_tree(this, NULL);
this->traverse(&lower_parse_tree);
for (std::vector<Type*>::iterator p = this->verify_types_.begin();
p != this->verify_types_.end();
++p)
Type::traverse(*p, &lower_parse_tree);
}
void
Gogo::lower_block(Named_object* function, Block* block)
{
Lower_parse_tree lower_parse_tree(this, function);
block->traverse(&lower_parse_tree);
}
void
Gogo::lower_expression(Named_object* function, Statement_inserter* inserter,
Expression** pexpr)
{
Lower_parse_tree lower_parse_tree(this, function);
if (inserter != NULL)
lower_parse_tree.set_inserter(inserter);
lower_parse_tree.expression(pexpr);
}
void
Gogo::lower_constant(Named_object* no)
{
go_assert(no->is_const());
Lower_parse_tree lower(this, NULL);
lower.constant(no, false);
}
class Create_function_descriptors : public Traverse
{
public:
Create_function_descriptors(Gogo* gogo)
: Traverse(traverse_functions | traverse_expressions),
gogo_(gogo)
{ }
int
function(Named_object*);
int
expression(Expression**);
private:
Gogo* gogo_;
};
int
Create_function_descriptors::function(Named_object* no)
{
if (no->is_function()
&& no->func_value()->enclosing() == NULL
&& !no->func_value()->is_method()
&& !Gogo::is_hidden_name(no->name())
&& !Gogo::is_thunk(no))
no->func_value()->descriptor(this->gogo_, no);
return TRAVERSE_CONTINUE;
}
int
Create_function_descriptors::expression(Expression** pexpr)
{
Expression* expr = *pexpr;
Func_expression* fe = expr->func_expression();
if (fe != NULL)
{
if (fe->closure() != NULL)
return TRAVERSE_CONTINUE;
Named_object* no = fe->named_object();
if (no->is_function() && !no->func_value()->is_method())
no->func_value()->descriptor(this->gogo_, no);
else if (no->is_function_declaration()
&& !no->func_declaration_value()->type()->is_method()
&& !Linemap::is_predeclared_location(no->location()))
no->func_declaration_value()->descriptor(this->gogo_, no);
return TRAVERSE_CONTINUE;
}
Bound_method_expression* bme = expr->bound_method_expression();
if (bme != NULL)
{
Bound_method_expression::create_thunk(this->gogo_, bme->method(),
bme->function());
return TRAVERSE_CONTINUE;
}
Interface_field_reference_expression* ifre =
expr->interface_field_reference_expression();
if (ifre != NULL)
{
Interface_type* type = ifre->expr()->type()->interface_type();
if (type != NULL)
Interface_field_reference_expression::create_thunk(this->gogo_, type,
ifre->name());
return TRAVERSE_CONTINUE;
}
Call_expression* ce = expr->call_expression();
if (ce != NULL)
{
Expression* fn = ce->fn();
if (fn->func_expression() != NULL
|| fn->bound_method_expression() != NULL
|| fn->interface_field_reference_expression() != NULL)
{
Expression_list* args = ce->args();
if (args != NULL)
{
if (args->traverse(this) == TRAVERSE_EXIT)
return TRAVERSE_EXIT;
}
return TRAVERSE_SKIP_COMPONENTS;
}
}
return TRAVERSE_CONTINUE;
}
void
Gogo::create_function_descriptors()
{
std::vector<Named_object*> fndecls;
Bindings* b = this->package_->bindings();
for (Bindings::const_declarations_iterator p = b->begin_declarations();
p != b->end_declarations();
++p)
{
Named_object* no = p->second;
if (no->is_function_declaration()
&& !no->func_declaration_value()->type()->is_method()
&& !Linemap::is_predeclared_location(no->location())
&& !Gogo::is_hidden_name(no->name()))
fndecls.push_back(no);
}
for (std::vector<Named_object*>::const_iterator p = fndecls.begin();
p != fndecls.end();
++p)
(*p)->func_declaration_value()->descriptor(this, *p);
fndecls.clear();
Create_function_descriptors cfd(this);
this->traverse(&cfd);
}
class Finalize_methods : public Traverse
{
public:
Finalize_methods(Gogo* gogo)
: Traverse(traverse_types),
gogo_(gogo)
{ }
int
type(Type*);
private:
Gogo* gogo_;
};
int
Finalize_methods::type(Type* t)
{
switch (t->classification())
{
case Type::TYPE_INTERFACE:
t->interface_type()->finalize_methods();
break;
case Type::TYPE_NAMED:
{
Named_type* nt = t->named_type();
Type* rt = nt->real_type();
if (rt->classification() != Type::TYPE_STRUCT)
{
if (Type::traverse(rt, this) == TRAVERSE_EXIT)
return TRAVERSE_EXIT;
nt->finalize_methods(this->gogo_);
}
else
{
const Struct_field_list* fields = rt->struct_type()->fields();
if (fields != NULL)
{
for (Struct_field_list::const_iterator pf = fields->begin();
pf != fields->end();
++pf)
{
if (pf->is_anonymous())
{
if (Type::traverse(pf->type(), this) == TRAVERSE_EXIT)
return TRAVERSE_EXIT;
}
}
}
nt->finalize_methods(this->gogo_);
if (rt->struct_type()->traverse_field_types(this) == TRAVERSE_EXIT)
return TRAVERSE_EXIT;
}
if (nt->named_object()->package() != NULL && nt->has_any_methods())
{
const Methods* methods = nt->methods();
for (Methods::const_iterator p = methods->begin();
p != methods->end();
++p)
{
if (Type::traverse(p->second->type(), this) == TRAVERSE_EXIT)
return TRAVERSE_EXIT;
}
}
if (nt->named_object()->package() == NULL
&& nt->local_methods() != NULL)
{
const Bindings* methods = nt->local_methods();
for (Bindings::const_declarations_iterator p =
methods->begin_declarations();
p != methods->end_declarations();
p++)
{
if (p->second->is_function_declaration())
{
Type* mt = p->second->func_declaration_value()->type();
if (Type::traverse(mt, this) == TRAVERSE_EXIT)
return TRAVERSE_EXIT;
}
}
}
return TRAVERSE_SKIP_COMPONENTS;
}
case Type::TYPE_STRUCT:
if (t->struct_type()->traverse_field_types(this) == TRAVERSE_EXIT)
return TRAVERSE_EXIT;
t->struct_type()->finalize_methods(this->gogo_);
return TRAVERSE_SKIP_COMPONENTS;
default:
break;
}
return TRAVERSE_CONTINUE;
}
void
Gogo::finalize_methods()
{
Finalize_methods finalize(this);
this->traverse(&finalize);
}
void
Gogo::determine_types()
{
Bindings* bindings = this->current_bindings();
for (Bindings::const_definitions_iterator p = bindings->begin_definitions();
p != bindings->end_definitions();
++p)
{
if ((*p)->is_function())
(*p)->func_value()->determine_types();
else if ((*p)->is_variable())
(*p)->var_value()->determine_type();
else if ((*p)->is_const())
(*p)->const_value()->determine_type();
if (!this->need_init_fn_ && (*p)->is_variable())
{
Variable* variable = (*p)->var_value();
if (!variable->is_global())
;
else if (variable->init() == NULL)
;
else if (variable->type()->interface_type() != NULL)
this->need_init_fn_ = true;
else if (variable->init()->is_constant())
;
else if (!variable->init()->is_composite_literal())
this->need_init_fn_ = true;
else if (variable->init()->is_nonconstant_composite_literal())
this->need_init_fn_ = true;
if (variable->is_global() && variable->type()->has_pointer())
this->need_init_fn_ = true;
}
}
for (Packages::const_iterator p = this->packages_.begin();
p != this->packages_.end();
++p)
p->second->determine_types();
}
class Check_types_traverse : public Traverse
{
public:
Check_types_traverse(Gogo* gogo)
: Traverse(traverse_variables
| traverse_constants
| traverse_functions
| traverse_statements
| traverse_expressions),
gogo_(gogo)
{ }
int
variable(Named_object*);
int
constant(Named_object*, bool);
int
function(Named_object*);
int
statement(Block*, size_t* pindex, Statement*);
int
expression(Expression**);
private:
Gogo* gogo_;
};
int
Check_types_traverse::variable(Named_object* named_object)
{
if (named_object->is_variable())
{
Variable* var = named_object->var_value();
var->type()->base();
Expression* init = var->init();
std::string reason;
if (init != NULL
&& !Type::are_assignable(var->type(), init->type(), &reason))
{
if (reason.empty())
go_error_at(var->location(), "incompatible type in initialization");
else
go_error_at(var->location(),
"incompatible type in initialization (%s)",
reason.c_str());
init = Expression::make_error(named_object->location());
var->clear_init();
}
else if (init != NULL
&& init->func_expression() != NULL)
{
Named_object* no = init->func_expression()->named_object();
Function_type* fntype;
if (no->is_function())
fntype = no->func_value()->type();
else if (no->is_function_declaration())
fntype = no->func_declaration_value()->type();
else
go_unreachable();
if (fntype->is_builtin())
{
go_error_at(init->location(),
"invalid use of special builtin function %qs; "
"must be called",
no->message_name().c_str());
}
}
if (!var->is_used()
&& !var->is_global()
&& !var->is_parameter()
&& !var->is_receiver()
&& !var->type()->is_error()
&& (init == NULL || !init->is_error_expression())
&& !Lex::is_invalid_identifier(named_object->name()))
go_error_at(var->location(), "%qs declared and not used",
named_object->message_name().c_str());
}
return TRAVERSE_CONTINUE;
}
int
Check_types_traverse::constant(Named_object* named_object, bool)
{
Named_constant* constant = named_object->const_value();
Type* ctype = constant->type();
if (ctype->integer_type() == NULL
&& ctype->float_type() == NULL
&& ctype->complex_type() == NULL
&& !ctype->is_boolean_type()
&& !ctype->is_string_type())
{
if (ctype->is_nil_type())
go_error_at(constant->location(), "const initializer cannot be nil");
else if (!ctype->is_error())
go_error_at(constant->location(), "invalid constant type");
constant->set_error();
}
else if (!constant->expr()->is_constant())
{
go_error_at(constant->expr()->location(), "expression is not constant");
constant->set_error();
}
else if (!Type::are_assignable(constant->type(), constant->expr()->type(),
NULL))
{
go_error_at(constant->location(),
"initialization expression has wrong type");
constant->set_error();
}
return TRAVERSE_CONTINUE;
}
int
Check_types_traverse::function(Named_object* no)
{
no->func_value()->check_labels();
return TRAVERSE_CONTINUE;
}
int
Check_types_traverse::statement(Block*, size_t*, Statement* s)
{
s->check_types(this->gogo_);
return TRAVERSE_CONTINUE;
}
int
Check_types_traverse::expression(Expression** expr)
{
(*expr)->check_types(this->gogo_);
return TRAVERSE_CONTINUE;
}
void
Gogo::check_types()
{
Check_types_traverse traverse(this);
this->traverse(&traverse);
Bindings* bindings = this->current_bindings();
for (Bindings::const_declarations_iterator p = bindings->begin_declarations();
p != bindings->end_declarations();
++p)
{
Named_object* no = p->second;
if (no->is_function_declaration())
no->func_declaration_value()->check_types();
}
}
void
Gogo::check_types_in_block(Block* block)
{
Check_types_traverse traverse(this);
block->traverse(&traverse);
}
class Find_shortcut : public Traverse
{
public:
Find_shortcut()
: Traverse(traverse_blocks
| traverse_statements
| traverse_expressions),
found_(NULL)
{ }
Expression**
found() const
{ return this->found_; }
protected:
int
block(Block*)
{ return TRAVERSE_SKIP_COMPONENTS; }
int
statement(Block*, size_t*, Statement*)
{ return TRAVERSE_SKIP_COMPONENTS; }
int
expression(Expression**);
private:
Expression** found_;
};
int
Find_shortcut::expression(Expression** pexpr)
{
Expression* expr = *pexpr;
Binary_expression* be = expr->binary_expression();
if (be == NULL)
return TRAVERSE_CONTINUE;
Operator op = be->op();
if (op != OPERATOR_OROR && op != OPERATOR_ANDAND)
return TRAVERSE_CONTINUE;
go_assert(this->found_ == NULL);
this->found_ = pexpr;
return TRAVERSE_EXIT;
}
class Shortcuts : public Traverse
{
public:
Shortcuts(Gogo* gogo)
: Traverse(traverse_variables
| traverse_statements),
gogo_(gogo)
{ }
protected:
int
variable(Named_object*);
int
statement(Block*, size_t*, Statement*);
private:
Statement*
convert_shortcut(Block* enclosing, Expression** pshortcut);
Gogo* gogo_;
};
int
Shortcuts::statement(Block* block, size_t* pindex, Statement* s)
{
if (s->switch_statement() != NULL)
return TRAVERSE_CONTINUE;
while (true)
{
Find_shortcut find_shortcut;
Variable_declaration_statement* vds = s->variable_declaration_statement();
Expression* init = NULL;
if (vds == NULL)
s->traverse_contents(&find_shortcut);
else
{
init = vds->var()->var_value()->init();
if (init == NULL)
return TRAVERSE_CONTINUE;
init->traverse(&init, &find_shortcut);
}
Expression** pshortcut = find_shortcut.found();
if (pshortcut == NULL)
return TRAVERSE_CONTINUE;
Statement* snew = this->convert_shortcut(block, pshortcut);
block->insert_statement_before(*pindex, snew);
++*pindex;
if (pshortcut == &init)
vds->var()->var_value()->set_init(init);
}
}
int
Shortcuts::variable(Named_object* no)
{
if (no->is_result_variable())
return TRAVERSE_CONTINUE;
Variable* var = no->var_value();
Expression* init = var->init();
if (!var->is_global() || init == NULL)
return TRAVERSE_CONTINUE;
while (true)
{
Find_shortcut find_shortcut;
init->traverse(&init, &find_shortcut);
Expression** pshortcut = find_shortcut.found();
if (pshortcut == NULL)
return TRAVERSE_CONTINUE;
Statement* snew = this->convert_shortcut(NULL, pshortcut);
var->add_preinit_statement(this->gogo_, snew);
if (pshortcut == &init)
var->set_init(init);
}
}
Statement*
Shortcuts::convert_shortcut(Block* enclosing, Expression** pshortcut)
{
Binary_expression* shortcut = (*pshortcut)->binary_expression();
Expression* left = shortcut->left();
Expression* right = shortcut->right();
Location loc = shortcut->location();
Block* retblock = new Block(enclosing, loc);
retblock->set_end_location(loc);
Temporary_statement* ts = Statement::make_temporary(shortcut->type(),
left, loc);
retblock->add_statement(ts);
Block* block = new Block(retblock, loc);
block->set_end_location(loc);
Expression* tmpref = Expression::make_temporary_reference(ts, loc);
Statement* assign = Statement::make_assignment(tmpref, right, loc);
block->add_statement(assign);
Expression* cond = Expression::make_temporary_reference(ts, loc);
if (shortcut->binary_expression()->op() == OPERATOR_OROR)
cond = Expression::make_unary(OPERATOR_NOT, cond, loc);
Statement* if_statement = Statement::make_if_statement(cond, block, NULL,
loc);
retblock->add_statement(if_statement);
*pshortcut = Expression::make_temporary_reference(ts, loc);
delete shortcut;
Shortcuts shortcuts(this->gogo_);
retblock->traverse(&shortcuts);
return Statement::make_block_statement(retblock, loc);
}
void
Gogo::remove_shortcuts()
{
Shortcuts shortcuts(this);
this->traverse(&shortcuts);
}
class Find_eval_ordering : public Traverse
{
private:
typedef std::vector<Expression**> Expression_pointers;
public:
Find_eval_ordering()
: Traverse(traverse_blocks
| traverse_statements
| traverse_expressions),
exprs_()
{ }
size_t
size() const
{ return this->exprs_.size(); }
typedef Expression_pointers::const_iterator const_iterator;
const_iterator
begin() const
{ return this->exprs_.begin(); }
const_iterator
end() const
{ return this->exprs_.end(); }
protected:
int
block(Block*)
{ return TRAVERSE_SKIP_COMPONENTS; }
int
statement(Block*, size_t*, Statement*)
{ return TRAVERSE_SKIP_COMPONENTS; }
int
expression(Expression**);
private:
Expression_pointers exprs_;
};
int
Find_eval_ordering::expression(Expression** expression_pointer)
{
if ((*expression_pointer)->traverse_subexpressions(this) == TRAVERSE_EXIT)
return TRAVERSE_EXIT;
if ((*expression_pointer)->must_eval_in_order())
this->exprs_.push_back(expression_pointer);
return TRAVERSE_SKIP_COMPONENTS;
}
class Order_eval : public Traverse
{
public:
Order_eval(Gogo* gogo)
: Traverse(traverse_variables
| traverse_statements),
gogo_(gogo)
{ }
int
variable(Named_object*);
int
statement(Block*, size_t*, Statement*);
private:
Gogo* gogo_;
};
int
Order_eval::statement(Block* block, size_t* pindex, Statement* stmt)
{
if (stmt->switch_statement() != NULL)
return TRAVERSE_CONTINUE;
Find_eval_ordering find_eval_ordering;
Variable_declaration_statement* vds = stmt->variable_declaration_statement();
Expression* init = NULL;
Expression* orig_init = NULL;
if (vds == NULL)
stmt->traverse_contents(&find_eval_ordering);
else
{
init = vds->var()->var_value()->init();
if (init == NULL)
return TRAVERSE_CONTINUE;
orig_init = init;
Expression::traverse(&init, &find_eval_ordering);
}
size_t c = find_eval_ordering.size();
if (c == 0)
return TRAVERSE_CONTINUE;
if (c == 1)
{
switch (stmt->classification())
{
case Statement::STATEMENT_ASSIGNMENT:
break;
case Statement::STATEMENT_EXPRESSION:
{
Expression* expr = stmt->expression_statement()->expr();
if (expr->call_expression() != NULL
&& expr->call_expression()->result_count() == 0)
break;
return TRAVERSE_CONTINUE;
}
default:
return TRAVERSE_CONTINUE;
}
}
bool is_thunk = stmt->thunk_statement() != NULL;
Expression_statement* es = stmt->expression_statement();
for (Find_eval_ordering::const_iterator p = find_eval_ordering.begin();
p != find_eval_ordering.end();
++p)
{
Expression** pexpr = *p;
if (is_thunk && p + 1 == find_eval_ordering.end())
break;
Location loc = (*pexpr)->location();
Statement* s;
if ((*pexpr)->call_expression() == NULL
|| (*pexpr)->call_expression()->result_count() < 2)
{
Temporary_statement* ts = Statement::make_temporary(NULL, *pexpr,
loc);
s = ts;
*pexpr = Expression::make_temporary_reference(ts, loc);
}
else
{
if (this->remember_expression(*pexpr))
s = NULL;
else if (es != NULL && *pexpr == es->expr())
s = NULL;
else
s = Statement::make_statement(*pexpr, true);
}
if (s != NULL)
{
block->insert_statement_before(*pindex, s);
++*pindex;
}
}
if (init != orig_init)
vds->var()->var_value()->set_init(init);
return TRAVERSE_CONTINUE;
}
int
Order_eval::variable(Named_object* no)
{
if (no->is_result_variable())
return TRAVERSE_CONTINUE;
Variable* var = no->var_value();
Expression* init = var->init();
if (!var->is_global() || init == NULL)
return TRAVERSE_CONTINUE;
Find_eval_ordering find_eval_ordering;
Expression::traverse(&init, &find_eval_ordering);
if (find_eval_ordering.size() <= 1)
{
return TRAVERSE_SKIP_COMPONENTS;
}
Expression* orig_init = init;
for (Find_eval_ordering::const_iterator p = find_eval_ordering.begin();
p != find_eval_ordering.end();
++p)
{
Expression** pexpr = *p;
Location loc = (*pexpr)->location();
Statement* s;
if ((*pexpr)->call_expression() == NULL
|| (*pexpr)->call_expression()->result_count() < 2)
{
Temporary_statement* ts = Statement::make_temporary(NULL, *pexpr,
loc);
s = ts;
*pexpr = Expression::make_temporary_reference(ts, loc);
}
else
{
s = Statement::make_statement(*pexpr, true);
}
var->add_preinit_statement(this->gogo_, s);
}
if (init != orig_init)
var->set_init(init);
return TRAVERSE_SKIP_COMPONENTS;
}
void
Gogo::order_evaluations()
{
Order_eval order_eval(this);
this->traverse(&order_eval);
}
class Flatten : public Traverse
{
public:
Flatten(Gogo* gogo, Named_object* function)
: Traverse(traverse_variables
| traverse_functions
| traverse_statements
| traverse_expressions),
gogo_(gogo), function_(function), inserter_()
{ }
void
set_inserter(const Statement_inserter* inserter)
{ this->inserter_ = *inserter; }
int
variable(Named_object*);
int
function(Named_object*);
int
statement(Block*, size_t* pindex, Statement*);
int
expression(Expression**);
private:
Gogo* gogo_;
Named_object* function_;
Statement_inserter inserter_;
};
int
Flatten::variable(Named_object* no)
{
if (!no->is_variable())
return TRAVERSE_CONTINUE;
if (no->is_variable() && no->var_value()->is_global())
{
no->var_value()->flatten_init_expression(this->gogo_, this->function_,
&this->inserter_);
return TRAVERSE_CONTINUE;
}
if (!no->var_value()->is_parameter()
&& !no->var_value()->is_receiver()
&& !no->var_value()->is_closure()
&& no->var_value()->is_non_escaping_address_taken()
&& !no->var_value()->is_in_heap()
&& no->var_value()->toplevel_decl() == NULL)
{
Block* top_block = function_->func_value()->block();
if (top_block->bindings()->lookup_local(no->name()) != no)
{
Variable* var = no->var_value();
Temporary_statement* ts =
Statement::make_temporary(var->type(), NULL, var->location());
ts->set_is_address_taken();
top_block->add_statement_at_front(ts);
var->set_toplevel_decl(ts);
}
}
go_assert(!no->var_value()->has_pre_init());
return TRAVERSE_SKIP_COMPONENTS;
}
int
Flatten::function(Named_object* no)
{
go_assert(this->function_ == NULL);
this->function_ = no;
int t = no->func_value()->traverse(this);
this->function_ = NULL;
if (t == TRAVERSE_EXIT)
return t;
return TRAVERSE_SKIP_COMPONENTS;
}
int
Flatten::statement(Block* block, size_t* pindex, Statement* sorig)
{
if (sorig->is_block_statement())
return TRAVERSE_CONTINUE;
Statement_inserter hold_inserter(this->inserter_);
this->inserter_ = Statement_inserter(block, pindex);
int t = sorig->traverse_contents(this);
if (t == TRAVERSE_EXIT)
{
this->inserter_ = hold_inserter;
return t;
}
Statement* s = sorig;
while (true)
{
Statement* snew = s->flatten(this->gogo_, this->function_, block,
&this->inserter_);
if (snew == s)
break;
s = snew;
t = s->traverse_contents(this);
if (t == TRAVERSE_EXIT)
{
this->inserter_ = hold_inserter;
return t;
}
}
if (s != sorig)
block->replace_statement(*pindex, s);
this->inserter_ = hold_inserter;
return TRAVERSE_SKIP_COMPONENTS;
}
int
Flatten::expression(Expression** pexpr)
{
while (true)
{
Expression* e = *pexpr;
if (e->traverse_subexpressions(this) == TRAVERSE_EXIT)
return TRAVERSE_EXIT;
Expression* enew = e->flatten(this->gogo_, this->function_,
&this->inserter_);
if (enew == e)
break;
*pexpr = enew;
}
return TRAVERSE_SKIP_COMPONENTS;
}
void
Gogo::flatten_block(Named_object* function, Block* block)
{
Flatten flatten(this, function);
block->traverse(&flatten);
}
void
Gogo::flatten_expression(Named_object* function, Statement_inserter* inserter,
Expression** pexpr)
{
Flatten flatten(this, function);
if (inserter != NULL)
flatten.set_inserter(inserter);
flatten.expression(pexpr);
}
void
Gogo::flatten()
{
Flatten flatten(this, NULL);
this->traverse(&flatten);
}
class Convert_recover : public Traverse
{
public:
Convert_recover(Named_object* arg)
: Traverse(traverse_expressions),
arg_(arg)
{ }
protected:
int
expression(Expression**);
private:
Named_object* arg_;
};
int
Convert_recover::expression(Expression** pp)
{
Call_expression* ce = (*pp)->call_expression();
if (ce != NULL && ce->is_recover_call())
ce->set_recover_arg(Expression::make_var_reference(this->arg_,
ce->location()));
return TRAVERSE_CONTINUE;
}
class Build_recover_thunks : public Traverse
{
public:
Build_recover_thunks(Gogo* gogo)
: Traverse(traverse_functions),
gogo_(gogo)
{ }
int
function(Named_object*);
private:
Expression*
can_recover_arg(Location);
Gogo* gogo_;
};
int
Build_recover_thunks::function(Named_object* orig_no)
{
Function* orig_func = orig_no->func_value();
if (!orig_func->calls_recover()
|| orig_func->is_recover_thunk()
|| orig_func->has_recover_thunk())
return TRAVERSE_CONTINUE;
Gogo* gogo = this->gogo_;
Location location = orig_func->location();
static int count;
char buf[50];
Function_type* orig_fntype = orig_func->type();
Typed_identifier_list* new_params = new Typed_identifier_list();
std::string receiver_name;
if (orig_fntype->is_method())
{
const Typed_identifier* receiver = orig_fntype->receiver();
snprintf(buf, sizeof buf, "rt.%u", count);
++count;
receiver_name = buf;
new_params->push_back(Typed_identifier(receiver_name, receiver->type(),
receiver->location()));
}
const Typed_identifier_list* orig_params = orig_fntype->parameters();
if (orig_params != NULL && !orig_params->empty())
{
for (Typed_identifier_list::const_iterator p = orig_params->begin();
p != orig_params->end();
++p)
{
snprintf(buf, sizeof buf, "pt.%u", count);
++count;
new_params->push_back(Typed_identifier(buf, p->type(),
p->location()));
}
}
snprintf(buf, sizeof buf, "pr.%u", count);
++count;
std::string can_recover_name = buf;
new_params->push_back(Typed_identifier(can_recover_name,
Type::lookup_bool_type(),
orig_fntype->location()));
const Typed_identifier_list* orig_results = orig_fntype->results();
Typed_identifier_list* new_results;
if (orig_results == NULL || orig_results->empty())
new_results = NULL;
else
{
new_results = new Typed_identifier_list();
for (Typed_identifier_list::const_iterator p = orig_results->begin();
p != orig_results->end();
++p)
new_results->push_back(Typed_identifier("", p->type(), p->location()));
}
Function_type *new_fntype = Type::make_function_type(NULL, new_params,
new_results,
orig_fntype->location());
if (orig_fntype->is_varargs())
new_fntype->set_is_varargs();
Type* rtype = NULL;
if (orig_fntype->is_method())
rtype = orig_fntype->receiver()->type();
std::string name(gogo->recover_thunk_name(orig_no->name(), rtype));
Named_object *new_no = gogo->start_function(name, new_fntype, false,
location);
Function *new_func = new_no->func_value();
if (orig_func->enclosing() != NULL)
new_func->set_enclosing(orig_func->enclosing());
Expression* closure = NULL;
if (orig_func->needs_closure())
{
Named_object* orig_closure_no = orig_func->closure_var();
Variable* orig_closure_var = orig_closure_no->var_value();
Variable* new_var = new Variable(orig_closure_var->type(), NULL, false,
false, false, location);
new_var->set_is_closure();
snprintf(buf, sizeof buf, "closure.%u", count);
++count;
Named_object* new_closure_no = Named_object::make_variable(buf, NULL,
new_var);
new_func->set_closure_var(new_closure_no);
closure = Expression::make_var_reference(new_closure_no, location);
}
Expression* fn = Expression::make_func_reference(new_no, closure, location);
Expression_list* args = new Expression_list();
if (new_params != NULL)
{
for (Typed_identifier_list::const_iterator p = new_params->begin();
p + 1 != new_params->end();
++p)
{
Named_object* p_no = gogo->lookup(p->name(), NULL);
go_assert(p_no != NULL
&& p_no->is_variable()
&& p_no->var_value()->is_parameter());
args->push_back(Expression::make_var_reference(p_no, location));
}
}
args->push_back(this->can_recover_arg(location));
gogo->start_block(location);
Call_expression* call = Expression::make_call(fn, args, false, location);
call->set_varargs_are_lowered();
Statement* s = Statement::make_return_from_call(call, location);
s->determine_types();
gogo->add_statement(s);
Block* b = gogo->finish_block(location);
gogo->add_block(b, location);
gogo->lower_block(new_no, b);
gogo->finish_function(location);
new_func->swap_for_recover(orig_func);
orig_func->set_is_recover_thunk();
new_func->set_calls_recover();
new_func->set_has_recover_thunk();
Bindings* orig_bindings = orig_func->block()->bindings();
Bindings* new_bindings = new_func->block()->bindings();
if (orig_fntype->is_method())
{
Named_object* orig_rec_no = orig_bindings->lookup_local(receiver_name);
go_assert(orig_rec_no != NULL
&& orig_rec_no->is_variable()
&& !orig_rec_no->var_value()->is_receiver());
orig_rec_no->var_value()->set_is_receiver();
std::string new_receiver_name(orig_fntype->receiver()->name());
if (new_receiver_name.empty())
{
for (Bindings::const_definitions_iterator p =
new_bindings->begin_definitions();
p != new_bindings->end_definitions();
++p)
{
const std::string& pname((*p)->name());
if (pname[0] == 'r' && pname[1] == '.')
{
new_receiver_name = pname;
break;
}
}
go_assert(!new_receiver_name.empty());
}
Named_object* new_rec_no = new_bindings->lookup_local(new_receiver_name);
if (new_rec_no == NULL)
go_assert(saw_errors());
else
{
go_assert(new_rec_no->is_variable()
&& new_rec_no->var_value()->is_receiver());
new_rec_no->var_value()->set_is_not_receiver();
}
}
Named_object* can_recover_no = orig_bindings->lookup_local(can_recover_name);
go_assert(can_recover_no != NULL
&& can_recover_no->is_variable()
&& can_recover_no->var_value()->is_parameter());
orig_bindings->remove_binding(can_recover_no);
Variable* can_recover_var = new Variable(Type::lookup_bool_type(), NULL,
false, true, false, location);
can_recover_no = new_bindings->add_variable(can_recover_name, NULL,
can_recover_var);
Convert_recover convert_recover(can_recover_no);
new_func->traverse(&convert_recover);
new_func->update_result_variables();
orig_func->update_result_variables();
return TRAVERSE_CONTINUE;
}
Expression*
Build_recover_thunks::can_recover_arg(Location location)
{
static Named_object* builtin_return_address;
if (builtin_return_address == NULL)
builtin_return_address =
Gogo::declare_builtin_rf_address("__builtin_return_address");
static Named_object* can_recover;
if (can_recover == NULL)
{
const Location bloc = Linemap::predeclared_location();
Typed_identifier_list* param_types = new Typed_identifier_list();
Type* voidptr_type = Type::make_pointer_type(Type::make_void_type());
param_types->push_back(Typed_identifier("a", voidptr_type, bloc));
Type* boolean_type = Type::lookup_bool_type();
Typed_identifier_list* results = new Typed_identifier_list();
results->push_back(Typed_identifier("", boolean_type, bloc));
Function_type* fntype = Type::make_function_type(NULL, param_types,
results, bloc);
can_recover =
Named_object::make_function_declaration("runtime_canrecover",
NULL, fntype, bloc);
can_recover->func_declaration_value()->set_asm_name("runtime.canrecover");
}
Expression* fn = Expression::make_func_reference(builtin_return_address,
NULL, location);
Expression* zexpr = Expression::make_integer_ul(0, NULL, location);
Expression_list *args = new Expression_list();
args->push_back(zexpr);
Expression* call = Expression::make_call(fn, args, false, location);
args = new Expression_list();
args->push_back(call);
fn = Expression::make_func_reference(can_recover, NULL, location);
return Expression::make_call(fn, args, false, location);
}
void
Gogo::build_recover_thunks()
{
Build_recover_thunks build_recover_thunks(this);
this->traverse(&build_recover_thunks);
}
Named_object*
Gogo::declare_builtin_rf_address(const char* name)
{
const Location bloc = Linemap::predeclared_location();
Typed_identifier_list* param_types = new Typed_identifier_list();
Type* uint32_type = Type::lookup_integer_type("uint32");
param_types->push_back(Typed_identifier("l", uint32_type, bloc));
Typed_identifier_list* return_types = new Typed_identifier_list();
Type* voidptr_type = Type::make_pointer_type(Type::make_void_type());
return_types->push_back(Typed_identifier("", voidptr_type, bloc));
Function_type* fntype = Type::make_function_type(NULL, param_types,
return_types, bloc);
Named_object* ret = Named_object::make_function_declaration(name, NULL,
fntype, bloc);
ret->func_declaration_value()->set_asm_name(name);
return ret;
}
Expression*
Gogo::runtime_error(int code, Location location)
{
Type* int32_type = Type::lookup_integer_type("int32");
Expression* code_expr = Expression::make_integer_ul(code, int32_type,
location);
return Runtime::make_call(Runtime::RUNTIME_ERROR, location, 1, code_expr);
}
class Build_method_tables : public Traverse
{
public:
Build_method_tables(Gogo* gogo,
const std::vector<Interface_type*>& interfaces)
: Traverse(traverse_types),
gogo_(gogo), interfaces_(interfaces)
{ }
int
type(Type*);
private:
Gogo* gogo_;
const std::vector<Interface_type*>& interfaces_;
};
void
Gogo::build_interface_method_tables()
{
if (saw_errors())
return;
std::vector<Interface_type*> hidden_interfaces;
hidden_interfaces.reserve(this->interface_types_.size());
for (std::vector<Interface_type*>::const_iterator pi =
this->interface_types_.begin();
pi != this->interface_types_.end();
++pi)
{
const Typed_identifier_list* methods = (*pi)->methods();
if (methods == NULL)
continue;
for (Typed_identifier_list::const_iterator pm = methods->begin();
pm != methods->end();
++pm)
{
if (Gogo::is_hidden_name(pm->name()))
{
hidden_interfaces.push_back(*pi);
break;
}
}
}
if (!hidden_interfaces.empty())
{
Build_method_tables bmt(this, hidden_interfaces);
this->traverse(&bmt);
}
this->interface_types_.clear();
}
int
Build_method_tables::type(Type* type)
{
Named_type* nt = type->named_type();
Struct_type* st = type->struct_type();
if (nt != NULL || st != NULL)
{
Translate_context context(this->gogo_, NULL, NULL, NULL);
for (std::vector<Interface_type*>::const_iterator p =
this->interfaces_.begin();
p != this->interfaces_.end();
++p)
{
if (nt != NULL)
{
if ((*p)->implements_interface(Type::make_pointer_type(nt),
NULL))
{
nt->interface_method_table(*p, false)->get_backend(&context);
nt->interface_method_table(*p, true)->get_backend(&context);
}
}
else
{
if ((*p)->implements_interface(Type::make_pointer_type(st),
NULL))
{
st->interface_method_table(*p, false)->get_backend(&context);
st->interface_method_table(*p, true)->get_backend(&context);
}
}
}
}
return TRAVERSE_CONTINUE;
}
Expression*
Gogo::allocate_memory(Type* type, Location location)
{
Expression* td = Expression::make_type_descriptor(type, location);
return Runtime::make_call(Runtime::NEW, location, 1, td);
}
class Check_return_statements_traverse : public Traverse
{
public:
Check_return_statements_traverse()
: Traverse(traverse_functions)
{ }
int
function(Named_object*);
};
int
Check_return_statements_traverse::function(Named_object* no)
{
Function* func = no->func_value();
const Function_type* fntype = func->type();
const Typed_identifier_list* results = fntype->results();
if (results == NULL || results->empty())
return TRAVERSE_CONTINUE;
if (func->block()->may_fall_through())
go_error_at(func->block()->end_location(),
"missing return at end of function");
return TRAVERSE_CONTINUE;
}
void
Gogo::check_return_statements()
{
Check_return_statements_traverse traverse;
this->traverse(&traverse);
}
void
Gogo::do_exports()
{
Stream_to_section stream(this->backend());
std::string prefix;
std::string pkgpath;
if (this->pkgpath_from_option_)
pkgpath = this->pkgpath_;
else if (this->prefix_from_option_)
prefix = this->prefix_;
else if (this->is_main_package())
pkgpath = "main";
else
prefix = "go";
Export exp(&stream);
exp.register_builtin_types(this);
exp.export_globals(this->package_name(),
prefix,
pkgpath,
this->packages_,
this->imports_,
(this->need_init_fn_ && !this->is_main_package()
? this->get_init_fn_name()
: ""),
this->imported_init_fns_,
this->package_->bindings());
if (!this->c_header_.empty() && !saw_errors())
this->write_c_header();
}
void
Gogo::write_c_header()
{
std::ofstream out;
out.open(this->c_header_.c_str());
if (out.fail())
{
go_error_at(Linemap::unknown_location(),
"cannot open %s: %m", this->c_header_.c_str());
return;
}
std::list<Named_object*> types;
Bindings* top = this->package_->bindings();
for (Bindings::const_definitions_iterator p = top->begin_definitions();
p != top->end_definitions();
++p)
{
Named_object* no = *p;
std::string name = Gogo::unpack_hidden_name(no->name());
if (name[0] == '_'
&& (name[1] < 'A' || name[1] > 'Z')
&& (name != "_defer" && name != "_panic"))
continue;
if (no->is_type() && no->type_value()->struct_type() != NULL)
types.push_back(no);
if (no->is_const()
&& no->const_value()->type()->integer_type() != NULL
&& !no->const_value()->is_sink())
{
Numeric_constant nc;
unsigned long val;
if (no->const_value()->expr()->numeric_constant_value(&nc)
&& nc.to_unsigned_long(&val) == Numeric_constant::NC_UL_VALID)
{
out << "#define " << no->message_name() << ' ' << val
<< std::endl;
}
}
}
std::vector<const Named_object*> written;
int loop = 0;
while (!types.empty())
{
Named_object* no = types.front();
types.pop_front();
std::vector<const Named_object*> requires;
std::vector<const Named_object*> declare;
if (!no->type_value()->struct_type()->can_write_to_c_header(&requires,
&declare))
continue;
bool ok = true;
for (std::vector<const Named_object*>::const_iterator pr
= requires.begin();
pr != requires.end() && ok;
++pr)
{
for (std::list<Named_object*>::const_iterator pt = types.begin();
pt != types.end() && ok;
++pt)
if (*pr == *pt)
ok = false;
}
if (!ok)
{
++loop;
if (loop > 10000)
{
go_unreachable();
}
types.push_back(no);
continue;
}
for (std::vector<const Named_object*>::const_iterator pd
= declare.begin();
pd != declare.end();
++pd)
{
if (*pd == no)
continue;
std::vector<const Named_object*> drequires;
std::vector<const Named_object*> ddeclare;
if (!(*pd)->type_value()->struct_type()->
can_write_to_c_header(&drequires, &ddeclare))
continue;
bool done = false;
for (std::vector<const Named_object*>::const_iterator pw
= written.begin();
pw != written.end();
++pw)
{
if (*pw == *pd)
{
done = true;
break;
}
}
if (!done)
{
out << std::endl;
out << "struct " << (*pd)->message_name() << ";" << std::endl;
written.push_back(*pd);
}
}
out << std::endl;
out << "struct " << no->message_name() << " {" << std::endl;
no->type_value()->struct_type()->write_to_c_header(out);
out << "};" << std::endl;
written.push_back(no);
}
out.close();
if (out.fail())
go_error_at(Linemap::unknown_location(),
"error writing to %s: %m", this->c_header_.c_str());
}
class Convert_named_types : public Traverse
{
public:
Convert_named_types(Gogo* gogo)
: Traverse(traverse_blocks),
gogo_(gogo)
{ }
protected:
int
block(Block* block);
private:
Gogo* gogo_;
};
int
Convert_named_types::block(Block* block)
{
this->gogo_->convert_named_types_in_bindings(block->bindings());
return TRAVERSE_CONTINUE;
}
void
Gogo::convert_named_types()
{
this->convert_named_types_in_bindings(this->globals_);
for (Packages::iterator p = this->packages_.begin();
p != this->packages_.end();
++p)
{
Package* package = p->second;
this->convert_named_types_in_bindings(package->bindings());
}
Convert_named_types cnt(this);
this->traverse(&cnt);
Type::make_type_descriptor_type();
Type::make_type_descriptor_ptr_type();
Function_type::make_function_type_descriptor_type();
Pointer_type::make_pointer_type_descriptor_type();
Struct_type::make_struct_type_descriptor_type();
Array_type::make_array_type_descriptor_type();
Array_type::make_slice_type_descriptor_type();
Map_type::make_map_type_descriptor_type();
Channel_type::make_chan_type_descriptor_type();
Interface_type::make_interface_type_descriptor_type();
Expression::make_func_descriptor_type();
Type::convert_builtin_named_types(this);
Runtime::convert_types(this);
this->named_types_are_converted_ = true;
Type::finish_pointer_types(this);
}
void
Gogo::convert_named_types_in_bindings(Bindings* bindings)
{
for (Bindings::const_definitions_iterator p = bindings->begin_definitions();
p != bindings->end_definitions();
++p)
{
if ((*p)->is_type())
(*p)->type_value()->convert(this);
}
}
Function::Function(Function_type* type, Named_object* enclosing, Block* block,
Location location)
: type_(type), enclosing_(enclosing), results_(NULL),
closure_var_(NULL), block_(block), location_(location), labels_(),
local_type_count_(0), descriptor_(NULL), fndecl_(NULL), defer_stack_(NULL),
pragmas_(0), nested_functions_(0), is_sink_(false),
results_are_named_(false), is_unnamed_type_stub_method_(false),
calls_recover_(false), is_recover_thunk_(false), has_recover_thunk_(false),
calls_defer_retaddr_(false), is_type_specific_function_(false),
in_unique_section_(false)
{
}
void
Function::create_result_variables(Gogo* gogo)
{
const Typed_identifier_list* results = this->type_->results();
if (results == NULL || results->empty())
return;
if (!results->front().name().empty())
this->results_are_named_ = true;
this->results_ = new Results();
this->results_->reserve(results->size());
Block* block = this->block_;
int index = 0;
for (Typed_identifier_list::const_iterator p = results->begin();
p != results->end();
++p, ++index)
{
std::string name = p->name();
if (name.empty() || Gogo::is_sink_name(name))
{
static int result_counter;
char buf[100];
snprintf(buf, sizeof buf, "$ret%d", result_counter);
++result_counter;
name = gogo->pack_hidden_name(buf, false);
}
Result_variable* result = new Result_variable(p->type(), this, index,
p->location());
Named_object* no = block->bindings()->add_result_variable(name, result);
if (no->is_result_variable())
this->results_->push_back(no);
else
{
static int dummy_result_count;
char buf[100];
snprintf(buf, sizeof buf, "$dret%d", dummy_result_count);
++dummy_result_count;
name = gogo->pack_hidden_name(buf, false);
no = block->bindings()->add_result_variable(name, result);
go_assert(no->is_result_variable());
this->results_->push_back(no);
}
}
}
void
Function::update_result_variables()
{
if (this->results_ == NULL)
return;
for (Results::iterator p = this->results_->begin();
p != this->results_->end();
++p)
(*p)->result_var_value()->set_function(this);
}
bool
Function::nointerface() const
{
go_assert(this->is_method());
return (this->pragmas_ & GOPRAGMA_NOINTERFACE) != 0;
}
void
Function::set_nointerface()
{
this->pragmas_ |= GOPRAGMA_NOINTERFACE;
}
Named_object*
Function::closure_var()
{
if (this->closure_var_ == NULL)
{
go_assert(this->descriptor_ == NULL);
Location loc = this->type_->location();
Struct_field_list* sfl = new Struct_field_list;
Struct_type* struct_type = Type::make_struct_type(sfl, loc);
struct_type->set_is_struct_incomparable();
Variable* var = new Variable(Type::make_pointer_type(struct_type),
NULL, false, false, false, loc);
var->set_is_used();
var->set_is_closure();
this->closure_var_ = Named_object::make_variable("$closure", NULL, var);
}
return this->closure_var_;
}
void
Function::set_closure_type()
{
if (this->closure_var_ == NULL)
return;
Named_object* closure = this->closure_var_;
Struct_type* st = closure->var_value()->type()->deref()->struct_type();
Type* voidptr_type = Type::make_pointer_type(Type::make_void_type());
st->push_field(Struct_field(Typed_identifier(".f", voidptr_type,
this->location_)));
unsigned int index = 1;
for (Closure_fields::const_iterator p = this->closure_fields_.begin();
p != this->closure_fields_.end();
++p, ++index)
{
Named_object* no = p->first;
char buf[20];
snprintf(buf, sizeof buf, "%u", index);
std::string n = no->name() + buf;
Type* var_type;
if (no->is_variable())
var_type = no->var_value()->type();
else
var_type = no->result_var_value()->type();
Type* field_type = Type::make_pointer_type(var_type);
st->push_field(Struct_field(Typed_identifier(n, field_type, p->second)));
}
}
bool
Function::is_method() const
{
return this->type_->is_method();
}
Label*
Function::add_label_definition(Gogo* gogo, const std::string& label_name,
Location location)
{
Label* lnull = NULL;
std::pair<Labels::iterator, bool> ins =
this->labels_.insert(std::make_pair(label_name, lnull));
Label* label;
if (label_name == "_")
{
label = Label::create_dummy_label();
if (ins.second)
ins.first->second = label;
}
else if (ins.second)
{
label = new Label(label_name);
ins.first->second = label;
}
else
{
label = ins.first->second;
if (label->is_defined())
{
go_error_at(location, "label %qs already defined",
Gogo::message_name(label_name).c_str());
go_inform(label->location(), "previous definition of %qs was here",
Gogo::message_name(label_name).c_str());
return new Label(label_name);
}
}
label->define(location, gogo->bindings_snapshot(location));
const std::vector<Bindings_snapshot*>& refs(label->refs());
for (std::vector<Bindings_snapshot*>::const_iterator p = refs.begin();
p != refs.end();
++p)
(*p)->check_goto_to(gogo->current_block());
label->clear_refs();
return label;
}
Label*
Function::add_label_reference(Gogo* gogo, const std::string& label_name,
Location location, bool issue_goto_errors)
{
Label* lnull = NULL;
std::pair<Labels::iterator, bool> ins =
this->labels_.insert(std::make_pair(label_name, lnull));
Label* label;
if (!ins.second)
{
label = ins.first->second;
}
else
{
go_assert(ins.first->second == NULL);
label = new Label(label_name);
ins.first->second = label;
}
label->set_is_used();
if (issue_goto_errors)
{
Bindings_snapshot* snapshot = label->snapshot();
if (snapshot != NULL)
snapshot->check_goto_from(gogo->current_block(), location);
else
label->add_snapshot_ref(gogo->bindings_snapshot(location));
}
return label;
}
void
Function::check_labels() const
{
for (Labels::const_iterator p = this->labels_.begin();
p != this->labels_.end();
p++)
{
Label* label = p->second;
if (!label->is_used())
go_error_at(label->location(), "label %qs defined and not used",
Gogo::message_name(label->name()).c_str());
}
}
void
Function::swap_for_recover(Function *x)
{
go_assert(this->enclosing_ == x->enclosing_);
std::swap(this->results_, x->results_);
std::swap(this->closure_var_, x->closure_var_);
std::swap(this->block_, x->block_);
go_assert(this->location_ == x->location_);
go_assert(this->fndecl_ == NULL && x->fndecl_ == NULL);
go_assert(this->defer_stack_ == NULL && x->defer_stack_ == NULL);
}
int
Function::traverse(Traverse* traverse)
{
unsigned int traverse_mask = traverse->traverse_mask();
if ((traverse_mask
& (Traverse::traverse_types | Traverse::traverse_expressions))
!= 0)
{
if (Type::traverse(this->type_, traverse) == TRAVERSE_EXIT)
return TRAVERSE_EXIT;
}
if (this->block_ != NULL
&& (traverse_mask
& (Traverse::traverse_variables
| Traverse::traverse_constants
| Traverse::traverse_blocks
| Traverse::traverse_statements
| Traverse::traverse_expressions
| Traverse::traverse_types)) != 0)
{
if (this->block_->traverse(traverse) == TRAVERSE_EXIT)
return TRAVERSE_EXIT;
}
return TRAVERSE_CONTINUE;
}
void
Function::determine_types()
{
if (this->block_ != NULL)
this->block_->determine_types();
}
Expression*
Function::descriptor(Gogo*, Named_object* no)
{
go_assert(!this->is_method());
go_assert(this->closure_var_ == NULL);
if (this->descriptor_ == NULL)
this->descriptor_ = Expression::make_func_descriptor(no);
return this->descriptor_;
}
Expression*
Function::defer_stack(Location location)
{
if (this->defer_stack_ == NULL)
{
Type* t = Type::lookup_bool_type();
Expression* n = Expression::make_boolean(false, location);
this->defer_stack_ = Statement::make_temporary(t, n, location);
this->defer_stack_->set_is_address_taken();
}
Expression* ref = Expression::make_temporary_reference(this->defer_stack_,
location);
return Expression::make_unary(OPERATOR_AND, ref, location);
}
void
Function::export_func(Export* exp, const std::string& name) const
{
Function::export_func_with_type(exp, name, this->type_,
this->is_method() && this->nointerface());
}
void
Function::export_func_with_type(Export* exp, const std::string& name,
const Function_type* fntype, bool nointerface)
{
exp->write_c_string("func ");
if (nointerface)
{
go_assert(fntype->is_method());
exp->write_c_string(" ");
}
if (fntype->is_method())
{
exp->write_c_string("(");
const Typed_identifier* receiver = fntype->receiver();
exp->write_name(receiver->name());
exp->write_escape(receiver->note());
exp->write_c_string(" ");
exp->write_type(receiver->type());
exp->write_c_string(") ");
}
exp->write_string(name);
exp->write_c_string(" (");
const Typed_identifier_list* parameters = fntype->parameters();
if (parameters != NULL)
{
size_t i = 0;
bool is_varargs = fntype->is_varargs();
bool first = true;
for (Typed_identifier_list::const_iterator p = parameters->begin();
p != parameters->end();
++p, ++i)
{
if (first)
first = false;
else
exp->write_c_string(", ");
exp->write_name(p->name());
exp->write_escape(p->note());
exp->write_c_string(" ");
if (!is_varargs || p + 1 != parameters->end())
exp->write_type(p->type());
else
{
exp->write_c_string("...");
exp->write_type(p->type()->array_type()->element_type());
}
}
}
exp->write_c_string(")");
const Typed_identifier_list* results = fntype->results();
if (results != NULL)
{
if (results->size() == 1 && results->begin()->name().empty())
{
exp->write_c_string(" ");
exp->write_type(results->begin()->type());
}
else
{
exp->write_c_string(" (");
bool first = true;
for (Typed_identifier_list::const_iterator p = results->begin();
p != results->end();
++p)
{
if (first)
first = false;
else
exp->write_c_string(", ");
exp->write_name(p->name());
exp->write_escape(p->note());
exp->write_c_string(" ");
exp->write_type(p->type());
}
exp->write_c_string(")");
}
}
exp->write_c_string(";\n");
}
void
Function::import_func(Import* imp, std::string* pname,
Typed_identifier** preceiver,
Typed_identifier_list** pparameters,
Typed_identifier_list** presults,
bool* is_varargs,
bool* nointerface)
{
imp->require_c_string("func ");
*nointerface = false;
if (imp->match_c_string(" ");
*nointerface = true;
go_assert(imp->peek_char() == '(');
}
*preceiver = NULL;
if (imp->peek_char() == '(')
{
imp->require_c_string("(");
std::string name = imp->read_name();
std::string escape_note = imp->read_escape();
imp->require_c_string(" ");
Type* rtype = imp->read_type();
*preceiver = new Typed_identifier(name, rtype, imp->location());
(*preceiver)->set_note(escape_note);
imp->require_c_string(") ");
}
*pname = imp->read_identifier();
Typed_identifier_list* parameters;
*is_varargs = false;
imp->require_c_string(" (");
if (imp->peek_char() == ')')
parameters = NULL;
else
{
parameters = new Typed_identifier_list();
while (true)
{
std::string name = imp->read_name();
std::string escape_note = imp->read_escape();
imp->require_c_string(" ");
if (imp->match_c_string("..."))
{
imp->advance(3);
*is_varargs = true;
}
Type* ptype = imp->read_type();
if (*is_varargs)
ptype = Type::make_array_type(ptype, NULL);
Typed_identifier t = Typed_identifier(name, ptype, imp->location());
t.set_note(escape_note);
parameters->push_back(t);
if (imp->peek_char() != ',')
break;
go_assert(!*is_varargs);
imp->require_c_string(", ");
}
}
imp->require_c_string(")");
*pparameters = parameters;
Typed_identifier_list* results;
if (imp->peek_char() != ' ')
results = NULL;
else
{
results = new Typed_identifier_list();
imp->require_c_string(" ");
if (imp->peek_char() != '(')
{
Type* rtype = imp->read_type();
results->push_back(Typed_identifier("", rtype, imp->location()));
}
else
{
imp->require_c_string("(");
while (true)
{
std::string name = imp->read_name();
std::string note = imp->read_escape();
imp->require_c_string(" ");
Type* rtype = imp->read_type();
Typed_identifier t = Typed_identifier(name, rtype,
imp->location());
t.set_note(note);
results->push_back(t);
if (imp->peek_char() != ',')
break;
imp->require_c_string(", ");
}
imp->require_c_string(")");
}
}
imp->require_c_string(";\n");
*presults = results;
}
Bfunction*
Function::get_or_make_decl(Gogo* gogo, Named_object* no)
{
if (this->fndecl_ == NULL)
{
bool is_visible = false;
bool is_init_fn = false;
Type* rtype = NULL;
if (no->package() != NULL)
;
else if (this->enclosing_ != NULL || Gogo::is_thunk(no))
;
else if (Gogo::unpack_hidden_name(no->name()) == "init"
&& !this->type_->is_method())
;
else if (no->name() == gogo->get_init_fn_name())
{
is_visible = true;
is_init_fn = true;
}
else if (Gogo::unpack_hidden_name(no->name()) == "main"
&& gogo->is_main_package())
is_visible = true;
else if (!Gogo::is_hidden_name(no->name())
|| this->type_->is_method())
{
if (!this->is_unnamed_type_stub_method_)
is_visible = true;
if (this->type_->is_method())
rtype = this->type_->receiver()->type();
}
std::string asm_name;
if (!this->asm_name_.empty())
{
asm_name = this->asm_name_;
is_visible = true;
}
else if (is_init_fn)
{
asm_name = no->name();
}
else
asm_name = gogo->function_asm_name(no->name(), NULL, rtype);
bool is_inlinable = !(this->calls_recover_ || this->is_recover_thunk_);
if (this->calls_defer_retaddr_)
is_inlinable = false;
if ((this->pragmas_ & GOPRAGMA_NOINLINE) != 0)
is_inlinable = false;
bool disable_split_stack = this->is_recover_thunk_;
if ((this->pragmas_ & GOPRAGMA_NOSPLIT) != 0)
disable_split_stack = true;
bool in_unique_section = (this->in_unique_section_
|| (this->is_method() && this->nointerface()));
Btype* functype = this->type_->get_backend_fntype(gogo);
this->fndecl_ =
gogo->backend()->function(functype, no->get_id(gogo), asm_name,
is_visible, false, is_inlinable,
disable_split_stack, false,
in_unique_section, this->location());
}
return this->fndecl_;
}
Bfunction*
Function_declaration::get_or_make_decl(Gogo* gogo, Named_object* no)
{
if (this->fndecl_ == NULL)
{
bool does_not_return = false;
if (!this->asm_name_.empty())
{
Bfunction* builtin_decl =
gogo->backend()->lookup_builtin(this->asm_name_);
if (builtin_decl != NULL)
{
this->fndecl_ = builtin_decl;
return this->fndecl_;
}
if (this->asm_name_ == "runtime.gopanic"
|| this->asm_name_ == "__go_runtime_error")
does_not_return = true;
}
std::string asm_name;
if (this->asm_name_.empty())
{
Type* rtype = NULL;
if (this->fntype_->is_method())
rtype = this->fntype_->receiver()->type();
asm_name = gogo->function_asm_name(no->name(), no->package(), rtype);
}
else if (go_id_needs_encoding(no->get_id(gogo)))
asm_name = go_encode_id(no->get_id(gogo));
Btype* functype = this->fntype_->get_backend_fntype(gogo);
this->fndecl_ =
gogo->backend()->function(functype, no->get_id(gogo), asm_name,
true, true, true, false, does_not_return,
false, this->location());
}
return this->fndecl_;
}
void
Function_declaration::build_backend_descriptor(Gogo* gogo)
{
if (this->descriptor_ != NULL)
{
Translate_context context(gogo, NULL, NULL, NULL);
this->descriptor_->get_backend(&context);
}
}
void
Function_declaration::check_types() const
{
Function_type* fntype = this->type();
if (fntype->receiver() != NULL)
fntype->receiver()->type()->base();
if (fntype->parameters() != NULL)
{
const Typed_identifier_list* params = fntype->parameters();
for (Typed_identifier_list::const_iterator p = params->begin();
p != params->end();
++p)
p->type()->base();
}
}
Bfunction*
Function::get_decl() const
{
go_assert(this->fndecl_ != NULL);
return this->fndecl_;
}
void
Function::build(Gogo* gogo, Named_object* named_function)
{
Translate_context context(gogo, named_function, NULL, NULL);
std::vector<Bvariable*> param_vars;
std::vector<Bvariable*> vars;
std::vector<Bexpression*> var_inits;
std::vector<Statement*> var_decls_stmts;
for (Bindings::const_definitions_iterator p =
this->block_->bindings()->begin_definitions();
p != this->block_->bindings()->end_definitions();
++p)
{
Location loc = (*p)->location();
if ((*p)->is_variable() && (*p)->var_value()->is_parameter())
{
Bvariable* bvar = (*p)->get_backend_variable(gogo, named_function);
Bvariable* parm_bvar = bvar;
if ((*p)->var_value()->is_receiver()
&& (*p)->var_value()->type()->points_to() == NULL)
{
std::string name = (*p)->name() + ".pointer";
Type* var_type = (*p)->var_value()->type();
Variable* parm_var =
new Variable(Type::make_pointer_type(var_type), NULL, false,
true, false, loc);
Named_object* parm_no =
Named_object::make_variable(name, NULL, parm_var);
parm_bvar = parm_no->get_backend_variable(gogo, named_function);
vars.push_back(bvar);
Expression* parm_ref =
Expression::make_var_reference(parm_no, loc);
parm_ref =
Expression::make_dereference(parm_ref,
Expression::NIL_CHECK_NEEDED,
loc);
if ((*p)->var_value()->is_in_heap())
parm_ref = Expression::make_heap_expression(parm_ref, loc);
var_inits.push_back(parm_ref->get_backend(&context));
}
else if ((*p)->var_value()->is_in_heap())
{
std::string parm_name = (*p)->name() + ".param";
Variable* parm_var = new Variable((*p)->var_value()->type(), NULL,
false, true, false, loc);
Named_object* parm_no =
Named_object::make_variable(parm_name, NULL, parm_var);
parm_bvar = parm_no->get_backend_variable(gogo, named_function);
vars.push_back(bvar);
Expression* var_ref =
Expression::make_var_reference(parm_no, loc);
var_ref = Expression::make_heap_expression(var_ref, loc);
var_inits.push_back(var_ref->get_backend(&context));
}
param_vars.push_back(parm_bvar);
}
else if ((*p)->is_result_variable())
{
Bvariable* bvar = (*p)->get_backend_variable(gogo, named_function);
Type* type = (*p)->result_var_value()->type();
Bexpression* init;
if (!(*p)->result_var_value()->is_in_heap())
{
Btype* btype = type->get_backend(gogo);
init = gogo->backend()->zero_expression(btype);
}
else
init = Expression::make_allocation(type,
loc)->get_backend(&context);
vars.push_back(bvar);
var_inits.push_back(init);
}
else if (this->defer_stack_ != NULL
&& (*p)->is_variable()
&& (*p)->var_value()->is_non_escaping_address_taken()
&& !(*p)->var_value()->is_in_heap())
{
Variable* var = (*p)->var_value();
Temporary_statement* ts =
Statement::make_temporary(var->type(), NULL, var->location());
ts->set_is_address_taken();
var->set_toplevel_decl(ts);
var_decls_stmts.push_back(ts);
}
}
if (!gogo->backend()->function_set_parameters(this->fndecl_, param_vars))
{
go_assert(saw_errors());
return;
}
if (this->closure_var_ != NULL)
{
go_assert(this->closure_var_->var_value()->is_closure());
this->closure_var_->get_backend_variable(gogo, named_function);
}
if (this->block_ != NULL)
{
Bblock* var_decls = NULL;
std::vector<Bstatement*> var_decls_bstmt_list;
Bstatement* defer_init = NULL;
if (!vars.empty() || this->defer_stack_ != NULL)
{
var_decls =
gogo->backend()->block(this->fndecl_, NULL, vars,
this->block_->start_location(),
this->block_->end_location());
if (this->defer_stack_ != NULL)
{
Translate_context dcontext(gogo, named_function, this->block_,
var_decls);
defer_init = this->defer_stack_->get_backend(&dcontext);
var_decls_bstmt_list.push_back(defer_init);
for (std::vector<Statement*>::iterator p = var_decls_stmts.begin();
p != var_decls_stmts.end();
++p)
{
Bstatement* bstmt = (*p)->get_backend(&dcontext);
var_decls_bstmt_list.push_back(bstmt);
}
}
}
Translate_context context(gogo, named_function, NULL, NULL);
Bblock* code_block = this->block_->get_backend(&context);
std::vector<Bstatement*> init;
go_assert(vars.size() == var_inits.size());
for (size_t i = 0; i < vars.size(); ++i)
{
Bstatement* init_stmt =
gogo->backend()->init_statement(this->fndecl_, vars[i],
var_inits[i]);
init.push_back(init_stmt);
}
Bstatement* var_init = gogo->backend()->statement_list(init);
Bstatement* code_stmt = gogo->backend()->block_statement(code_block);
code_stmt = gogo->backend()->compound_statement(var_init, code_stmt);
Bstatement* except = NULL;
Bstatement* fini = NULL;
if (defer_init != NULL)
{
this->build_defer_wrapper(gogo, named_function, &except, &fini);
code_stmt =
gogo->backend()->exception_handler_statement(code_stmt,
except, fini,
this->location_);
}
if (var_decls != NULL)
{
var_decls_bstmt_list.push_back(code_stmt);
gogo->backend()->block_add_statements(var_decls, var_decls_bstmt_list);
code_stmt = gogo->backend()->block_statement(var_decls);
}
if (!gogo->backend()->function_set_body(this->fndecl_, code_stmt))
{
go_assert(saw_errors());
return;
}
}
if (this->descriptor_ != NULL)
{
Translate_context context(gogo, NULL, NULL, NULL);
this->descriptor_->get_backend(&context);
}
}
void
Function::build_defer_wrapper(Gogo* gogo, Named_object* named_function,
Bstatement** except, Bstatement** fini)
{
Location end_loc = this->block_->end_location();
std::vector<Bstatement*> stmts;
Expression* call = Runtime::make_call(Runtime::CHECKDEFER, end_loc, 1,
this->defer_stack(end_loc));
Translate_context context(gogo, named_function, NULL, NULL);
Bexpression* defer = call->get_backend(&context);
stmts.push_back(gogo->backend()->expression_statement(this->fndecl_, defer));
Bstatement* ret_bstmt = this->return_value(gogo, named_function, end_loc);
if (ret_bstmt != NULL)
stmts.push_back(ret_bstmt);
go_assert(*except == NULL);
*except = gogo->backend()->statement_list(stmts);
call = Runtime::make_call(Runtime::CHECKDEFER, end_loc, 1,
this->defer_stack(end_loc));
defer = call->get_backend(&context);
call = Runtime::make_call(Runtime::DEFERRETURN, end_loc, 1,
this->defer_stack(end_loc));
Bexpression* undefer = call->get_backend(&context);
Bstatement* function_defer =
gogo->backend()->function_defer_statement(this->fndecl_, undefer, defer,
end_loc);
stmts = std::vector<Bstatement*>(1, function_defer);
if (this->type_->results() != NULL
&& !this->type_->results()->empty()
&& !this->type_->results()->front().name().empty())
{
ret_bstmt = this->return_value(gogo, named_function, end_loc);
Bexpression* nil = Expression::make_nil(end_loc)->get_backend(&context);
Bexpression* ret =
gogo->backend()->compound_expression(ret_bstmt, nil, end_loc);
Expression* ref =
Expression::make_temporary_reference(this->defer_stack_, end_loc);
Bexpression* bref = ref->get_backend(&context);
ret = gogo->backend()->conditional_expression(this->fndecl_,
NULL, bref, ret, NULL,
end_loc);
stmts.push_back(gogo->backend()->expression_statement(this->fndecl_, ret));
}
go_assert(*fini == NULL);
*fini = gogo->backend()->statement_list(stmts);
}
Bstatement*
Function::return_value(Gogo* gogo, Named_object* named_function,
Location location) const
{
const Typed_identifier_list* results = this->type_->results();
if (results == NULL || results->empty())
return NULL;
go_assert(this->results_ != NULL);
if (this->results_->size() != results->size())
{
go_assert(saw_errors());
return gogo->backend()->error_statement();
}
std::vector<Bexpression*> vals(results->size());
for (size_t i = 0; i < vals.size(); ++i)
{
Named_object* no = (*this->results_)[i];
Bvariable* bvar = no->get_backend_variable(gogo, named_function);
Bexpression* val = gogo->backend()->var_expression(bvar, location);
if (no->result_var_value()->is_in_heap())
{
Btype* bt = no->result_var_value()->type()->get_backend(gogo);
val = gogo->backend()->indirect_expression(bt, val, true, location);
}
vals[i] = val;
}
return gogo->backend()->return_statement(this->fndecl_, vals, location);
}
Block::Block(Block* enclosing, Location location)
: enclosing_(enclosing), statements_(),
bindings_(new Bindings(enclosing == NULL
? NULL
: enclosing->bindings())),
start_location_(location),
end_location_(Linemap::unknown_location())
{
}
void
Block::add_statement(Statement* statement)
{
this->statements_.push_back(statement);
}
void
Block::add_statement_at_front(Statement* statement)
{
this->statements_.insert(this->statements_.begin(), statement);
}
void
Block::replace_statement(size_t index, Statement* s)
{
go_assert(index < this->statements_.size());
this->statements_[index] = s;
}
void
Block::insert_statement_before(size_t index, Statement* s)
{
go_assert(index < this->statements_.size());
this->statements_.insert(this->statements_.begin() + index, s);
}
void
Block::insert_statement_after(size_t index, Statement* s)
{
go_assert(index < this->statements_.size());
this->statements_.insert(this->statements_.begin() + index + 1, s);
}
int
Block::traverse(Traverse* traverse)
{
unsigned int traverse_mask = traverse->traverse_mask();
if ((traverse_mask & Traverse::traverse_blocks) != 0)
{
int t = traverse->block(this);
if (t == TRAVERSE_EXIT)
return TRAVERSE_EXIT;
else if (t == TRAVERSE_SKIP_COMPONENTS)
return TRAVERSE_CONTINUE;
}
if ((traverse_mask
& (Traverse::traverse_variables
| Traverse::traverse_constants
| Traverse::traverse_expressions
| Traverse::traverse_types)) != 0)
{
const unsigned int e_or_t = (Traverse::traverse_expressions
| Traverse::traverse_types);
const unsigned int e_or_t_or_s = (e_or_t
| Traverse::traverse_statements);
for (Bindings::const_definitions_iterator pb =
this->bindings_->begin_definitions();
pb != this->bindings_->end_definitions();
++pb)
{
int t = TRAVERSE_CONTINUE;
switch ((*pb)->classification())
{
case Named_object::NAMED_OBJECT_CONST:
if ((traverse_mask & Traverse::traverse_constants) != 0)
t = traverse->constant(*pb, false);
if (t == TRAVERSE_CONTINUE
&& (traverse_mask & e_or_t) != 0)
{
Type* tc = (*pb)->const_value()->type();
if (tc != NULL
&& Type::traverse(tc, traverse) == TRAVERSE_EXIT)
return TRAVERSE_EXIT;
t = (*pb)->const_value()->traverse_expression(traverse);
}
break;
case Named_object::NAMED_OBJECT_VAR:
case Named_object::NAMED_OBJECT_RESULT_VAR:
if ((traverse_mask & Traverse::traverse_variables) != 0)
t = traverse->variable(*pb);
if (t == TRAVERSE_CONTINUE
&& (traverse_mask & e_or_t) != 0)
{
if ((*pb)->is_result_variable()
|| (*pb)->var_value()->has_type())
{
Type* tv = ((*pb)->is_variable()
? (*pb)->var_value()->type()
: (*pb)->result_var_value()->type());
if (tv != NULL
&& Type::traverse(tv, traverse) == TRAVERSE_EXIT)
return TRAVERSE_EXIT;
}
}
if (t == TRAVERSE_CONTINUE
&& (traverse_mask & e_or_t_or_s) != 0
&& (*pb)->is_variable())
t = (*pb)->var_value()->traverse_expression(traverse,
traverse_mask);
break;
case Named_object::NAMED_OBJECT_FUNC:
case Named_object::NAMED_OBJECT_FUNC_DECLARATION:
go_unreachable();
case Named_object::NAMED_OBJECT_TYPE:
if ((traverse_mask & e_or_t) != 0)
t = Type::traverse((*pb)->type_value(), traverse);
break;
case Named_object::NAMED_OBJECT_TYPE_DECLARATION:
case Named_object::NAMED_OBJECT_UNKNOWN:
case Named_object::NAMED_OBJECT_ERRONEOUS:
break;
case Named_object::NAMED_OBJECT_PACKAGE:
case Named_object::NAMED_OBJECT_SINK:
go_unreachable();
default:
go_unreachable();
}
if (t == TRAVERSE_EXIT)
return TRAVERSE_EXIT;
}
}
for (size_t i = 0; i < this->statements_.size(); ++i)
{
if (this->statements_[i]->traverse(this, &i, traverse) == TRAVERSE_EXIT)
return TRAVERSE_EXIT;
}
return TRAVERSE_CONTINUE;
}
void
Block::determine_types()
{
for (Bindings::const_definitions_iterator pb =
this->bindings_->begin_definitions();
pb != this->bindings_->end_definitions();
++pb)
{
if ((*pb)->is_variable())
(*pb)->var_value()->determine_type();
else if ((*pb)->is_const())
(*pb)->const_value()->determine_type();
}
for (std::vector<Statement*>::const_iterator ps = this->statements_.begin();
ps != this->statements_.end();
++ps)
(*ps)->determine_types();
}
bool
Block::may_fall_through() const
{
if (this->statements_.empty())
return true;
return this->statements_.back()->may_fall_through();
}
Bblock*
Block::get_backend(Translate_context* context)
{
Gogo* gogo = context->gogo();
Named_object* function = context->function();
std::vector<Bvariable*> vars;
vars.reserve(this->bindings_->size_definitions());
for (Bindings::const_definitions_iterator pv =
this->bindings_->begin_definitions();
pv != this->bindings_->end_definitions();
++pv)
{
if ((*pv)->is_variable() && !(*pv)->var_value()->is_parameter())
vars.push_back((*pv)->get_backend_variable(gogo, function));
}
go_assert(function != NULL);
Bfunction* bfunction =
function->func_value()->get_or_make_decl(gogo, function);
Bblock* ret = context->backend()->block(bfunction, context->bblock(),
vars, this->start_location_,
this->end_location_);
Translate_context subcontext(gogo, function, this, ret);
std::vector<Bstatement*> bstatements;
bstatements.reserve(this->statements_.size());
for (std::vector<Statement*>::const_iterator p = this->statements_.begin();
p != this->statements_.end();
++p)
bstatements.push_back((*p)->get_backend(&subcontext));
context->backend()->block_add_statements(ret, bstatements);
return ret;
}
Bindings_snapshot::Bindings_snapshot(const Block* b, Location location)
: block_(b), counts_(), location_(location)
{
while (b != NULL)
{
this->counts_.push_back(b->bindings()->size_definitions());
b = b->enclosing();
}
}
void
Bindings_snapshot::check_goto_from(const Block* b, Location loc)
{
size_t dummy;
if (!this->check_goto_block(loc, b, this->block_, &dummy))
return;
this->check_goto_defs(loc, this->block_,
this->block_->bindings()->size_definitions(),
this->counts_[0]);
}
void
Bindings_snapshot::check_goto_to(const Block* b)
{
size_t index;
if (!this->check_goto_block(this->location_, this->block_, b, &index))
return;
this->check_goto_defs(this->location_, b, this->counts_[index],
b->bindings()->size_definitions());
}
bool
Bindings_snapshot::check_goto_block(Location loc, const Block* bfrom,
const Block* bto, size_t* pindex)
{
size_t index = 0;
for (const Block* pb = bfrom; pb != bto; pb = pb->enclosing(), ++index)
{
if (pb == NULL)
{
go_error_at(loc, "goto jumps into block");
go_inform(bto->start_location(), "goto target block starts here");
return false;
}
}
*pindex = index;
return true;
}
void
Bindings_snapshot::check_goto_defs(Location loc, const Block* block,
size_t cfrom, size_t cto)
{
if (cfrom < cto)
{
Bindings::const_definitions_iterator p =
block->bindings()->begin_definitions();
for (size_t i = 0; i < cfrom; ++i)
{
go_assert(p != block->bindings()->end_definitions());
++p;
}
go_assert(p != block->bindings()->end_definitions());
for (; p != block->bindings()->end_definitions(); ++p)
{
if ((*p)->is_variable())
{
std::string n = (*p)->message_name();
go_error_at(loc, "goto jumps over declaration of %qs", n.c_str());
go_inform((*p)->location(), "%qs defined here", n.c_str());
}
}
}
}
bool
Function_declaration::is_method() const
{
return this->fntype_->is_method();
}
bool
Function_declaration::nointerface() const
{
go_assert(this->is_method());
return (this->pragmas_ & GOPRAGMA_NOINTERFACE) != 0;
}
void
Function_declaration::set_nointerface()
{
this->pragmas_ |= GOPRAGMA_NOINTERFACE;
}
Expression*
Function_declaration::descriptor(Gogo*, Named_object* no)
{
go_assert(!this->fntype_->is_method());
if (this->descriptor_ == NULL)
this->descriptor_ = Expression::make_func_descriptor(no);
return this->descriptor_;
}
Variable::Variable(Type* type, Expression* init, bool is_global,
bool is_parameter, bool is_receiver,
Location location)
: type_(type), init_(init), preinit_(NULL), location_(location),
backend_(NULL), is_global_(is_global), is_parameter_(is_parameter),
is_closure_(false), is_receiver_(is_receiver),
is_varargs_parameter_(false), is_used_(false),
is_address_taken_(false), is_non_escaping_address_taken_(false),
seen_(false), init_is_lowered_(false), init_is_flattened_(false),
type_from_init_tuple_(false), type_from_range_index_(false),
type_from_range_value_(false), type_from_chan_element_(false),
is_type_switch_var_(false), determined_type_(false),
in_unique_section_(false), escapes_(true),
toplevel_decl_(NULL)
{
go_assert(type != NULL || init != NULL);
go_assert(!is_parameter || init == NULL);
}
int
Variable::traverse_expression(Traverse* traverse, unsigned int traverse_mask)
{
if (this->preinit_ != NULL)
{
if (this->preinit_->traverse(traverse) == TRAVERSE_EXIT)
return TRAVERSE_EXIT;
}
if (this->init_ != NULL
&& ((traverse_mask
& (Traverse::traverse_expressions | Traverse::traverse_types))
!= 0))
{
if (Expression::traverse(&this->init_, traverse) == TRAVERSE_EXIT)
return TRAVERSE_EXIT;
}
return TRAVERSE_CONTINUE;
}
void
Variable::lower_init_expression(Gogo* gogo, Named_object* function,
Statement_inserter* inserter)
{
Named_object* dep = gogo->var_depends_on(this);
if (dep != NULL && dep->is_variable())
dep->var_value()->lower_init_expression(gogo, function, inserter);
if (this->init_ != NULL && !this->init_is_lowered_)
{
if (this->seen_)
{
return;
}
this->seen_ = true;
Statement_inserter global_inserter;
if (this->is_global_)
{
global_inserter = Statement_inserter(gogo, this);
inserter = &global_inserter;
}
gogo->lower_expression(function, inserter, &this->init_);
this->seen_ = false;
this->init_is_lowered_ = true;
}
}
void
Variable::flatten_init_expression(Gogo* gogo, Named_object* function,
Statement_inserter* inserter)
{
Named_object* dep = gogo->var_depends_on(this);
if (dep != NULL && dep->is_variable())
dep->var_value()->flatten_init_expression(gogo, function, inserter);
if (this->init_ != NULL && !this->init_is_flattened_)
{
if (this->seen_)
{
return;
}
this->seen_ = true;
Statement_inserter global_inserter;
if (this->is_global_)
{
global_inserter = Statement_inserter(gogo, this);
inserter = &global_inserter;
}
gogo->flatten_expression(function, inserter, &this->init_);
if (this->type_ != NULL
&& !Type::are_identical(this->type_, this->init_->type(), false,
NULL)
&& this->init_->type()->interface_type() != NULL
&& !this->init_->is_variable())
{
Temporary_statement* temp =
Statement::make_temporary(NULL, this->init_, this->location_);
inserter->insert(temp);
this->init_ = Expression::make_temporary_reference(temp,
this->location_);
}
this->seen_ = false;
this->init_is_flattened_ = true;
}
}
Block*
Variable::preinit_block(Gogo* gogo)
{
go_assert(this->is_global_);
if (this->preinit_ == NULL)
this->preinit_ = new Block(NULL, this->location());
gogo->set_need_init_fn();
return this->preinit_;
}
void
Variable::add_preinit_statement(Gogo* gogo, Statement* s)
{
Block* b = this->preinit_block(gogo);
b->add_statement(s);
b->set_end_location(s->location());
}
bool
Variable::has_type() const
{
if (this->type_ == NULL)
return false;
if (this->is_type_switch_var_
&& this->type_->is_nil_constant_as_type())
return false;
return true;
}
Type*
Variable::type_from_tuple(Expression* expr, bool report_error) const
{
if (expr->map_index_expression() != NULL)
{
Map_type* mt = expr->map_index_expression()->get_map_type();
if (mt == NULL)
return Type::make_error_type();
return mt->val_type();
}
else if (expr->receive_expression() != NULL)
{
Expression* channel = expr->receive_expression()->channel();
Type* channel_type = channel->type();
if (channel_type->channel_type() == NULL)
return Type::make_error_type();
return channel_type->channel_type()->element_type();
}
else
{
if (report_error)
go_error_at(this->location(), "invalid tuple definition");
return Type::make_error_type();
}
}
Type*
Variable::type_from_range(Expression* expr, bool get_index_type,
bool report_error) const
{
Type* t = expr->type();
if (t->array_type() != NULL
|| (t->points_to() != NULL
&& t->points_to()->array_type() != NULL
&& !t->points_to()->is_slice_type()))
{
if (get_index_type)
return Type::lookup_integer_type("int");
else
return t->deref()->array_type()->element_type();
}
else if (t->is_string_type())
{
if (get_index_type)
return Type::lookup_integer_type("int");
else
return Type::lookup_integer_type("int32");
}
else if (t->map_type() != NULL)
{
if (get_index_type)
return t->map_type()->key_type();
else
return t->map_type()->val_type();
}
else if (t->channel_type() != NULL)
{
if (get_index_type)
return t->channel_type()->element_type();
else
{
if (report_error)
go_error_at(this->location(),
("invalid definition of value variable "
"for channel range"));
return Type::make_error_type();
}
}
else
{
if (report_error)
go_error_at(this->location(), "invalid type for range clause");
return Type::make_error_type();
}
}
Type*
Variable::type_from_chan_element(Expression* expr, bool report_error) const
{
Type* t = expr->type();
if (t->channel_type() != NULL)
return t->channel_type()->element_type();
else
{
if (report_error)
go_error_at(this->location(), "expected channel");
return Type::make_error_type();
}
}
Type*
Variable::type()
{
Type* type = this->type_;
Expression* init = this->init_;
if (this->is_type_switch_var_
&& type != NULL
&& this->type_->is_nil_constant_as_type())
{
Type_guard_expression* tge = this->init_->type_guard_expression();
go_assert(tge != NULL);
init = tge->expr();
type = NULL;
}
if (this->seen_)
{
if (this->type_ == NULL || !this->type_->is_error_type())
{
go_error_at(this->location_, "variable initializer refers to itself");
this->type_ = Type::make_error_type();
}
return this->type_;
}
this->seen_ = true;
if (type != NULL)
;
else if (this->type_from_init_tuple_)
type = this->type_from_tuple(init, false);
else if (this->type_from_range_index_ || this->type_from_range_value_)
type = this->type_from_range(init, this->type_from_range_index_, false);
else if (this->type_from_chan_element_)
type = this->type_from_chan_element(init, false);
else
{
go_assert(init != NULL);
type = init->type();
go_assert(type != NULL);
if (type->is_abstract())
type = type->make_non_abstract_type();
if (type->is_void_type())
type = Type::make_error_type();
}
this->seen_ = false;
return type;
}
Type*
Variable::type() const
{
go_assert(this->type_ != NULL);
return this->type_;
}
void
Variable::determine_type()
{
if (this->determined_type_)
return;
this->determined_type_ = true;
if (this->preinit_ != NULL)
this->preinit_->determine_types();
if (this->is_type_switch_var_
&& this->type_ != NULL
&& this->type_->is_nil_constant_as_type())
{
Type_guard_expression* tge = this->init_->type_guard_expression();
go_assert(tge != NULL);
this->type_ = NULL;
this->init_ = tge->expr();
}
if (this->init_ == NULL)
go_assert(this->type_ != NULL && !this->type_->is_abstract());
else if (this->type_from_init_tuple_)
{
Expression *init = this->init_;
init->determine_type_no_context();
this->type_ = this->type_from_tuple(init, true);
this->init_ = NULL;
}
else if (this->type_from_range_index_ || this->type_from_range_value_)
{
Expression* init = this->init_;
init->determine_type_no_context();
this->type_ = this->type_from_range(init, this->type_from_range_index_,
true);
this->init_ = NULL;
}
else if (this->type_from_chan_element_)
{
Expression* init = this->init_;
init->determine_type_no_context();
this->type_ = this->type_from_chan_element(init, true);
this->init_ = NULL;
}
else
{
Type_context context(this->type_, false);
this->init_->determine_type(&context);
if (this->type_ == NULL)
{
Type* type = this->init_->type();
go_assert(type != NULL);
if (type->is_abstract())
type = type->make_non_abstract_type();
if (type->is_void_type())
{
go_error_at(this->location_, "variable has no type");
type = Type::make_error_type();
}
else if (type->is_nil_type())
{
go_error_at(this->location_, "variable defined to nil type");
type = Type::make_error_type();
}
else if (type->is_call_multiple_result_type())
{
go_error_at(this->location_,
"single variable set to multiple-value function call");
type = Type::make_error_type();
}
this->type_ = type;
}
}
}
Bexpression*
Variable::get_init(Gogo* gogo, Named_object* function)
{
go_assert(this->preinit_ == NULL);
Location loc = this->location();
if (this->init_ == NULL)
{
go_assert(!this->is_parameter_);
if (this->is_global_ || this->is_in_heap())
return NULL;
Btype* btype = this->type()->get_backend(gogo);
return gogo->backend()->zero_expression(btype);
}
else
{
Translate_context context(gogo, function, NULL, NULL);
Expression* init = Expression::make_cast(this->type(), this->init_, loc);
return init->get_backend(&context);
}
}
Bstatement*
Variable::get_init_block(Gogo* gogo, Named_object* function,
Bvariable* var_decl)
{
go_assert(this->preinit_ != NULL);
Translate_context context(gogo, function, NULL, NULL);
Bblock* bblock = this->preinit_->get_backend(&context);
Bfunction* bfunction =
function->func_value()->get_or_make_decl(gogo, function);
Bstatement* decl_init = NULL;
if (this->init_ != NULL)
{
if (var_decl == NULL)
{
Bexpression* init_bexpr = this->init_->get_backend(&context);
decl_init = gogo->backend()->expression_statement(bfunction,
init_bexpr);
}
else
{
Location loc = this->location();
Expression* val_expr =
Expression::make_cast(this->type(), this->init_, loc);
Bexpression* val = val_expr->get_backend(&context);
Bexpression* var_ref =
gogo->backend()->var_expression(var_decl, loc);
decl_init = gogo->backend()->assignment_statement(bfunction, var_ref,
val, loc);
}
}
Bstatement* block_stmt = gogo->backend()->block_statement(bblock);
if (decl_init != NULL)
block_stmt = gogo->backend()->compound_statement(block_stmt, decl_init);
return block_stmt;
}
void
Variable::export_var(Export* exp, const std::string& name) const
{
go_assert(this->is_global_);
exp->write_c_string("var ");
exp->write_string(name);
exp->write_c_string(" ");
exp->write_type(this->type());
exp->write_c_string(";\n");
}
void
Variable::import_var(Import* imp, std::string* pname, Type** ptype)
{
imp->require_c_string("var ");
*pname = imp->read_identifier();
imp->require_c_string(" ");
*ptype = imp->read_type();
imp->require_c_string(";\n");
}
Bvariable*
Variable::get_backend_variable(Gogo* gogo, Named_object* function,
const Package* package, const std::string& name)
{
if (this->backend_ == NULL)
{
Backend* backend = gogo->backend();
Type* type = this->type_;
if (type->is_error_type()
|| (type->is_undefined()
&& (!this->is_global_ || package == NULL)))
this->backend_ = backend->error_variable();
else
{
bool is_parameter = this->is_parameter_;
if (this->is_receiver_ && type->points_to() == NULL)
is_parameter = false;
if (this->is_in_heap())
{
is_parameter = false;
type = Type::make_pointer_type(type);
}
const std::string n = Gogo::unpack_hidden_name(name);
Btype* btype = type->get_backend(gogo);
Bvariable* bvar;
if (Map_type::is_zero_value(this))
bvar = Map_type::backend_zero_value(gogo);
else if (this->is_global_)
{
std::string var_name(package != NULL
? package->package_name()
: gogo->package_name());
var_name.push_back('.');
var_name.append(n);
std::string asm_name(gogo->global_var_asm_name(name, package));
bool is_hidden = Gogo::is_hidden_name(name);
if (gogo->compiling_runtime()
&& var_name == "runtime.writeBarrier")
is_hidden = false;
bvar = backend->global_variable(var_name,
asm_name,
btype,
package != NULL,
is_hidden,
this->in_unique_section_,
this->location_);
}
else if (function == NULL)
{
go_assert(saw_errors());
bvar = backend->error_variable();
}
else
{
Bfunction* bfunction = function->func_value()->get_decl();
bool is_address_taken = (this->is_non_escaping_address_taken_
&& !this->is_in_heap());
if (this->is_closure())
bvar = backend->static_chain_variable(bfunction, n, btype,
this->location_);
else if (is_parameter)
bvar = backend->parameter_variable(bfunction, n, btype,
is_address_taken,
this->location_);
else
{
Bvariable* bvar_decl = NULL;
if (this->toplevel_decl_ != NULL)
{
Translate_context context(gogo, NULL, NULL, NULL);
bvar_decl = this->toplevel_decl_->temporary_statement()
->get_backend_variable(&context);
}
bvar = backend->local_variable(bfunction, n, btype,
bvar_decl,
is_address_taken,
this->location_);
}
}
this->backend_ = bvar;
}
}
return this->backend_;
}
Bvariable*
Result_variable::get_backend_variable(Gogo* gogo, Named_object* function,
const std::string& name)
{
if (this->backend_ == NULL)
{
Backend* backend = gogo->backend();
Type* type = this->type_;
if (type->is_error())
this->backend_ = backend->error_variable();
else
{
if (this->is_in_heap())
type = Type::make_pointer_type(type);
Btype* btype = type->get_backend(gogo);
Bfunction* bfunction = function->func_value()->get_decl();
std::string n = Gogo::unpack_hidden_name(name);
bool is_address_taken = (this->is_non_escaping_address_taken_
&& !this->is_in_heap());
this->backend_ = backend->local_variable(bfunction, n, btype,
NULL, is_address_taken,
this->location_);
}
}
return this->backend_;
}
void
Named_constant::set_type(Type* t)
{
go_assert(this->type_ == NULL || t->is_error_type());
this->type_ = t;
}
int
Named_constant::traverse_expression(Traverse* traverse)
{
return Expression::traverse(&this->expr_, traverse);
}
void
Named_constant::determine_type()
{
if (this->type_ != NULL)
{
Type_context context(this->type_, false);
this->expr_->determine_type(&context);
}
else
{
Type_context context(NULL, true);
this->expr_->determine_type(&context);
this->type_ = this->expr_->type();
go_assert(this->type_ != NULL);
}
}
void
Named_constant::set_error()
{
this->type_ = Type::make_error_type();
this->expr_ = Expression::make_error(this->location_);
}
void
Named_constant::export_const(Export* exp, const std::string& name) const
{
exp->write_c_string("const ");
exp->write_string(name);
exp->write_c_string(" ");
if (!this->type_->is_abstract())
{
exp->write_type(this->type_);
exp->write_c_string(" ");
}
exp->write_c_string("= ");
this->expr()->export_expression(exp);
exp->write_c_string(";\n");
}
void
Named_constant::import_const(Import* imp, std::string* pname, Type** ptype,
Expression** pexpr)
{
imp->require_c_string("const ");
*pname = imp->read_identifier();
imp->require_c_string(" ");
if (imp->peek_char() == '=')
*ptype = NULL;
else
{
*ptype = imp->read_type();
imp->require_c_string(" ");
}
imp->require_c_string("= ");
*pexpr = Expression::import_expression(imp);
imp->require_c_string(";\n");
}
Bexpression*
Named_constant::get_backend(Gogo* gogo, Named_object* const_no)
{
if (this->bconst_ == NULL)
{
Translate_context subcontext(gogo, NULL, NULL, NULL);
Type* type = this->type();
Location loc = this->location();
Expression* const_ref = Expression::make_const_reference(const_no, loc);
Bexpression* const_decl = const_ref->get_backend(&subcontext);
if (type != NULL && type->is_numeric_type())
{
Btype* btype = type->get_backend(gogo);
std::string name = const_no->get_id(gogo);
const_decl =
gogo->backend()->named_constant_expression(btype, name,
const_decl, loc);
}
this->bconst_ = const_decl;
}
return this->bconst_;
}
Named_object*
Type_declaration::add_method(const std::string& name, Function* function)
{
Named_object* ret = Named_object::make_function(name, NULL, function);
this->methods_.push_back(ret);
return ret;
}
Named_object*
Type_declaration::add_method_declaration(const std::string&  name,
Package* package,
Function_type* type,
Location location)
{
Named_object* ret = Named_object::make_function_declaration(name, package,
type, location);
this->methods_.push_back(ret);
return ret;
}
bool
Type_declaration::has_methods() const
{
return !this->methods_.empty();
}
void
Type_declaration::define_methods(Named_type* nt)
{
if (this->methods_.empty())
return;
while (nt->is_alias())
{
Type *t = nt->real_type()->forwarded();
if (t->named_type() != NULL)
nt = t->named_type();
else if (t->forward_declaration_type() != NULL)
{
Named_object* no = t->forward_declaration_type()->named_object();
Type_declaration* td = no->type_declaration_value();
td->methods_.insert(td->methods_.end(), this->methods_.begin(),
this->methods_.end());
this->methods_.clear();
return;
}
else
{
for (std::vector<Named_object*>::const_iterator p =
this->methods_.begin();
p != this->methods_.end();
++p)
go_error_at((*p)->location(),
("invalid receiver type "
"(receiver must be a named type"));
return;
}
}
for (std::vector<Named_object*>::const_iterator p = this->methods_.begin();
p != this->methods_.end();
++p)
{
if (!(*p)->func_value()->is_sink())
nt->add_existing_method(*p);
}
}
bool
Type_declaration::using_type()
{
bool ret = !this->issued_warning_;
this->issued_warning_ = true;
return ret;
}
void
Unknown_name::set_real_named_object(Named_object* no)
{
go_assert(this->real_named_object_ == NULL);
go_assert(!no->is_unknown());
this->real_named_object_ = no;
}
Named_object::Named_object(const std::string& name,
const Package* package,
Classification classification)
: name_(name), package_(package), classification_(classification),
is_redefinition_(false)
{
if (Gogo::is_sink_name(name))
go_assert(classification == NAMED_OBJECT_SINK);
}
Named_object*
Named_object::make_unknown_name(const std::string& name,
Location location)
{
Named_object* named_object = new Named_object(name, NULL,
NAMED_OBJECT_UNKNOWN);
Unknown_name* value = new Unknown_name(location);
named_object->u_.unknown_value = value;
return named_object;
}
Named_object*
Named_object::make_constant(const Typed_identifier& tid,
const Package* package, Expression* expr,
int iota_value)
{
Named_object* named_object = new Named_object(tid.name(), package,
NAMED_OBJECT_CONST);
Named_constant* named_constant = new Named_constant(tid.type(), expr,
iota_value,
tid.location());
named_object->u_.const_value = named_constant;
return named_object;
}
Named_object*
Named_object::make_type(const std::string& name, const Package* package,
Type* type, Location location)
{
Named_object* named_object = new Named_object(name, package,
NAMED_OBJECT_TYPE);
Named_type* named_type = Type::make_named_type(named_object, type, location);
named_object->u_.type_value = named_type;
return named_object;
}
Named_object*
Named_object::make_type_declaration(const std::string& name,
const Package* package,
Location location)
{
Named_object* named_object = new Named_object(name, package,
NAMED_OBJECT_TYPE_DECLARATION);
Type_declaration* type_declaration = new Type_declaration(location);
named_object->u_.type_declaration = type_declaration;
return named_object;
}
Named_object*
Named_object::make_variable(const std::string& name, const Package* package,
Variable* variable)
{
Named_object* named_object = new Named_object(name, package,
NAMED_OBJECT_VAR);
named_object->u_.var_value = variable;
return named_object;
}
Named_object*
Named_object::make_result_variable(const std::string& name,
Result_variable* result)
{
Named_object* named_object = new Named_object(name, NULL,
NAMED_OBJECT_RESULT_VAR);
named_object->u_.result_var_value = result;
return named_object;
}
Named_object*
Named_object::make_sink()
{
return new Named_object("_", NULL, NAMED_OBJECT_SINK);
}
Named_object*
Named_object::make_function(const std::string& name, const Package* package,
Function* function)
{
Named_object* named_object = new Named_object(name, package,
NAMED_OBJECT_FUNC);
named_object->u_.func_value = function;
return named_object;
}
Named_object*
Named_object::make_function_declaration(const std::string& name,
const Package* package,
Function_type* fntype,
Location location)
{
Named_object* named_object = new Named_object(name, package,
NAMED_OBJECT_FUNC_DECLARATION);
Function_declaration *func_decl = new Function_declaration(fntype, location);
named_object->u_.func_declaration_value = func_decl;
return named_object;
}
Named_object*
Named_object::make_package(const std::string& alias, Package* package)
{
Named_object* named_object = new Named_object(alias, NULL,
NAMED_OBJECT_PACKAGE);
named_object->u_.package_value = package;
return named_object;
}
std::string
Named_object::message_name() const
{
if (this->package_ == NULL)
return Gogo::message_name(this->name_);
std::string ret;
if (this->package_->has_package_name())
ret = this->package_->package_name();
else
ret = this->package_->pkgpath();
ret = Gogo::message_name(ret);
ret += '.';
ret += Gogo::message_name(this->name_);
return ret;
}
void
Named_object::set_type_value(Named_type* named_type)
{
go_assert(this->classification_ == NAMED_OBJECT_TYPE_DECLARATION);
Type_declaration* td = this->u_.type_declaration;
td->define_methods(named_type);
unsigned int index;
Named_object* in_function = td->in_function(&index);
if (in_function != NULL)
named_type->set_in_function(in_function, index);
delete td;
this->classification_ = NAMED_OBJECT_TYPE;
this->u_.type_value = named_type;
}
void
Named_object::set_function_value(Function* function)
{
go_assert(this->classification_ == NAMED_OBJECT_FUNC_DECLARATION);
if (this->func_declaration_value()->has_descriptor())
{
Expression* descriptor =
this->func_declaration_value()->descriptor(NULL, NULL);
function->set_descriptor(descriptor);
}
this->classification_ = NAMED_OBJECT_FUNC;
this->u_.func_value = function;
}
void
Named_object::declare_as_type()
{
go_assert(this->classification_ == NAMED_OBJECT_UNKNOWN);
Unknown_name* unk = this->u_.unknown_value;
this->classification_ = NAMED_OBJECT_TYPE_DECLARATION;
this->u_.type_declaration = new Type_declaration(unk->location());
delete unk;
}
Location
Named_object::location() const
{
switch (this->classification_)
{
default:
case NAMED_OBJECT_UNINITIALIZED:
go_unreachable();
case NAMED_OBJECT_ERRONEOUS:
return Linemap::unknown_location();
case NAMED_OBJECT_UNKNOWN:
return this->unknown_value()->location();
case NAMED_OBJECT_CONST:
return this->const_value()->location();
case NAMED_OBJECT_TYPE:
return this->type_value()->location();
case NAMED_OBJECT_TYPE_DECLARATION:
return this->type_declaration_value()->location();
case NAMED_OBJECT_VAR:
return this->var_value()->location();
case NAMED_OBJECT_RESULT_VAR:
return this->result_var_value()->location();
case NAMED_OBJECT_SINK:
go_unreachable();
case NAMED_OBJECT_FUNC:
return this->func_value()->location();
case NAMED_OBJECT_FUNC_DECLARATION:
return this->func_declaration_value()->location();
case NAMED_OBJECT_PACKAGE:
return this->package_value()->location();
}
}
void
Named_object::export_named_object(Export* exp) const
{
switch (this->classification_)
{
default:
case NAMED_OBJECT_UNINITIALIZED:
case NAMED_OBJECT_UNKNOWN:
go_unreachable();
case NAMED_OBJECT_ERRONEOUS:
break;
case NAMED_OBJECT_CONST:
this->const_value()->export_const(exp, this->name_);
break;
case NAMED_OBJECT_TYPE:
this->type_value()->export_named_type(exp, this->name_);
break;
case NAMED_OBJECT_TYPE_DECLARATION:
go_error_at(this->type_declaration_value()->location(),
"attempt to export %<%s%> which was declared but not defined",
this->message_name().c_str());
break;
case NAMED_OBJECT_FUNC_DECLARATION:
this->func_declaration_value()->export_func(exp, this->name_);
break;
case NAMED_OBJECT_VAR:
this->var_value()->export_var(exp, this->name_);
break;
case NAMED_OBJECT_RESULT_VAR:
case NAMED_OBJECT_SINK:
go_unreachable();
case NAMED_OBJECT_FUNC:
this->func_value()->export_func(exp, this->name_);
break;
}
}
Bvariable*
Named_object::get_backend_variable(Gogo* gogo, Named_object* function)
{
if (this->classification_ == NAMED_OBJECT_VAR)
return this->var_value()->get_backend_variable(gogo, function,
this->package_, this->name_);
else if (this->classification_ == NAMED_OBJECT_RESULT_VAR)
return this->result_var_value()->get_backend_variable(gogo, function,
this->name_);
else
go_unreachable();
}
std::string
Named_object::get_id(Gogo* gogo)
{
go_assert(!this->is_variable()
&& !this->is_result_variable()
&& !this->is_type());
std::string decl_name;
if (this->is_function_declaration()
&& !this->func_declaration_value()->asm_name().empty())
decl_name = this->func_declaration_value()->asm_name();
else
{
std::string package_name;
if (this->package_ == NULL)
package_name = gogo->package_name();
else
package_name = this->package_->package_name();
decl_name = package_name + '.' + Gogo::unpack_hidden_name(this->name_);
Function_type* fntype;
if (this->is_function())
fntype = this->func_value()->type();
else if (this->is_function_declaration())
fntype = this->func_declaration_value()->type();
else
fntype = NULL;
if (fntype != NULL && fntype->is_method())
{
decl_name.push_back('.');
decl_name.append(fntype->receiver()->type()->mangled_name(gogo));
}
}
return decl_name;
}
void
Named_object::get_backend(Gogo* gogo, std::vector<Bexpression*>& const_decls,
std::vector<Btype*>& type_decls,
std::vector<Bfunction*>& func_decls)
{
if (this->is_redefinition_)
{
go_assert(saw_errors());
return;
}
switch (this->classification_)
{
case NAMED_OBJECT_CONST:
if (!Gogo::is_erroneous_name(this->name_))
const_decls.push_back(this->u_.const_value->get_backend(gogo, this));
break;
case NAMED_OBJECT_TYPE:
{
Named_type* named_type = this->u_.type_value;
if (!Gogo::is_erroneous_name(this->name_))
type_decls.push_back(named_type->get_backend(gogo));
if (this->package_ == NULL && !saw_errors())
{
named_type->
type_descriptor_pointer(gogo, Linemap::predeclared_location());
named_type->gc_symbol_pointer(gogo);
Type* pn = Type::make_pointer_type(named_type);
pn->type_descriptor_pointer(gogo, Linemap::predeclared_location());
pn->gc_symbol_pointer(gogo);
}
}
break;
case NAMED_OBJECT_TYPE_DECLARATION:
go_error_at(Linemap::unknown_location(),
"reference to undefined type %qs",
this->message_name().c_str());
return;
case NAMED_OBJECT_VAR:
case NAMED_OBJECT_RESULT_VAR:
case NAMED_OBJECT_SINK:
go_unreachable();
case NAMED_OBJECT_FUNC:
{
Function* func = this->u_.func_value;
if (!Gogo::is_erroneous_name(this->name_))
func_decls.push_back(func->get_or_make_decl(gogo, this));
if (func->block() != NULL)
func->build(gogo, this);
}
break;
case NAMED_OBJECT_ERRONEOUS:
break;
default:
go_unreachable();
}
}
Bindings::Bindings(Bindings* enclosing)
: enclosing_(enclosing), named_objects_(), bindings_()
{
}
void
Bindings::clear_file_scope(Gogo* gogo)
{
Contour::iterator p = this->bindings_.begin();
while (p != this->bindings_.end())
{
bool keep;
if (p->second->package() != NULL)
keep = false;
else if (p->second->is_package())
keep = false;
else if (p->second->is_function()
&& !p->second->func_value()->type()->is_method()
&& Gogo::unpack_hidden_name(p->second->name()) == "init")
keep = false;
else
keep = true;
if (keep)
++p;
else
{
gogo->add_file_block_name(p->second->name(), p->second->location());
p = this->bindings_.erase(p);
}
}
}
Named_object*
Bindings::lookup(const std::string& name) const
{
Contour::const_iterator p = this->bindings_.find(name);
if (p != this->bindings_.end())
return p->second->resolve();
else if (this->enclosing_ != NULL)
return this->enclosing_->lookup(name);
else
return NULL;
}
Named_object*
Bindings::lookup_local(const std::string& name) const
{
Contour::const_iterator p = this->bindings_.find(name);
if (p == this->bindings_.end())
return NULL;
return p->second;
}
void
Bindings::remove_binding(Named_object* no)
{
Contour::iterator pb = this->bindings_.find(no->name());
go_assert(pb != this->bindings_.end());
this->bindings_.erase(pb);
for (std::vector<Named_object*>::iterator pn = this->named_objects_.begin();
pn != this->named_objects_.end();
++pn)
{
if (*pn == no)
{
this->named_objects_.erase(pn);
return;
}
}
go_unreachable();
}
void
Bindings::add_method(Named_object* method)
{
this->named_objects_.push_back(method);
}
Named_object*
Bindings::add_named_object_to_contour(Contour* contour,
Named_object* named_object)
{
go_assert(named_object == named_object->resolve());
const std::string& name(named_object->name());
go_assert(!Gogo::is_sink_name(name));
std::pair<Contour::iterator, bool> ins =
contour->insert(std::make_pair(name, named_object));
if (!ins.second)
{
if (named_object->package() != NULL
&& ins.first->second->package() == named_object->package()
&& (ins.first->second->classification()
== named_object->classification()))
{
return ins.first->second;
}
ins.first->second = this->new_definition(ins.first->second,
named_object);
return ins.first->second;
}
else
{
if (!named_object->is_type_declaration()
&& !named_object->is_function_declaration()
&& !named_object->is_unknown())
this->named_objects_.push_back(named_object);
return named_object;
}
}
Named_object*
Bindings::new_definition(Named_object* old_object, Named_object* new_object)
{
if (new_object->is_erroneous() && !old_object->is_erroneous())
return new_object;
std::string reason;
switch (old_object->classification())
{
default:
case Named_object::NAMED_OBJECT_UNINITIALIZED:
go_unreachable();
case Named_object::NAMED_OBJECT_ERRONEOUS:
return old_object;
case Named_object::NAMED_OBJECT_UNKNOWN:
{
Named_object* real = old_object->unknown_value()->real_named_object();
if (real != NULL)
return this->new_definition(real, new_object);
go_assert(!new_object->is_unknown());
old_object->unknown_value()->set_real_named_object(new_object);
if (!new_object->is_type_declaration()
&& !new_object->is_function_declaration())
this->named_objects_.push_back(new_object);
return new_object;
}
case Named_object::NAMED_OBJECT_CONST:
break;
case Named_object::NAMED_OBJECT_TYPE:
if (new_object->is_type_declaration())
return old_object;
break;
case Named_object::NAMED_OBJECT_TYPE_DECLARATION:
if (new_object->is_type_declaration())
return old_object;
if (new_object->is_type())
{
old_object->set_type_value(new_object->type_value());
new_object->type_value()->set_named_object(old_object);
this->named_objects_.push_back(old_object);
return old_object;
}
break;
case Named_object::NAMED_OBJECT_VAR:
case Named_object::NAMED_OBJECT_RESULT_VAR:
if ((new_object->is_variable()
&& new_object->var_value()->is_parameter())
|| new_object->is_result_variable())
return old_object;
break;
case Named_object::NAMED_OBJECT_SINK:
go_unreachable();
case Named_object::NAMED_OBJECT_FUNC:
break;
case Named_object::NAMED_OBJECT_FUNC_DECLARATION:
{
if (new_object->is_function()
&& ((Linemap::is_predeclared_location(old_object->location())
&& Linemap::is_predeclared_location(new_object->location()))
|| (Gogo::unpack_hidden_name(old_object->name()) == "main"
&& Linemap::is_unknown_location(old_object->location()))))
{
Function_type* old_type =
old_object->func_declaration_value()->type();
Function_type* new_type = new_object->func_value()->type();
if (old_type->is_valid_redeclaration(new_type, &reason))
{
Function_declaration* fd =
old_object->func_declaration_value();
go_assert(fd->asm_name().empty());
old_object->set_function_value(new_object->func_value());
this->named_objects_.push_back(old_object);
return old_object;
}
}
}
break;
case Named_object::NAMED_OBJECT_PACKAGE:
break;
}
std::string n = old_object->message_name();
if (reason.empty())
go_error_at(new_object->location(), "redefinition of %qs", n.c_str());
else
go_error_at(new_object->location(), "redefinition of %qs: %s", n.c_str(),
reason.c_str());
old_object->set_is_redefinition();
new_object->set_is_redefinition();
if (!Linemap::is_unknown_location(old_object->location())
&& !Linemap::is_predeclared_location(old_object->location()))
go_inform(old_object->location(), "previous definition of %qs was here",
n.c_str());
return old_object;
}
Named_object*
Bindings::add_named_type(Named_type* named_type)
{
return this->add_named_object(named_type->named_object());
}
Named_object*
Bindings::add_function(const std::string& name, const Package* package,
Function* function)
{
return this->add_named_object(Named_object::make_function(name, package,
function));
}
Named_object*
Bindings::add_function_declaration(const std::string& name,
const Package* package,
Function_type* type,
Location location)
{
Named_object* no = Named_object::make_function_declaration(name, package,
type, location);
return this->add_named_object(no);
}
void
Bindings::define_type(Named_object* no, Named_type* type)
{
no->set_type_value(type);
this->named_objects_.push_back(no);
}
void
Bindings::mark_locals_used()
{
for (std::vector<Named_object*>::iterator p = this->named_objects_.begin();
p != this->named_objects_.end();
++p)
if ((*p)->is_variable())
(*p)->var_value()->set_is_used();
}
int
Bindings::traverse(Traverse* traverse, bool is_global)
{
unsigned int traverse_mask = traverse->traverse_mask();
const unsigned int e_or_t = (Traverse::traverse_expressions
| Traverse::traverse_types);
const unsigned int e_or_t_or_s = (e_or_t
| Traverse::traverse_statements);
for (size_t i = 0; i < this->named_objects_.size(); ++i)
{
Named_object* p = this->named_objects_[i];
int t = TRAVERSE_CONTINUE;
switch (p->classification())
{
case Named_object::NAMED_OBJECT_CONST:
if ((traverse_mask & Traverse::traverse_constants) != 0)
t = traverse->constant(p, is_global);
if (t == TRAVERSE_CONTINUE
&& (traverse_mask & e_or_t) != 0)
{
Type* tc = p->const_value()->type();
if (tc != NULL
&& Type::traverse(tc, traverse) == TRAVERSE_EXIT)
return TRAVERSE_EXIT;
t = p->const_value()->traverse_expression(traverse);
}
break;
case Named_object::NAMED_OBJECT_VAR:
case Named_object::NAMED_OBJECT_RESULT_VAR:
if ((traverse_mask & Traverse::traverse_variables) != 0)
t = traverse->variable(p);
if (t == TRAVERSE_CONTINUE
&& (traverse_mask & e_or_t) != 0)
{
if (p->is_result_variable()
|| p->var_value()->has_type())
{
Type* tv = (p->is_variable()
? p->var_value()->type()
: p->result_var_value()->type());
if (tv != NULL
&& Type::traverse(tv, traverse) == TRAVERSE_EXIT)
return TRAVERSE_EXIT;
}
}
if (t == TRAVERSE_CONTINUE
&& (traverse_mask & e_or_t_or_s) != 0
&& p->is_variable())
t = p->var_value()->traverse_expression(traverse, traverse_mask);
break;
case Named_object::NAMED_OBJECT_FUNC:
if ((traverse_mask & Traverse::traverse_functions) != 0)
t = traverse->function(p);
if (t == TRAVERSE_CONTINUE
&& (traverse_mask
& (Traverse::traverse_variables
| Traverse::traverse_constants
| Traverse::traverse_functions
| Traverse::traverse_blocks
| Traverse::traverse_statements
| Traverse::traverse_expressions
| Traverse::traverse_types)) != 0)
t = p->func_value()->traverse(traverse);
break;
case Named_object::NAMED_OBJECT_PACKAGE:
go_assert(is_global);
break;
case Named_object::NAMED_OBJECT_TYPE:
if ((traverse_mask & e_or_t) != 0)
t = Type::traverse(p->type_value(), traverse);
break;
case Named_object::NAMED_OBJECT_TYPE_DECLARATION:
case Named_object::NAMED_OBJECT_FUNC_DECLARATION:
case Named_object::NAMED_OBJECT_UNKNOWN:
case Named_object::NAMED_OBJECT_ERRONEOUS:
break;
case Named_object::NAMED_OBJECT_SINK:
default:
go_unreachable();
}
if (t == TRAVERSE_EXIT)
return TRAVERSE_EXIT;
}
if ((traverse_mask & e_or_t) != 0)
{
for (Bindings::const_declarations_iterator p =
this->begin_declarations();
p != this->end_declarations();
++p)
{
if (p->second->is_function_declaration())
{
if (Type::traverse(p->second->func_declaration_value()->type(),
traverse)
== TRAVERSE_EXIT)
return TRAVERSE_EXIT;
}
else if (p->second->is_type_declaration())
{
const std::vector<Named_object*>* methods =
p->second->type_declaration_value()->methods();
for (std::vector<Named_object*>::const_iterator pm =
methods->begin();
pm != methods->end();
pm++)
{
Named_object* no = *pm;
Type *t;
if (no->is_function())
t = no->func_value()->type();
else if (no->is_function_declaration())
t = no->func_declaration_value()->type();
else
continue;
if (Type::traverse(t, traverse) == TRAVERSE_EXIT)
return TRAVERSE_EXIT;
}
}
}
}
if ((traverse_mask & Traverse::traverse_func_declarations) != 0)
{
for (Bindings::const_declarations_iterator p = this->begin_declarations();
p != this->end_declarations();
++p)
{
if (p->second->is_function_declaration())
{
if (traverse->function_declaration(p->second) == TRAVERSE_EXIT)
return TRAVERSE_EXIT;
}
}
}
return TRAVERSE_CONTINUE;
}
void
Label::clear_refs()
{
for (std::vector<Bindings_snapshot*>::iterator p = this->refs_.begin();
p != this->refs_.end();
++p)
delete *p;
this->refs_.clear();
}
Blabel*
Label::get_backend_label(Translate_context* context)
{
if (this->blabel_ == NULL)
{
Function* function = context->function()->func_value();
Bfunction* bfunction = function->get_decl();
this->blabel_ = context->backend()->label(bfunction, this->name_,
this->location_);
}
return this->blabel_;
}
Bexpression*
Label::get_addr(Translate_context* context, Location location)
{
Blabel* label = this->get_backend_label(context);
return context->backend()->label_address(label, location);
}
Label*
Label::create_dummy_label()
{
static Label* dummy_label;
if (dummy_label == NULL)
{
dummy_label = new Label("_");
dummy_label->set_is_used();
}
return dummy_label;
}
Blabel*
Unnamed_label::get_blabel(Translate_context* context)
{
if (this->blabel_ == NULL)
{
Function* function = context->function()->func_value();
Bfunction* bfunction = function->get_decl();
this->blabel_ = context->backend()->label(bfunction, "",
this->location_);
}
return this->blabel_;
}
Bstatement*
Unnamed_label::get_definition(Translate_context* context)
{
Blabel* blabel = this->get_blabel(context);
return context->backend()->label_definition_statement(blabel);
}
Bstatement*
Unnamed_label::get_goto(Translate_context* context, Location location)
{
Blabel* blabel = this->get_blabel(context);
return context->backend()->goto_statement(blabel, location);
}
Package::Package(const std::string& pkgpath,
const std::string& pkgpath_symbol, Location location)
: pkgpath_(pkgpath), pkgpath_symbol_(pkgpath_symbol),
package_name_(), bindings_(new Bindings(NULL)),
location_(location)
{
go_assert(!pkgpath.empty());
}
void
Package::set_package_name(const std::string& package_name, Location location)
{
go_assert(!package_name.empty());
if (this->package_name_.empty())
this->package_name_ = package_name;
else if (this->package_name_ != package_name)
go_error_at(location,
("saw two different packages with "
"the same package path %s: %s, %s"),
this->pkgpath_.c_str(), this->package_name_.c_str(),
package_name.c_str());
}
std::string
Package::pkgpath_symbol() const
{
if (this->pkgpath_symbol_.empty())
return Gogo::pkgpath_for_symbol(this->pkgpath_);
return this->pkgpath_symbol_;
}
void
Package::set_pkgpath_symbol(const std::string& pkgpath_symbol)
{
go_assert(!pkgpath_symbol.empty());
if (this->pkgpath_symbol_.empty())
this->pkgpath_symbol_ = pkgpath_symbol;
else
go_assert(this->pkgpath_symbol_ == pkgpath_symbol);
}
void
Package::note_usage(const std::string& alias) const
{
Aliases::const_iterator p = this->aliases_.find(alias);
go_assert(p != this->aliases_.end());
p->second->note_usage();
}
void
Package::forget_usage(Expression* usage) const
{
if (this->fake_uses_.empty())
return;
std::set<Expression*>::iterator p = this->fake_uses_.find(usage);
go_assert(p != this->fake_uses_.end());
this->fake_uses_.erase(p);
if (this->fake_uses_.empty())
go_error_at(this->location(), "imported and not used: %s",
Gogo::message_name(this->package_name()).c_str());
}
void
Package::clear_used()
{
std::string dot_alias = "." + this->package_name();
Aliases::const_iterator p = this->aliases_.find(dot_alias);
if (p != this->aliases_.end() && p->second->used() > this->fake_uses_.size())
this->fake_uses_.clear();
this->aliases_.clear();
}
Package_alias*
Package::add_alias(const std::string& alias, Location location)
{
Aliases::const_iterator p = this->aliases_.find(alias);
if (p == this->aliases_.end())
{
std::pair<Aliases::iterator, bool> ret;
ret = this->aliases_.insert(std::make_pair(alias,
new Package_alias(location)));
p = ret.first;
}
return p->second;
}
void
Package::determine_types()
{
Bindings* bindings = this->bindings_;
for (Bindings::const_definitions_iterator p = bindings->begin_definitions();
p != bindings->end_definitions();
++p)
{
if ((*p)->is_const())
(*p)->const_value()->determine_type();
}
}
Traverse::~Traverse()
{
if (this->types_seen_ != NULL)
delete this->types_seen_;
if (this->expressions_seen_ != NULL)
delete this->expressions_seen_;
}
bool
Traverse::remember_type(const Type* type)
{
if (type->is_error_type())
return true;
go_assert((this->traverse_mask() & traverse_types) != 0
|| (this->traverse_mask() & traverse_expressions) != 0);
if (type->classification() != Type::TYPE_NAMED
&& type->classification() != Type::TYPE_INTERFACE)
return false;
if (this->types_seen_ == NULL)
this->types_seen_ = new Types_seen();
std::pair<Types_seen::iterator, bool> ins = this->types_seen_->insert(type);
return !ins.second;
}
bool
Traverse::remember_expression(const Expression* expression)
{
if (this->expressions_seen_ == NULL)
this->expressions_seen_ = new Expressions_seen();
std::pair<Expressions_seen::iterator, bool> ins =
this->expressions_seen_->insert(expression);
return !ins.second;
}
int
Traverse::variable(Named_object*)
{
go_unreachable();
}
int
Traverse::constant(Named_object*, bool)
{
go_unreachable();
}
int
Traverse::function(Named_object*)
{
go_unreachable();
}
int
Traverse::block(Block*)
{
go_unreachable();
}
int
Traverse::statement(Block*, size_t*, Statement*)
{
go_unreachable();
}
int
Traverse::expression(Expression**)
{
go_unreachable();
}
int
Traverse::type(Type*)
{
go_unreachable();
}
int
Traverse::function_declaration(Named_object*)
{
go_unreachable();
}
void
Statement_inserter::insert(Statement* s)
{
if (this->block_ != NULL)
{
go_assert(this->pindex_ != NULL);
this->block_->insert_statement_before(*this->pindex_, s);
++*this->pindex_;
}
else if (this->var_ != NULL)
this->var_->add_preinit_statement(this->gogo_, s);
else
go_assert(saw_errors());
}
