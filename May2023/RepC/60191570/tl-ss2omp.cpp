#include "tl-ss2omp.hpp"
#include "tl-augmented-symbol.hpp"
#include "tl-region.hpp"
#include "tl-region-list.hpp"
#include "tl-parameter-region-list.hpp"
#include "cxx-utils.h"
namespace TL
{
static void add_to_clause(Region & region, 
const Type& parameter, 
const ParameterDeclaration& parameter_decl, 
Source& clause_args,
const std::string& dir_str,
AST_t context_tree,
ScopeLink sl)
{
DEBUG_CODE()
{
std::cerr << "Parameter region '" << parameter_decl.get_name() << "'" << std::endl
<< " direction=" << dir_str << std::endl
<< " dimension_count=" << region.get_dimension_count() << std::endl
<< " is_full=" << region.is_full() 
<< std::endl;
}
if (region.get_dimension_count() == 0)
{
if (parameter.is_pointer())
{
clause_args.append_with_separator(
"*" + parameter_decl.get_name().prettyprint(),
",");
}
else if (parameter.is_reference())
{
clause_args.append_with_separator(
parameter_decl.get_name().prettyprint(),
",");
}
}
else
{
Source array_sections;
for (unsigned int j = 1;
j <= region.get_dimension_count();
j++)
{
Region::DimensionSpecifier &dim_spec(region[region.get_dimension_count() - j]);
DEBUG_CODE()
{
std::cerr << "Region: #" << j << std::endl
<< " dimension_start: " << dim_spec.get_dimension_start() << std::endl
<< " accessed_length: " << dim_spec.get_accessed_length() << std::endl
<< " dimension_length: " << dim_spec.get_dimension_length() << std::endl;
}
Source lower_bound_src, upper_bound_src;
array_sections
<< "["
<< lower_bound_src
<< ";"
<< upper_bound_src
<< "]"
;
lower_bound_src << dim_spec.get_dimension_start()
;
upper_bound_src << dim_spec.get_accessed_length();
AST_t upper_bound_tree = upper_bound_src.parse_expression(context_tree, 
sl);
}
if (parameter.is_pointer()
)
{
Source shape_dims;
for (unsigned int j = 1;
j <= region.get_dimension_count();
j++)
{
Region::DimensionSpecifier& dim_spec(region[region.get_dimension_count() - j]);
shape_dims
<< "[" << dim_spec.get_dimension_length() << "]"
;
}
Source shape_src;
shape_src
<< "(" << shape_dims << " "
<< parameter_decl.get_name().prettyprint() 
<< ")"
;
clause_args.append_with_separator(
shape_src.get_source() + array_sections.get_source(),
",");
}
else
{
clause_args.append_with_separator(
parameter_decl.get_name().prettyprint() + array_sections.get_source(),
",");
}
}
DEBUG_CODE()
{
std::cerr << std::endl;
}
}
void SS2OpenMP::on_post_task(PragmaCustomConstruct construct)
{
AugmentedSymbol augmented_sym = AugmentedSymbol::invalid();
DeclaredEntity decl_entity(NULL, construct.get_scope_link());
AST_t context_tree = construct.get_ast();
if (construct.is_function_definition())
{
FunctionDefinition task_definition(construct.get_declaration(),
construct.get_scope_link());
decl_entity = task_definition.get_declared_entity();
augmented_sym = AugmentedSymbol(task_definition.get_function_symbol());
context_tree = task_definition.get_function_body().get_ast();
}
else
{
Declaration decl(construct.get_declaration(), construct.get_scope_link());
decl_entity = decl.get_declared_entities()[0];
augmented_sym = AugmentedSymbol(decl_entity.get_declared_symbol());
context_tree = decl_entity.get_parameter_declarations()[0].get_ast();
}
if (!augmented_sym.is_task())
{
internal_error("This is not a task!", 0);
}
ObjectList<ParameterDeclaration> parameter_decls = decl_entity.get_parameter_declarations();
RefPtr<ParameterRegionList> parameter_region_list = augmented_sym.get_parameter_region_list();
Source new_pragma_construct_src, clauses, device_line;
new_pragma_construct_src
<< "#line " << construct.get_ast().get_line() << " \"" << construct.get_ast().get_file() << "\"\n"
<< device_line
<< "#pragma omp task " << clauses << "\n"
<< ";"
;
Source input_clause_args;
Source output_clause_args;
Source inout_clause_args;
Source reduction_clause_args;
ObjectList<Type> parameters = augmented_sym.get_type().nonadjusted_parameters();
int i = 0;
for (ObjectList<RegionList>::iterator it = parameter_region_list->begin();
it != parameter_region_list->end();
it++)
{
RegionList &region_list(*it);
for (ObjectList<Region>::iterator itt = region_list.begin();
itt != region_list.end();
itt++)
{
Region &region(*itt);
bool is_reduction = region.get_reduction() == Region::REDUCTION;
Source *clause_args = NULL;
int dir = (int)region.get_direction();
switch ((int)region.get_direction())
{
case Region::INPUT_DIR:
{
add_to_clause(region, parameters[i], parameter_decls[i], input_clause_args, "input", context_tree,
construct.get_scope_link());
break;
}
case Region::OUTPUT_DIR:
{
add_to_clause(region, parameters[i], parameter_decls[i], output_clause_args, "output", context_tree,
construct.get_scope_link());
break;
}
case Region::INOUT_DIR:
{
if (!is_reduction)
{
add_to_clause(region, parameters[i], parameter_decls[i], inout_clause_args, "inout", context_tree,
construct.get_scope_link());
}
else
{
add_to_clause(region, parameters[i], parameter_decls[i], reduction_clause_args, "reduction", context_tree,
construct.get_scope_link());
}
break;
}
default:
{
internal_error("Invalid directionality", 0);
break;
}
}
}
i++;
}
clauses << " untied";
if (augmented_sym.has_high_priority())
{
clauses << " __priority(99)";
}
if (!input_clause_args.empty())
{
clauses << " input(" << input_clause_args << ")";
}
if (!output_clause_args.empty())
{
clauses << " output(" << output_clause_args << ")";
}
if (!inout_clause_args.empty())
{
clauses << " inout(" << inout_clause_args << ")";
}
if (!reduction_clause_args.empty())
{
clauses << " __shared_reduction(" << reduction_clause_args << ")";
}
Source implements;
PragmaCustomClause device_clause = construct.get_clause("device");
if (device_clause.is_defined())
{
Source device_list;
device_line
<< "#pragma omp target device(" << device_list << ") copy_deps" << implements << "\n"
;
ObjectList<std::string> arg_list = device_clause.get_arguments(ExpressionTokenizerTrim());
for (ObjectList<std::string>::iterator it = arg_list.begin();
it != arg_list.end();
it++)
{
if (*it == "ppu")
{
device_list.append_with_separator("smp", ",");
}
else 
device_list.append_with_separator(*it, ",");
}
}
PragmaCustomClause implements_clause = construct.get_clause("implements");
if (implements_clause.is_defined())
{
implements << " implements(" << concat_strings(implements_clause.get_arguments(), ",") << ")";
}
AST_t pragma_decl = construct.get_declaration();
AST_t new_pragma_tree_list = new_pragma_construct_src.parse_declaration(construct.get_ast(),
construct.get_scope_link());
ASTIterator iterator_list = new_pragma_tree_list.get_list_iterator();
AST_t decl = iterator_list.item();
construct.get_ast().replace(decl);
while (is_pragma_custom("omp", decl, construct.get_scope_link()))
{
PragmaCustomConstruct pragma(decl, construct.get_scope_link());
decl = pragma.get_declaration();
}
decl.replace(pragma_decl);
}
void SS2OpenMP::on_post_wait(PragmaCustomConstruct construct)
{
Source taskwait_directive_src, on_clause_src;
taskwait_directive_src
<< "#line " << construct.get_ast().get_line() << " \"" << construct.get_ast().get_file() << "\"\n"
<< "#pragma omp taskwait " << on_clause_src << "\n"
;
if (construct.get_clause("on").is_defined())
{
Source on_clause_args;
on_clause_src << "on(" << on_clause_args << ")"; 
ObjectList<Expression> expr_list = construct.get_clause("on").get_expression_list();
for (ObjectList<Expression>::iterator it = expr_list.begin();
it != expr_list.end();
it++)
{
Expression in_clause_expr(*it);
if (in_clause_expr.get_operation_kind() == Expression::REFERENCE)
{
in_clause_expr = in_clause_expr.get_unary_operand();
}
on_clause_args.append_with_separator(
in_clause_expr.prettyprint(),
",");
}
}
AST_t taskwait_directive_tree = taskwait_directive_src.parse_statement(construct.get_ast(),
construct.get_scope_link());
construct.get_ast().replace(taskwait_directive_tree);
}
void SS2OpenMP::on_post_finish(PragmaCustomConstruct construct)
{
Source taskwait_directive_src;
taskwait_directive_src
<< "#line " << construct.get_ast().get_line() << " \"" << construct.get_ast().get_file() << "\"\n"
<< "#pragma omp taskwait\n"
;
AST_t taskwait_directive_tree = taskwait_directive_src.parse_statement(construct.get_ast(),
construct.get_scope_link());
construct.get_ast().replace(taskwait_directive_tree);
}
void SS2OpenMP::on_post_barrier(PragmaCustomConstruct construct)
{
Source taskwait_directive_src;
taskwait_directive_src
<< "#line " << construct.get_ast().get_line() << " \"" << construct.get_ast().get_file() << "\"\n"
<< "#pragma omp taskwait\n"
;
AST_t taskwait_directive_tree = taskwait_directive_src.parse_statement(construct.get_ast(),
construct.get_scope_link());
construct.get_ast().replace(taskwait_directive_tree);
}
void SS2OpenMP::remove_directive(PragmaCustomConstruct construct)
{
construct.get_ast().remove_in_list();
}
void SS2OpenMP::directive_not_implemented(PragmaCustomConstruct construct)
{
std::cerr << construct.get_ast().get_locus_str() << ": warning: directive not implemented yet, skipping" << std::endl;
construct.get_ast().remove_in_list();
}
void SS2OpenMP::construct_not_implemented(PragmaCustomConstruct construct)
{
std::cerr << construct.get_ast().get_locus_str() << ": warning: construct not implemented yet, skipping" << std::endl;
construct.get_ast().replace(construct.get_declaration());
}
void SS2OpenMP::run(DTO& dto)
{
PragmaCustomCompilerPhase::run(dto);
}
}
EXPORT_PHASE(TL::SS2OpenMP)
