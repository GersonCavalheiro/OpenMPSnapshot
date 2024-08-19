#ifndef __PREPROCESSOR_PRAGMA_HH__
#define __PREPROCESSOR_PRAGMA_HH__

#include <map>
#include <string>
#include <utility>

#include <ast/ast.hh>
#include <pragma-parser.hh>

#include "types.hh"
#include "util.hh"

namespace __preprocessor__ {
namespace __pragma__ {
const std::string pragma_name_identifier="pragma_omp_dag_marker_";
std::string pragma_name(size_t id);
bool is_pragma_name(const std::string& name);
size_t pragma_id(const std::string& name);
ptok_t pragma_type(const pragma_ast_node_t& pragma);
std::string pragma_thread_num(const pragma_ast_t& ast,pragma_ast_node_t* root=nullptr);
std::vector<pragma_ast_node_t*> environment_pragmas(const std::vector<pragma_ast_node_t*>& pragmas);
std::pair<std::map<pragma_ast_node_t*,size_t>,std::vector<pragma_ast_node_t*>>
transform_pragmas_identifiers(std::vector<std::string>& src,const pragma_ast_t& ast);
std::pair<ptok_t,std::vector<std::string>> coarsening_options(const pragma_ast_t& ast,pragma_ast_node_t* root=nullptr);
std::vector<dependency_t> dependency_list(const pragma_ast_t& ast,pragma_ast_node_t* root=nullptr);
}
}
#endif
