#ifndef __PREPROCESSOR_TYPES_HH__
#define __PREPROCESSOR_TYPES_HH__

#include <pragma-parser.hh>
#include "../clang/clang-wrapper.hh"
#include "../util.hh"
#include "util.hh"

namespace __preprocessor__ {
enum variable_type_E {
REGULAR_VAR,
REFERENCE_VAR,
FUNCTION_VAR,
FOR_VAR,
LAMBDA_FUNCTION,
LAMBDA_VAR,
NOT_VAR
};
typedef variable_type_E variable_t;

struct for_statement_t {
clang_ast_node_t* self=nullptr;
clang_ast_node_t* declaration=nullptr;
clang_ast_node_t* condition=nullptr;
clang_ast_node_t* step=nullptr;
clang_ast_node_t* block=nullptr;
for_statement_t(clang_ast_node_t* self=nullptr,clang_ast_node_t* declaration=nullptr,clang_ast_node_t* condition=nullptr,clang_ast_node_t* step=nullptr,clang_ast_node_t* block=nullptr);
for_statement_t(const clang_ast_node_t& node);
for_statement_t(const clang_ast_node_t& node,const std::vector<std::string>& src);
void init(const std::vector<std::string>& src);
std::string begining(const std::vector<std::string>& src);
};

struct dependency_t {
std::string out="";
std::string in="";
std::string condition="";
ptok_t type=ptok_t::DEPEND;
dependency_t(std::string out="",std::string in="",std::string condition="",ptok_t type=ptok_t::DEPEND);
};

std::ostream& operator<<(std::ostream& ost,const for_statement_t& statement);
std::ostream& operator<<(std::ostream& ost,const dependency_t& dependency);
std::ostream& operator<<(std::ostream& ost,variable_t type);
}
#endif
