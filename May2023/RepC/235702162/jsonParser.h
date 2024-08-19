#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <vector>
#include <map>
#include <algorithm>
using namespace std;
class JSONParser {
private:
typedef struct parallel_loop {
std::string filename; 
int loop_id; 
int loop_line; 
int loop_column; 
bool ordered; 
std::string loop_function; 
bool offload; 
bool multiversioned; 
std::string induction_variable; 
std::string pragma_type; 
std::vector<std::string> shared; 
std::vector<std::string> private_prag; 
std::vector<std::string> first_private_prag; 
std::vector<std::string> last_private_prag; 
std::vector<std::string> linear_prag; 
std::vector<std::string> reduction_prag_op; 
std::vector<std::string> reduction_prag; 
std::vector<std::string> map_to; 
std::vector<std::string> map_from; 
std::vector<std::string> map_tofrom; 
std::vector<std::string> dependence_list; 
} parallel_loop;
typedef struct generic_obj {
std::string filename; 
std::string function_name; 
int loop_id; 
int instruction_id; 
std::string pragma_type; 
std::vector<std::string> code_snippet; 
} generic_obj; 
std::map<std::pair<std::string,std::pair<std::string,int> >, parallel_loop> json_loops;
std::map<std::string, generic_obj> json_objs;
std::string replace_all(std::string str, std::string from, std::string to);
void cleanStr(std::string & desc);
void extractReductionOperator(std::string & clause, std::string & op);
void extractSTRINGObject(std::string & desc, std::string & str);
void createVectorToARRAYObj(std::string & desc, std::vector<std::string> & obj, bool recover_op,  std::vector<std::string> & ops);
void extractARRAYObject(std::string & desc, std::string & str);
void extractCLASSObject(std::string & desc, std::string & str);
void extractObject(std::string & desc, std::string & name, std::string & obj);
void readFile(std::string filename);
void insertField(parallel_loop *data, std::string name, std::string obj);
void processGenericObject(std::string & objName, std::string & desc);
void processParallelLoop(std::string & desc);
std::string generateOMPLoop(parallel_loop loop);
std::string generateOMPDirective(generic_obj obj);
std::string getListofClauses(std::vector<std::string> & clauses);
std::string getListofClauses(std::vector<std::string> & ops, std::vector<std::string> & clauses);
public:
std::map<std::string, std::string> directives;
JSONParser(std::string fileRef) { 
json_loops.erase(json_loops.begin(), json_loops.end());
readFile(fileRef);
}
JSONParser() { }
};
