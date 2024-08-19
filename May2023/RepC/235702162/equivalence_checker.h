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
bool detailed;
std::map<std::pair<std::string,std::pair<std::string,int> >, parallel_loop> reference;
std::map<std::pair<std::string,std::pair<std::string,int> >, parallel_loop> tool;
std::map<std::string, generic_obj> reference_objs;
std::map<std::string, generic_obj> tool_objs;
std::map<std::string, std::string> equivalent_classes;
int unique_obj_id;
std::string flag;
std::string replace_all(std::string str, std::string from, std::string to);
void cleanStr(std::string & desc);
void extractReductionOperator(std::string & clause, std::string & op);
void extractSTRINGObject(std::string & desc, std::string & str);
void createVectorToARRAYObj(std::string & desc, std::vector<std::string> & obj, bool recover_op,  std::vector<std::string> & ops);
void extractARRAYObject(std::string & desc, std::string & str);
void extractCLASSObject(std::string & desc, std::string & str);
void extractObject(std::string & desc, std::string & name, std::string & obj);
void readEquivalenceClasses(std::string filename);
void readFile(std::string filename, bool fileType);
void insertField(parallel_loop *data, std::string name, std::string obj);
bool representSameDirective(std::string & pragma_ref, std::string & pragma_tool);
bool areObjectsEquivalent(std::string & objRef, std::string & objTool);
bool areObjectsEquivalent(std::vector<std::string> & vectRef, std::vector<std::string> & vectTool);
void normalizeCompoundAssignmentOperator(std::string op, std::string & desc) ;
bool containsCompoundAssignmentOperator(std::string & desc);
bool normalizeExpression(std::string & desc);
void denotateAtomicEquivalenttoReduction(std::vector<std::string> & dependence_list, std::vector<std::string> & reduction_prag, bool fileType);
void processGenericObject(std::string & objName, std::string & desc, bool fileType);
void processParallelLoop(std::string & desc, bool fileType);
void appendVector(std::vector<std::string> & dest, std::vector<std::string> & src);
bool equivalentVectors(std::vector<std::string> & vectRef, std::vector<std::string> & vectTool);
bool equivalentReductions(std::vector<std::string> & vectRef, std::vector<std::string> & vectRef_ops, std::vector<std::string> & vectTool, std::vector<std::string> & vectTool_ops);
std::string getEquivalentAtTargetContext(std::string prag, bool offload);
bool isEquivalent(std::string prag1, std::string prag2);
void classify(parallel_loop *refData, parallel_loop *toolData); 
void classify();
void printVectDiffs(std::vector<std::string> refData, std::vector<std::string> toolData);
void printVectDiffs(std::vector<std::string> refData, std::vector<std::string> refData_ops, std::vector<std::string> toolData, std::vector<std::string> toolData_ops);
void debug_classification(parallel_loop *refData, parallel_loop *toolData);
void debug_classification();
std::string writeGenericObj(generic_obj *data, std::vector<std::string> & depList);
std::string writeArrayObj(std::vector<std::string> vct, std::vector<std::string> vct_op);
std::string writeArrayObj(std::vector<std::string> vct);
std::string writeLoop(parallel_loop *data, bool reference, parallel_loop *data_ref);
std::string joinJSONS(parallel_loop *refData, parallel_loop *toolData, bool existsToolVersion);
void joinJSONS(std::string outputJSON);
public:
JSONParser(std::string fileRef, std::string fileTool, std::string outputJSON, std::string flagUsed) { 
flag = flagUsed;
reference.erase(reference.begin(), reference.end());
tool.erase(tool.begin(), tool.end());
readEquivalenceClasses("equivalent_pragmas.json");
readFile(fileRef, true);
readFile(fileTool, false);
detailed = false;
if (flag == "-check-detailed") {
detailed = true;
classify();
}
if (flag == "-check")
classify();
if (flag == "-join")
joinJSONS(outputJSON);
if (flag == "-report-check")
debug_classification();
}
JSONParser() { }
};
