package imop.lib.analysis;
import imop.ast.info.DataSharingAttribute;
import imop.ast.node.external.*;
import imop.ast.node.internal.*;
import imop.lib.analysis.flowanalysis.*;
import imop.lib.analysis.typesystem.ArrayType;
import imop.lib.analysis.typesystem.FunctionType;
import imop.lib.analysis.typesystem.Type;
import imop.lib.util.CellList;
import imop.lib.util.Misc;
import imop.lib.util.NodePair;
import imop.parser.Program;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
public class SVEChecker {
public static final Set<String> variableFunctions = new HashSet<>();
static {
variableFunctions.add("omp_get_thread_num");
variableFunctions.add("gettimeofday");
}
private static Set<Node> singleValuedExpressions = new HashSet<>();
private static Set<Node> multiValuedExpressions = new HashSet<>();
private static boolean disable = true;
public static long cpredaTimer = 0;
public static long sveQueryTimer = 0;
public static void resetStaticFields() {
SVEChecker.cpredaTimer = 0;
SVEChecker.sveQueryTimer = 0;
SVEChecker.singleValuedExpressions.clear();
SVEChecker.multiValuedExpressions.clear();
CoExistenceChecker.resetStaticFields();
}
public static boolean isSingleValuedPredicate(Expression exp) {
if (Program.sveNoCheck) {
return true;
}
long timer = System.nanoTime();
if (!Misc.isAPredicate(exp)) {
SVEChecker.sveQueryTimer += (System.nanoTime() - timer);
return false;
} else {
boolean retVal = SVEChecker.isSingleValuedPredicate(exp, new HashSet<>(), new HashSet<>());
SVEChecker.sveQueryTimer += (System.nanoTime() - timer);
return retVal;
}
}
static boolean isSingleValuedPredicate(Expression exp, Set<Expression> expSet, Set<NodePair> nodePairs) {
if (disable) {
Node parent = exp.getParent();
if (parent instanceof IfStatement || parent instanceof SwitchStatement) {
return true;
} else if (parent instanceof WhileStatement || parent instanceof DoStatement) {
} else if (parent instanceof NodeOptional) {
Node grandParent = parent.getParent();
if (grandParent instanceof ForStatement) {
ForStatement forStmt = (ForStatement) grandParent;
if (forStmt.getInfo().getCFGInfo().getTerminationExpression() == exp) {
parent = forStmt;
}
}
}
if (parent != null) {
if (exp.getInfo().getNodePhaseInfo().getPhaseSet().size() == 1) {
return false;
}
if (!Misc.getExactEnclosee(parent, BarrierDirective.class).isEmpty()) {
return true;
} else {
return false;
}
}
}
if (exp == null) {
return false;
}
if (expSet.contains(exp)) {
return true;
} else if (SVEChecker.singleValuedExpressions.contains(exp)) {
return true;
} else if (SVEChecker.multiValuedExpressions.contains(exp)) {
return false;
} else {
expSet.add(exp);
}
CellList readList = exp.getInfo().getReads();
if (readList.isEmpty()) {
expSet.remove(exp);
SVEChecker.singleValuedExpressions.add(exp);
return true;
}
CellList writeList = exp.getInfo().getWrites();
if (writeList.size() > 1) {
expSet.remove(exp);
SVEChecker.multiValuedExpressions.add(exp);
return false;
}
for (Cell r : readList) {
if (r == Cell.getNullCell()) {
continue;
} else if (r instanceof AddressCell) {
Cell pointedElem = ((AddressCell) r).getPointedElement();
if (exp.getInfo().getSharingAttribute(pointedElem) == DataSharingAttribute.SHARED) {
continue;
} else {
expSet.remove(exp);
SVEChecker.multiValuedExpressions.add(exp);
return false;
}
} else if (r instanceof FieldCell) {
expSet.remove(exp);
SVEChecker.multiValuedExpressions.add(exp);
return false;
} else if (r instanceof HeapCell) {
expSet.remove(exp);
SVEChecker.multiValuedExpressions.add(exp);
return false;
} else if (r instanceof FreeVariable) {
expSet.remove(exp);
SVEChecker.multiValuedExpressions.add(exp);
return false;
}
Symbol variable = (Symbol) r;
Type varType = variable.getType();
if (varType != null && varType instanceof FunctionType) {
continue;
}
if (exp.getInfo().getSharingAttribute(variable) == DataSharingAttribute.SHARED) {
if (variable.getType() instanceof ArrayType) {
continue;
}
if (CoExistenceChecker.isWrittenInPhase(exp, variable, expSet, nodePairs)) {
expSet.remove(exp);
SVEChecker.multiValuedExpressions.add(exp);
return false;
} else {
continue;
}
} else {
if (variable.getType() instanceof ArrayType) {
expSet.remove(exp);
SVEChecker.multiValuedExpressions.add(exp);
return false;
}
if (!ensureSameValue(variable, exp, expSet, nodePairs)) {
expSet.remove(exp);
SVEChecker.multiValuedExpressions.add(exp);
return false;
} else {
continue;
}
}
}
expSet.remove(exp);
SVEChecker.singleValuedExpressions.add(exp);
return true;
}
public static boolean ensureSameValue(Symbol sym, Node exp, Set<Expression> expSet, Set<NodePair> nodePairs) {
Node cfgNode = Misc.getCFGNodeFor(exp);
Set<Definition> reachingDefinitions = cfgNode.getInfo().getReachingDefinitions(sym);
if (reachingDefinitions == null || reachingDefinitions.isEmpty()) {
return false;
}
if (reachingDefinitions.size() == 1) {
Node rd = Misc.getAnyElement(reachingDefinitions).getDefiningNode();
if (SVEChecker.writesSingleValue(rd, expSet, nodePairs)) {
return true;
} else {
return false;
}
}
for (Definition rdDef : reachingDefinitions) {
Node rd = rdDef.getDefiningNode();
if (!SVEChecker.writesSingleValue(rd, expSet, nodePairs)) {
return false;
}
if (!CoExistenceChecker.existsForAll(rd, expSet, nodePairs)) {
return false;
}
}
return true;
}
public static boolean writesSingleValue(Node node) {
return SVEChecker.writesSingleValue(node, new HashSet<>(), new HashSet<>());
}
private static boolean writesSingleValue(Node node, Set<Expression> expSet, Set<NodePair> nodePairs) {
if (node instanceof Declaration) {
Declaration decl = (Declaration) node;
if (!decl.getInfo().hasInitializer()) {
return false;
}
return SVEChecker.isSingleValuedPredicate(decl.getInfo().getInitializer(), expSet, nodePairs);
} else if (node instanceof ParameterDeclaration) {
return false;
} else if (node instanceof OmpForInitExpression) {
return false;
} else if (node instanceof OmpForCondition) {
assert (false);
} else if (node instanceof OmpForReinitExpression) {
return false;
} else if (node instanceof ExpressionStatement) {
ExpressionStatement exp = (ExpressionStatement) node;
boolean ret = SVEChecker.isSingleValuedPredicate((Expression) exp.getF0().getNode(), expSet, nodePairs);
return ret;
} else if (node instanceof PostCallNode) {
PostCallNode post = (PostCallNode) node;
CallStatement callStmt = post.getParent();
if (!callStmt.getInfo().hasKnownCalledFunctionSymbol()) {
return false;
}
Symbol calledFunction = (Symbol) callStmt.getInfo().getCalledSymbols().get(0);
if (SVEChecker.variableFunctions.contains(calledFunction.getName())) {
return false;
} else {
List<SimplePrimaryExpression> argumentList = callStmt.getPreCallNode().getArgumentList();
if (argumentList.isEmpty()) {
return true;
}
for (SimplePrimaryExpression spe : argumentList) {
if (!SVEChecker.isSingleValuedPredicate(spe, expSet, nodePairs)) {
return false;
}
}
return true;
}
} else if (node instanceof PreCallNode) {
return false;
} else if (node instanceof Expression) {
boolean ret = SVEChecker.isSingleValuedPredicate((Expression) node, expSet, nodePairs);
return ret;
}
assert (false) : node.getClass().getSimpleName() + " " + node + " type of node is invalid here.";
return false;
}
public static void extractSVEPragmas() {
for (CompoundStatement compStmt : Misc.getInheritedPostOrderEnclosee(Program.getRoot(),
CompoundStatement.class)) {
List<Node> elements = compStmt.getInfo().getCFGInfo().getElementList();
for (Object elementObj : elements.stream().filter(n -> n instanceof UnknownPragma).toArray()) {
Node element = (Node) elementObj;
if (!element.toString().contains("imop predicate sve")) {
continue;
}
int indexAnnotated = elements.indexOf(element) + 1;
if (indexAnnotated >= elements.size()) {
Misc.exitDueToError("Incorrect sve annotation at line #" + Misc.getLineNum(element));
}
Node annotatedElem = elements.get(indexAnnotated);
if (annotatedElem instanceof IfStatement) {
((IfStatement) annotatedElem).getInfo().getCFGInfo().getPredicate().getInfo().setSVEAnnotated();
compStmt.getInfo().getCFGInfo().removeElement(element);
} else if (annotatedElem instanceof SwitchStatement) {
((SwitchStatement) annotatedElem).getInfo().getCFGInfo().getPredicate().getInfo().setSVEAnnotated();
compStmt.getInfo().getCFGInfo().removeElement(element);
} else if (annotatedElem instanceof WhileStatement) {
((WhileStatement) annotatedElem).getInfo().getCFGInfo().getPredicate().getInfo().setSVEAnnotated();
compStmt.getInfo().getCFGInfo().removeElement(element);
} else if (annotatedElem instanceof DoStatement) {
((DoStatement) annotatedElem).getInfo().getCFGInfo().getPredicate().getInfo().setSVEAnnotated();
compStmt.getInfo().getCFGInfo().removeElement(element);
} else {
Misc.warnDueToLackOfFeature("Ignoring the sve annotation at line #" + Misc.getLineNum(element),
null);
}
}
}
}
}
