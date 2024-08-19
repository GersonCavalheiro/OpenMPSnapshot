package imop;
import com.microsoft.z3.*;
import imop.ast.annotation.Label;
import imop.ast.info.cfgNodeInfo.ForConstructInfo;
import imop.ast.node.external.*;
import imop.ast.node.internal.*;
import imop.lib.analysis.Assignment;
import imop.lib.analysis.CoExistenceChecker;
import imop.lib.analysis.SVEChecker;
import imop.lib.analysis.flowanalysis.Cell;
import imop.lib.analysis.flowanalysis.Definition;
import imop.lib.analysis.flowanalysis.HeapCell;
import imop.lib.analysis.flowanalysis.Symbol;
import imop.lib.analysis.flowanalysis.generic.FlowAnalysis;
import imop.lib.analysis.mhp.AbstractPhase;
import imop.lib.analysis.mhp.DependenceCounter;
import imop.lib.analysis.mhp.incMHP.BeginPhasePoint;
import imop.lib.analysis.mhp.incMHP.Phase;
import imop.lib.analysis.mhp.yuan.YPhase;
import imop.lib.analysis.mhp.yuan.YuanConcurrencyAnalysis;
import imop.lib.analysis.mhp.yuan.YuanConcurrencyAnalysis.YuanStaticPhase;
import imop.lib.analysis.solver.ConstraintsGenerator;
import imop.lib.analysis.solver.FieldSensitivity;
import imop.lib.analysis.solver.SeedConstraint;
import imop.lib.analysis.solver.SyntacticAccessExpression;
import imop.lib.analysis.solver.tokens.ExpressionTokenizer;
import imop.lib.analysis.solver.tokens.IdOrConstToken;
import imop.lib.analysis.solver.tokens.OperatorToken;
import imop.lib.analysis.solver.tokens.Tokenizable;
import imop.lib.analysis.typesystem.ArrayType;
import imop.lib.analysis.typesystem.CharType;
import imop.lib.analysis.typesystem.SignedCharType;
import imop.lib.analysis.typesystem.SignedIntType;
import imop.lib.analysis.typesystem.SignedLongIntType;
import imop.lib.analysis.typesystem.SignedShortIntType;
import imop.lib.analysis.typesystem.Type;
import imop.lib.analysis.typesystem.UnsignedCharType;
import imop.lib.analysis.typesystem.UnsignedIntType;
import imop.lib.analysis.typesystem.UnsignedLongIntType;
import imop.lib.analysis.typesystem.UnsignedShortIntType;
import imop.lib.cfg.info.CFGInfo;
import imop.lib.cg.CallStack;
import imop.lib.cg.NodeWithStack;
import imop.lib.getter.LoopAndOmpProfiler;
import imop.lib.transform.BarrierCounterInstrumentation;
import imop.lib.transform.BasicTransform;
import imop.lib.transform.CopyEliminator;
import imop.lib.transform.percolate.DriverModule;
import imop.lib.transform.percolate.FencePercolationVerifier;
import imop.lib.transform.percolate.LoopInstructionsRescheduler;
import imop.lib.transform.percolate.UpwardPercolater;
import imop.lib.transform.simplify.DeclarationEscalator;
import imop.lib.transform.simplify.FunctionInliner;
import imop.lib.transform.simplify.ParallelConstructExpander;
import imop.lib.transform.simplify.RedundantSynchronizationRemoval;
import imop.lib.transform.updater.*;
import imop.lib.util.*;
import imop.parser.FrontEnd;
import imop.parser.Program;
import imop.parser.Program.ConcurrencyAlgorithm;
import java.io.FileNotFoundException;
import java.util.*;
@SuppressWarnings("unused")
public class Main {
public static long totalTime;
public static String globalString = "";
public static boolean timerOn = false;
public static void main(String[] args) throws FileNotFoundException, InterruptedException {
FrontEnd.getK2Builtins();
FrontEnd.getMacBuiltins();
FrontEnd.getOtherBuiltins();
totalTime = System.nanoTime();
Program.parseNormalizeInput(args);
DriverModule.clientAutoUpdateIDFA();
System.exit(0);
}
public static void countPhases() {
assert (Program.concurrencyAlgorithm == ConcurrencyAlgorithm.YCON)
: "Run this method only with Yuan's analysis. Single-barrier paired static phases won't make sense otherwise.";
Set<YPhase> aggPhaseSet = new HashSet<>();
for (ParallelConstruct parCons : Misc.getExactEnclosee(Program.getRoot(), ParallelConstruct.class)) {
for (AbstractPhase<?, ?> ph : parCons.getInfo().getConnectedPhases()) {
aggPhaseSet.add((YPhase) ph);
}
}
int numStaticPhases = 0;
for (ParallelConstruct parCons : Misc.getExactEnclosee(Program.getRoot(), ParallelConstruct.class)) {
numStaticPhases += YuanConcurrencyAnalysis.getStaticPhases(parCons).size();
}
System.out.println(aggPhaseSet.size() + ": " + numStaticPhases);
System.exit(0);
}
public static void checkHighLevelCFGTransformations() {
testInsertImmediatePredecessor();
DumpSnapshot.dumpRoot("test");
System.exit(0);
}
private static void checkSVEAnnotations() {
FunctionDefinition main = Program.getRoot().getInfo().getMainFunction();
if (main == null) {
return;
}
int count = 0;
for (Node leaf : main.getInfo().getCFGInfo().getIntraTaskCFGLeafContents()) {
if (!(leaf instanceof Expression)) {
continue;
}
Expression exp = (Expression) leaf;
if (!Misc.isAPredicate(exp)) {
continue;
}
count++;
boolean isSVE = SVEChecker.isSingleValuedPredicate(exp);
System.out.println(leaf + " is " + (isSVE ? "" : "not") + " a single-valued expression.");
}
System.out.println("Total number of predicates: " + count);
System.out.println("Time spent in SVE queries: " + SVEChecker.cpredaTimer / (1e9 * 1.0) + "s.");
System.exit(0);
}
private static void criticalAnalyzer() {
for (ParallelConstruct parCons : Misc.getInheritedEnclosee(Program.getRoot(), ParallelConstruct.class)) {
for (AbstractPhase<?, ?> absPh : parCons.getInfo().getConnectedPhases()) {
Phase ph = (Phase) absPh;
CellSet allCells = new CellSet();
Set<CriticalNode> nodes = new HashSet<>();
for (Node node : ph.getNodeSet()) {
if (node instanceof BeginNode) {
BeginNode begin = (BeginNode) node;
if (begin.getParent() instanceof CriticalConstruct) {
CriticalConstruct cc = (CriticalConstruct) begin.getParent();
CellSet cellSet = cc.getInfo().getSharedAccesses();
allCells.addAll(cellSet);
nodes.add(new CriticalNode(cc, cellSet));
}
}
}
for (CriticalNode cn1 : nodes) {
for (CriticalNode cn2 : nodes) {
if (cn1.cc == cn2.cc) {
continue;
}
if (cn1.getAccessedCells().containsAll(cn2.getAccessedCells())) {
System.out.println(cn1.getAccessedCells() + " with " + cn2.getAccessedCells());
CriticalNode.connect(cn1, cn2);
}
}
}
}
}
System.exit(0);
}
private static void debugDeclarationEscalator(WhileStatement whileStmt) {
DeclarationEscalator
.pushAllDeclarationsUpFromLevel((CompoundStatement) whileStmt.getInfo().getCFGInfo().getBody());
DumpSnapshot.dumpVisibleSharedReadWrittenCells("");
System.exit(0);
}
private static void profileStructure1() {
LoopAndOmpProfiler profiler = new LoopAndOmpProfiler();
Program.getRoot().accept(profiler);
DumpSnapshot.printToFile(profiler.str.toString(), Program.fileName + "_sample.txt");
System.exit(0);
}
private static void profileStructure2() {
for (IterationStatement itStmt : Misc.getInheritedEnclosee(Program.getRoot(), IterationStatement.class)) {
if (itStmt.getClass() == IterationStatement.class) {
continue;
}
itStmt = (IterationStatement) Misc.getCFGNodeFor(itStmt);
int externalBarriersInLoops = 0;
int internalBarriersInLoops = 0;
boolean foundExternalRegion = false;
for (NodeWithStack nws : itStmt.getInfo().getCFGInfo()
.getIntraTaskCFGLeafContentsOfSameParLevel(new CallStack())) {
Node internalNode = nws.getNode();
if (internalNode instanceof BarrierDirective) {
externalBarriersInLoops++;
foundExternalRegion = true;
}
if (!foundExternalRegion) {
if (internalNode instanceof BeginNode) {
BeginNode beginNode = (BeginNode) internalNode;
if (beginNode.getParent() instanceof ParallelConstruct) {
internalBarriersInLoops += 2;
}
}
}
}
if (foundExternalRegion) {
if (externalBarriersInLoops > 1) {
System.out.println(
"Loop#" + Misc.getLineNum(itStmt) + ": " + externalBarriersInLoops + "B of outer regions.");
}
} else {
if (internalBarriersInLoops > 1) {
System.out.println(
"Loop#" + Misc.getLineNum(itStmt) + ": " + internalBarriersInLoops + "B of inner regions.");
}
}
}
}
private static void removeBarriers() {
RedundantSynchronizationRemoval.removeBarriers(Program.getRoot());
System.err.println("Time spent in SVE queries: " + SVEChecker.cpredaTimer / (1e9 * 1.0) + "s.");
System.err.println("Time spent in forward IDFA updates -- ");
for (FlowAnalysis<?> analysis : FlowAnalysis.getAllAnalyses().values()) {
System.err.println(
"\t For " + analysis.getAnalysisName() + ": " + analysis.flowAnalysisUpdateTimer / (1e9) + "s.");
}
System.err.println("Time spent in having uni-task precision in IDFA edge creation: "
+ CFGInfo.uniPrecisionTimer / (1e9 * 1.0) + "s.");
System.err.println("Time spent in field-sensitive queries: " + FieldSensitivity.timer / (1e9 * 1.0) + "s.");
System.err.println("Total number of times nodes were processed for automated updates -- ");
for (FlowAnalysis<?> analysis : FlowAnalysis.getAllAnalyses().values()) {
System.err.println("\t For " + analysis.getAnalysisName() + ": " + analysis.nodesProcessedDuringUpdate);
}
System.err.println("Number of field-sensitive queries: " + FieldSensitivity.counter);
BasicTransform.removeEmptyConstructs(Program.getRoot());
Program.getRoot().getInfo().removeUnusedElements();
RedundantSynchronizationRemoval.removeBarriers(Program.getRoot());
DumpSnapshot.printToFile(Program.getRoot(), "imop_output.i");
DumpSnapshot.dumpPhases("final");
DumpSnapshot.dumpVisibleSharedReadWrittenCells("final");
DumpSnapshot.dumpCopyInfo("final");
DumpSnapshot.dumpPhaseAndCopyInfo("final");
System.exit(0);
}
public static boolean tempChecker() {
for (ParallelConstruct parCons : Misc.getInheritedEnclosee(Program.getRoot(), ParallelConstruct.class)) {
for (AbstractPhase<?, ?> absPh : parCons.getInfo().getConnectedPhases()) {
Phase ph = (Phase) absPh;
Set<Node> stmtList = ph.getNodeSet();
Node stmt1 = null;
Node stmt2 = null;
for (Node s : stmtList) {
if (s instanceof DummyFlushDirective) {
continue;
}
if (s.toString().contains("/ 4.0")) {
stmt1 = s;
} else if (s.toString().contains("fabs")) {
stmt2 = s;
}
}
if (stmt1 != null && stmt2 != null) {
System.out.println(
"Phase id #" + ph.getPhaseId() + " has these statements: " + stmt1 + " and " + stmt2);
return true;
}
}
}
return false;
}
private static void temporaryChecker1() {
Statement s1 = null, s2 = null;
for (Statement stmt : Misc.getInheritedEnclosee(Program.getRoot(), Statement.class)) {
Node cfgNode = Misc.getCFGNodeFor(stmt);
if (cfgNode == null || !(cfgNode instanceof Statement)) {
continue;
}
Statement cfgStmt = (Statement) cfgNode;
if (!cfgStmt.getInfo().hasLabelAnnotations()) {
continue;
}
List<Label> labels = cfgStmt.getInfo().getLabelAnnotations();
if (labels.stream().anyMatch(l -> l.toString().contains("tc11"))) {
s1 = cfgStmt;
} else if (labels.stream().anyMatch(l -> l.toString().contains("tc12"))) {
s2 = cfgStmt;
}
}
if (s1 == null || s2 == null) {
Misc.exitDueToError("Could not find labels tc11 and tc12 to run this check.");
}
System.out.println(FieldSensitivity.canConflictWithTwoThreads(s1, s2));
SyntacticAccessExpression sae1 = s1.getInfo().getBaseAccessExpressionWrites().get(0);
SyntacticAccessExpression sae2 = s2.getInfo().getBaseAccessExpressionReads().get(1);
System.out.println(sae1);
System.out.println(sae2);
SeedConstraint tempEquality = new SeedConstraint(sae1.getIndexExpression(), s1,
IdOrConstToken.getNewIdToken("tid1"), sae2.getIndexExpression(), s2,
IdOrConstToken.getNewIdToken("tid2"), OperatorToken.ASSIGN);
Set<SeedConstraint> allEqualityConstraints = new HashSet<>();
allEqualityConstraints.add(tempEquality);
long timer = System.nanoTime();
ConstraintsGenerator.reinitInductionSet();
boolean satisfiable = ConstraintsGenerator.mayBeSatisfiable(allEqualityConstraints);
if (satisfiable) {
System.out.println("System has a solution.");
} else {
System.out.println("System is UNSATISFIABLE.");
}
System.exit(0);
}
private static void temporaryChecker2() {
Statement s1 = null;
for (Statement stmt : Misc.getInheritedEnclosee(Program.getRoot(), Statement.class)) {
Node cfgNode = Misc.getCFGNodeFor(stmt);
if (cfgNode == null || !(cfgNode instanceof Statement)) {
continue;
}
Statement cfgStmt = (Statement) cfgNode;
if (!cfgStmt.getInfo().hasLabelAnnotations()) {
continue;
}
List<Label> labels = cfgStmt.getInfo().getLabelAnnotations();
if (labels.stream().anyMatch(l -> l.toString().contains("tc11"))) {
s1 = cfgStmt;
}
}
if (s1 == null) {
Misc.exitDueToError("Could not find label  tc11 to run this check.");
}
HashMap<String, String> cfg = new HashMap<>();
cfg.put("model_validate", "true");
Context c = new Context(cfg);
Solver solver = c.mkSimpleSolver();
System.out.println(ExpressionTokenizer.getNormalizedForm(ExpressionTokenizer.getAssigningExpression(s1)));
c.close();
System.exit(0);
}
private static void temporaryChecker3() {
FunctionDefinition main = Program.getRoot().getInfo().getMainFunction();
if (main == null) {
System.exit(0);
}
for (Node leaf1 : main.getInfo().getCFGInfo().getIntraTaskCFGLeafContents()) {
if (!Misc.isAPredicate(leaf1)) {
continue;
}
System.err.println("##### Attempting eqaulity comparison of " + leaf1 + " with 0 #####");
List<Tokenizable> e1 = ExpressionTokenizer.getAssigningExpression(leaf1);
List<Tokenizable> e2 = new ArrayList<>();
e2.add(IdOrConstToken.getNewConstantToken("13"));
SeedConstraint tempEquality = new SeedConstraint(e1, leaf1, IdOrConstToken.getNewIdToken("tid1"), e2, leaf1,
IdOrConstToken.getNewIdToken("tid2"), OperatorToken.ASSIGN);
Set<SeedConstraint> allEqualityConstraints = new HashSet<>();
allEqualityConstraints.add(tempEquality);
long timer = System.nanoTime();
ConstraintsGenerator.reinitInductionSet();
boolean satisfiable = ConstraintsGenerator.mayBeSatisfiable(allEqualityConstraints);
if (satisfiable) {
System.err.println("System has a solution.");
}
}
DumpSnapshot.printToFile(ConstraintsGenerator.allConstraintString, Program.fileName + "_z3_queries.txt");
}
public static void testBarrierCounterInstrumentation() {
BarrierCounterInstrumentation.insertBarrierCounters();
DumpSnapshot.dumpRoot("counted");
System.exit(0);
}
private static void testBarrierPercolation() {
RedundantSynchronizationRemoval.removeBarriers(Program.getRoot());
System.err.println(
"Skipping par-cons expansion and merge...\nMoving on to upward percolation and barrier removal.");
for (ParallelConstruct parCons : Misc.getInheritedEnclosee(Program.getRoot(), ParallelConstruct.class)) {
UpwardPercolater.initPercolate(parCons);
}
DumpSnapshot.dumpPhases("trial");
RedundantSynchronizationRemoval.removeBarriers(Program.getRoot());
DumpSnapshot.dumpRoot("optimized");
System.err.println("Time spent in SVE queries: " + SVEChecker.cpredaTimer / (1e9 * 1.0) + "s.");
DumpSnapshot.dumpPhaseAndCopyInfo("final");
System.exit(0);
}
public static void testBaseExpressionGetter() {
for (ExpressionStatement expStmt : Misc.getInheritedEnclosee(Program.getRoot(), ExpressionStatement.class)) {
if (expStmt.getF0().present()) {
Expression exp = (Expression) expStmt.getF0().getNode();
List<SyntacticAccessExpression> accessReads = exp.getInfo().getBaseAccessExpressionReads();
List<SyntacticAccessExpression> accessWrites = exp.getInfo().getBaseAccessExpressionWrites();
if (!accessReads.isEmpty() || !accessWrites.isEmpty()) {
System.err.println("For expression " + exp);
}
for (SyntacticAccessExpression ae : accessReads) {
System.err.println("\tREAD:" + ae + ".");
}
for (SyntacticAccessExpression ae : accessWrites) {
System.err.println("\tWRITE:" + ae + ".");
}
}
}
System.exit(0);
}
public static void testClauses() {
OmpClause newClause;
newClause = FrontEnd.parseAndNormalize("final(x)", FinalClause.class);
System.out.println(newClause);
newClause = FrontEnd.parseAndNormalize("if (x < 2)", IfClause.class);
System.out.println(newClause);
newClause = FrontEnd.parseAndNormalize("mergeable", MergeableClause.class);
System.out.println(newClause);
newClause = FrontEnd.parseAndNormalize("nowait", NowaitClause.class);
System.out.println(newClause);
newClause = FrontEnd.parseAndNormalize("num_threads(2)", NumThreadsClause.class);
System.out.println(newClause);
newClause = FrontEnd.parseAndNormalize("copyin(x)", OmpCopyinClause.class);
System.out.println(newClause);
newClause = FrontEnd.parseAndNormalize("default(shared)", OmpDfltSharedClause.class);
System.out.println(newClause);
newClause = FrontEnd.parseAndNormalize("default(none)", OmpDfltNoneClause.class);
System.out.println(newClause);
newClause = FrontEnd.parseAndNormalize("firstprivate(x)", OmpFirstPrivateClause.class);
System.out.println(newClause);
newClause = FrontEnd.parseAndNormalize("lastprivate(x)", OmpLastPrivateClause.class);
System.out.println(newClause);
newClause = FrontEnd.parseAndNormalize("private(x, y)", OmpPrivateClause.class);
System.out.println(newClause);
newClause = FrontEnd.parseAndNormalize("reduction(+:x)", OmpReductionClause.class);
System.out.println(newClause);
newClause = FrontEnd.parseAndNormalize("shared(x)", OmpSharedClause.class);
System.out.println(newClause);
System.exit(0);
}
public static void testCoExistence() {
Node n1 = Program.getRoot().getInfo().getStatementWithLabel("l1");
Node n2 = Program.getRoot().getInfo().getStatementWithLabel("l2");
if (n1 == null || n2 == null) {
Misc.warnDueToLackOfFeature("Input should contain two statements labeled l1 and l2.", null);
} else {
System.out.println(CoExistenceChecker.canCoExistInAnyPhase(n1, n2));
}
System.exit(0);
}
private static void testCopyDetector() {
for (IterationStatement itStmt : Misc.getInheritedEnclosee(Program.getRoot(), IterationStatement.class)) {
if (itStmt.getClass() == IterationStatement.class) {
continue;
}
CopyEliminator.detectSwapCode(itStmt);
}
System.exit(0);
}
private static void testCopyElimination() {
DumpSnapshot.dumpReachingDefinitions("final");
long timer = System.nanoTime();
FunctionDefinition main = Program.getRoot().getInfo().getMainFunction();
int counter = 0;
while (true) {
DumpSnapshot.dumpPhaseAndCopyInfo(counter++ + "");
boolean internalChanged;
do {
internalChanged = false;
while (CopyEliminator.removeAllCopyKillers(main)) {
internalChanged = true;
}
while (CopyEliminator.eliminateDeadCopies(main)) {
internalChanged = true;
}
} while (internalChanged);
Set<Node> copySources = CopyEliminator.replaceCopiesIn(main);
if (copySources.isEmpty()) {
break;
}
}
DumpSnapshot.dumpCopyInfo("finalCopy");
timer = System.nanoTime() - timer;
BasicTransform.removeEmptyConstructs(main);
main.getInfo().removeUnusedElements();
RedundantSynchronizationRemoval.removeBarriers(main);
DumpSnapshot.dumpRoot("final");
System.out.println("Time spent in removing copies: " + timer / (1e9 * 1.0) + "s.");
System.err.println("Time spent in forward IDFA updates -- ");
for (FlowAnalysis<?> analysis : FlowAnalysis.getAllAnalyses().values()) {
System.err.println(
"\t For " + analysis.getAnalysisName() + ": " + analysis.flowAnalysisUpdateTimer / (1e9) + "s.");
}
System.exit(0);
}
private static void testCopyEliminationFromLoops() {
for (IterationStatement itStmt : Misc.getInheritedEnclosee(Program.getRoot(), IterationStatement.class)) {
if (itStmt instanceof WhileStatement) {
DriverModule.performCopyElimination(itStmt);
}
}
BasicTransform.removeEmptyConstructs(Program.getRoot());
Program.getRoot().getInfo().removeUnusedElements();
DumpSnapshot.dumpRoot("final");
DumpSnapshot.dumpPhaseAndCopyInfo("final");
System.exit(0);
}
private static void testDriverAndHeuristic() {
int counter = 0;
for (WhileStatement whileStmt : Misc.getInheritedEnclosee(Program.getRoot(), WhileStatement.class)) {
LoopInstructionsRescheduler.peelFirstBarrier(whileStmt);
DumpSnapshot.dumpPhases("cheker" + counter++);
}
System.exit(0);
}
public static void testExpressionTokenizer() {
for (BracketExpression brack : Misc.getInheritedEnclosee(Program.getRoot(), BracketExpression.class)) {
Expression exp = brack.getF1();
List<Tokenizable> tokens = ExpressionTokenizer.getInfixTokens(exp);
if (tokens.isEmpty()) {
continue;
}
System.err.println(exp + ": " + tokens + "\n Postfix: " + ExpressionTokenizer.getPostfixTokens(exp)
+ "\n Prefix: " + ExpressionTokenizer.getPrefixTokens(exp));
}
System.exit(0);
}
public static void testFencePercolation() {
for (ParallelConstruct parConsNode : Misc.getInheritedEnclosee(Program.getRoot(), ParallelConstruct.class)) {
for (AbstractPhase<?, ?> absPh : parConsNode.getInfo().getConnectedPhases()) {
Phase ph = (Phase) absPh;
System.out.println("Verifying fence percolation in phase #" + ph.getPhaseId());
boolean valid = FencePercolationVerifier.isPhaseSafeForFencePercolations(ph);
System.out.println(
"\t Percolation across fences " + (valid ? "is" : "is NOT") + " safe in this phase.\n");
}
}
System.exit(0);
}
private static void testFieldSensitivity() {
FunctionDefinition main = Program.getRoot().getInfo().getMainFunction();
for (Node leaf1 : main.getInfo().getCFGInfo().getIntraTaskCFGLeafContents()) {
for (Node leaf2 : main.getInfo().getCFGInfo().getIntraTaskCFGLeafContents()) {
if (leaf1 == leaf2) {
continue;
}
if (FieldSensitivity.canConflictWithTwoThreads(leaf1, leaf2)) {
System.out.println("Following pair may conflict: (" + leaf1 + ", " + leaf2 + ")");
}
}
}
System.exit(0);
}
private static void testIDFAUpdate() {
FunctionDefinition main = Program.getRoot().getInfo().getMainFunction();
Statement stmt = main.getInfo().getStatementWithLabel("l1");
NodeRemover.removeNode(stmt);
System.out.println(Program.getRoot());
System.exit(0);
}
private static void testInlining() {
FunctionDefinition mainFunc = Program.getRoot().getInfo().getMainFunction();
FunctionInliner.inline(mainFunc);
DumpSnapshot.dumpRoot("inlined");
System.err.println("Number of times IDFA update were triggered -- ");
for (FlowAnalysis<?> analysis : FlowAnalysis.getAllAnalyses().values()) {
System.err.println("\t For " + analysis.getAnalysisName() + ": " + analysis.autoUpdateTriggerCounter);
}
System.err.println("Total number of times nodes were processed during automated IDFA update -- ");
for (FlowAnalysis<?> analysis : FlowAnalysis.getAllAnalyses().values()) {
System.err.println("\t For " + analysis.getAnalysisName() + ": " + analysis.nodesProcessedDuringUpdate);
}
System.err.println("Time spent in forward IDFA updates -- ");
for (FlowAnalysis<?> analysis : FlowAnalysis.getAllAnalyses().values()) {
System.err.println(
"\t For " + analysis.getAnalysisName() + ": " + analysis.flowAnalysisUpdateTimer / (1e9) + "s.");
}
System.err.println("Time spent in SVE queries: " + SVEChecker.cpredaTimer / (1e9 * 1.0) + "s.");
System.err.println("Time spent in having uni-task precision in IDFA edge creation: "
+ CFGInfo.uniPrecisionTimer / (1e9 * 1.0) + "s.");
System.err.println("Number of field-sensitive queries: " + FieldSensitivity.counter);
System.err.println("Time spent in field-sensitive queries: " + FieldSensitivity.timer / (1e9 * 1.0) + "s.");
System.err.println("Time spent in inlining: " + FunctionInliner.inliningTimer / (1e9 * 1.0) + "s.");
System.exit(0);
}
public static void testInsertImmediatePredecessor() {
ExpressionStatement expStmt;
int i = 0;
for (FunctionDefinition function : Program.getRoot().getInfo().getAllFunctionDefinitions()) {
for (Node n : function.getInfo().getCFGInfo().getLexicalCFGContents()) {
if (n instanceof CompoundStatement) {
CompoundStatement parCons = (CompoundStatement) n;
for (Node comp : parCons.getInfo().getCFGInfo().getAllComponents()) {
Declaration decl = FrontEnd.parseAndNormalize("int x" + i++ + " = 333;", Declaration.class);
expStmt = FrontEnd.parseAndNormalize(i++ + ";", ExpressionStatement.class);
System.out.println(InsertImmediatePredecessor.insert(comp, decl));
System.out.println(InsertImmediatePredecessor.insert(comp, expStmt));
}
}
if (n instanceof ForStatement) {
ForStatement parCons = (ForStatement) n;
expStmt = FrontEnd.parseAndNormalize("x = " + i++ + ";", ExpressionStatement.class);
Statement stmt = FrontEnd.parseAndNormalize("continue;", Statement.class);
System.out.println(InsertImmediatePredecessor
.insert(parCons.getInfo().getCFGInfo().getTerminationExpression(), stmt));
if (parCons.getInfo().getCFGInfo().hasTerminationExpression()) {
System.out.println(InsertImmediatePredecessor
.insert(parCons.getInfo().getCFGInfo().getTerminationExpression(), expStmt));
}
}
}
}
}
public static void testInsertImmediateSuccessor() {
ExpressionStatement expStmt;
int i = 0;
for (FunctionDefinition function : Program.getRoot().getInfo().getAllFunctionDefinitions()) {
for (Node n : function.getInfo().getCFGInfo().getLexicalCFGContents()) {
if (n instanceof FunctionDefinition) {
FunctionDefinition func = (FunctionDefinition) n;
for (Node comp : func.getInfo().getCFGInfo().getAllComponents()) {
expStmt = FrontEnd.parseAndNormalize(i++ + ";", ExpressionStatement.class);
System.out.println(InsertImmediateSuccessor.insert(comp, expStmt));
}
}
}
}
}
public static void testInsertOnTheEdge() {
ExpressionStatement expStmt;
FunctionDefinition main = Program.getRoot().getInfo().getMainFunction();
if (main == null) {
System.out.println("Found no main. Exiting.");
System.exit(0);
}
for (Node n : main.getInfo().getCFGInfo().getLexicalCFGContents()) {
if (n instanceof CompoundStatement) {
CompoundStatement body = (CompoundStatement) n;
BeginNode b = body.getInfo().getCFGInfo().getNestedCFG().getBegin();
EndNode e = body.getInfo().getCFGInfo().getNestedCFG().getEnd();
int i = 0;
}
if (n instanceof Expression) {
int i = 10;
if (!Misc.isCFGLeafNode(n)) {
continue;
}
for (Node succ : new ArrayList<>(n.getInfo().getCFGInfo().getSuccessors())) {
expStmt = FrontEnd.parseAndNormalize((10 + i++) + ";", ExpressionStatement.class);
System.err.println(InsertOnTheEdge.insert(n, succ, expStmt));
}
}
}
}
public static void testLabels() {
FunctionDefinition mainFunc = Program.getRoot().getInfo().getMainFunction();
if (mainFunc == null) {
System.exit(0);
}
Statement stmt = mainFunc.getInfo().getStatementWithLabel("l1");
Statement newStmt = FrontEnd.parseAndNormalize("l2: X;", Statement.class);
NodeReplacer.replaceNodes(stmt, newStmt);
System.out.println(Program.getRoot());
System.exit(0);
}
public static void testLCACS() {
for (ExpressionStatement e1 : Misc.getInheritedEnclosee(Program.getRoot(), ExpressionStatement.class)) {
for (ExpressionStatement e2 : Misc.getInheritedEnclosee(Program.getRoot(), ExpressionStatement.class)) {
if (e1 == e2) {
continue;
}
System.out.println(e1 + " and " + e2 + " in " + Misc.getLCAScope(e1, e2));
}
}
}
private static void testLoopingSingle() {
FunctionDefinition main = Program.getRoot().getInfo().getMainFunction();
if (main == null) {
System.exit(0);
}
for (Node leafNode : main.getInfo().getCFGInfo().getIntraTaskCFGLeafContents()) {
if (leafNode instanceof BeginNode) {
BeginNode beginNode = (BeginNode) leafNode;
if (beginNode.getParent() instanceof SingleConstruct) {
SingleConstruct singleCons = (SingleConstruct) beginNode.getParent();
System.out.println("Single at line #" + Misc.getLineNum(singleCons) + " is "
+ (singleCons.getInfo().isLoopedInPhase() ? "within" : "not within") + " a loop.");
}
}
}
System.exit(0);
}
public static void testMapMerge() {
CellMap<String> firstMap = new CellMap<>();
CellMap<String> secondMap = new CellMap<>();
FunctionDefinition mainFunc = Program.getRoot().getInfo().getMainFunction();
CellSet allCells = new CellSet();
for (Cell c : mainFunc.getInfo().getAccesses()) {
allCells.add(c);
}
CellSet selected = new CellSet();
for (Cell c : allCells) {
if (c.toString().contains("y")) {
firstMap.put(c, c.toString());
}
if (c.toString().contains("x")) {
secondMap.put(c, "Meow");
}
if (c.toString().contains("z") || c.toString().contains("y")) {
selected.add(c);
}
}
System.out.println(firstMap);
System.out.println(secondMap);
System.out.println("===");
firstMap.mergeWith(secondMap, (s1, s2) -> {
String s3 = "";
if (s1 != null) {
s3 += s1;
}
s3 += s2;
return s3;
}, selected);
System.out.println(firstMap);
System.exit(0);
}
private static void testNewCellMap() {
try {
Thread.sleep(1000);
} catch (InterruptedException e) {
e.printStackTrace();
System.exit(0);
}
class MyString implements Immutable {
public String s1;
public MyString(String name) {
s1 = name;
}
@Override
public boolean equals(Object obj) {
if (this == obj) {
return true;
}
if (obj == null) {
return false;
}
if (getClass() != obj.getClass()) {
return false;
}
MyString other = (MyString) obj;
if (s1 == null) {
if (other.s1 != null) {
return false;
}
} else if (!s1.equals(other.s1)) {
return false;
}
return true;
}
@Override
public int hashCode() {
final int prime = 31;
int result = 1;
result = prime * result + ((s1 == null) ? 0 : s1.hashCode());
return result;
}
}
List<Symbol> symList = new ArrayList<>();
List<ExtensibleCellMap<MyString>> cellMapList = new ArrayList<>();
ExtensibleCellMap<MyString> testMap;
ExtensibleCellMap<MyString> newMap;
int COUNT = 1000;
for (int i = 0; i < COUNT; i++) {
symList.add(new Symbol("s" + i, SignedIntType.type(), null, null));
}
int counter = 0;
Random rand = new Random();
int index = -1;
long constTimer = System.nanoTime();
while (counter < COUNT) {
index++;
int size = Math.max(1, Math.abs((rand.nextInt()) % (COUNT / 200)));
size = Math.min(COUNT - counter, size);
if (size < 1) {
break;
}
counter += size;
if (index == 0) {
newMap = new ExtensibleCellMap<>();
} else {
newMap = new ExtensibleCellMap<>(cellMapList.get(cellMapList.size() - 1), 3);
}
cellMapList.add(newMap);
System.out.println("Link Length: " + ExtensibleCellMap.getLinkLength(newMap));
for (int i = 0; i < size; i++) {
int symRandIndex = Math.abs(rand.nextInt() % COUNT);
Symbol tempSym = symList.get(symRandIndex);
newMap.put(tempSym, new MyString(tempSym.getName()));
}
}
System.out.println("Total number of maps produced: " + cellMapList.size() + "; Time: "
+ (System.nanoTime() - constTimer) / 1e9);
int random = Math.max(0, Math.abs(rand.nextInt() % cellMapList.size()));
testMap = cellMapList.get(random);
long timer = System.nanoTime();
for (int i = 0; i < 1e6; i++) {
int randomSym = Math.max(0, Math.abs(rand.nextInt() % symList.size()));
testMap.get(symList.get(randomSym));
testMap.containsKey(symList.get(randomSym));
}
System.out.println("Time taken to perform operations: " + (System.nanoTime() - timer) / 1e9);
System.exit(0);
}
public static void testNodeReplacedString() {
for (WhileStatement whileStmt : Misc.getInheritedEnclosee(Program.getRoot(), WhileStatement.class)) {
System.out.println(
whileStmt.getInfo().getNodeReplacedString(whileStmt.getInfo().getCFGInfo().getPredicate(), "33"));
}
System.exit(0);
}
private static void testNodeShifting() {
FunctionDefinition main = Program.getRoot().getInfo().getMainFunction();
Statement stmt1 = main.getInfo().getStatementWithLabel("l1");
Statement stmt2 = main.getInfo().getStatementWithLabel("l2");
System.out.println(stmt1);
System.out.println(stmt2);
DumpSnapshot.dumpNestedCFG(Program.getRoot(), "first");
InsertImmediatePredecessor.insert(stmt2, stmt1);
DumpSnapshot.dumpNestedCFG(Program.getRoot(), "second");
System.out.println(Program.getRoot());
System.exit(0);
}
private static void testNoNullParent() {
for (Node n : Misc.getInheritedEnclosee(Program.getRoot(), Node.class)) {
if (n instanceof TranslationUnit) {
continue;
}
if (n.getParent() == null) {
String str = n.getInfo().getString();
if (n instanceof BeginNode) {
str = "BeginNode";
} else if (n instanceof EndNode) {
str = "EndNode";
}
System.out
.println("Following node has some issue: " + str + " of type " + n.getClass().getSimpleName());
}
}
for (FunctionDefinition func : Misc.getInheritedEnclosee(Program.getRoot(), FunctionDefinition.class)) {
for (Node cfgNode : func.getInfo().getCFGInfo().getLexicalCFGContents()) {
if (cfgNode instanceof FunctionDefinition) {
continue;
}
if (Misc.getEnclosingFunction(cfgNode) != func) {
String str = cfgNode.getInfo().getString();
if (cfgNode instanceof BeginNode) {
str = "BeginNode";
} else if (cfgNode instanceof EndNode) {
str = "EndNode";
}
System.out.println(
"Following node has some issue: " + str + " of type " + cfgNode.getClass().getSimpleName());
}
for (Node n : cfgNode.getInfo().getCFGInfo().getSuccBlocks()) {
if (n instanceof ForConstruct) {
ForConstructInfo forInfo = (ForConstructInfo) n.getInfo();
if (!forInfo.getOmpClauseList().stream().anyMatch(c -> c instanceof NowaitClause)) {
System.out.println("Following node has some issue: " + cfgNode + " of type "
+ cfgNode.getClass().getSimpleName());
}
}
}
for (Node pred : cfgNode.getInfo().getCFGInfo().getPredBlocks()) {
if (!pred.getInfo().getCFGInfo().getSuccBlocks().contains(cfgNode)) {
System.out.println("Following pair are not connected bothways: " + pred + " with " + cfgNode);
}
}
for (Node succ : cfgNode.getInfo().getCFGInfo().getSuccBlocks()) {
if (!succ.getInfo().getCFGInfo().getPredBlocks().contains(cfgNode)) {
System.out.println("Following pair are not connected bothways: " + cfgNode + " with " + succ);
}
}
}
}
System.exit(0);
}
private static void testNormalization() {
ForConstruct forCons = FrontEnd.parseAndNormalize("\n#pragma omp for\nfor (i=0;i<10;i++){}",
ForConstruct.class);
for (DoStatement doStmt : Misc.getInheritedEnclosee(Program.getRoot().getInfo().getMainFunction(),
DoStatement.class)) {
doStmt.getInfo().getCFGInfo().setBody(forCons);
System.out.println(doStmt);
Expression exp = FrontEnd.parseAndNormalize("2 && foo()", Expression.class);
doStmt.getInfo().getCFGInfo().setPredicate(exp);
}
System.out.println(Program.getRoot());
DumpSnapshot.dumpRoot("final");
DumpSnapshot.dumpNestedCFG(Program.getRoot(), Program.fileName);
System.exit(0);
}
private static void testPA_A4() {
CellList riskyDerefs = new CellList();
for (FunctionDefinition func : Program.getRoot().getInfo().getAllFunctionDefinitions()) {
for (CastExpression castExp : func.getInfo().getPointerDereferencedExpressionReads()) {
Node cfgNode = Misc.getCFGNodeFor(castExp);
for (Cell accessed : castExp.getInfo().getLocationsOf()) {
boolean risky = false; 
if (accessed.getPointsTo(cfgNode).contains(Cell.getNullCell())) {
risky = true;
} else {
for (Cell cell : accessed.getPointsTo(cfgNode)) {
if (cell instanceof HeapCell) {
HeapCell heapCell = (HeapCell) cell;
if (!heapCell.isValidAt(cfgNode)) {
risky = true;
break;
}
}
}
}
if (risky) {
if (cfgNode instanceof ExpressionStatement) {
Cell temporaryCopy;
ExpressionStatement expStmt = (ExpressionStatement) cfgNode;
for (Assignment assign : expStmt.getInfo().getLexicalAssignments()) {
if (assign.rhs.toString().contains("*" + accessed)) {
riskyDerefs.addAll(assign.getLHSLocations());
}
}
System.out.println("Deleting the following assignment: " + expStmt);
NodeRemover.removeNode(expStmt);
}
}
}
}
}
for (CallStatement callStmt : Misc.getInheritedEnclosee(Program.getRoot(), CallStatement.class)) {
if (callStmt.getFunctionDesignatorNode().toString().equals("printf")) {
CellList reads = callStmt.getInfo().getReads();
if (Misc.doIntersect(reads, riskyDerefs)) {
System.out.println("Deleting the following print statement: " + callStmt);
NodeRemover.removeNode(callStmt);
}
}
}
DumpSnapshot.dumpRoot("output");
System.exit(0);
}
private static void testPA_Sample() {
for (FunctionDefinition func : Program.getRoot().getInfo().getAllFunctionDefinitions()) {
for (CastExpression castExp : func.getInfo().getPointerDereferencedExpressionWrites()) {
Node cfgNode = Misc.getCFGNodeFor(castExp);
String printStr = "printf(\"true\");";
Statement callStmt = FrontEnd.parseAndNormalize(printStr, Statement.class);
InsertImmediatePredecessor.insert(cfgNode, callStmt);
}
}
DumpSnapshot.dumpRoot("sample");
System.exit(0);
}
private static void testParConsExpansionAndFusion() {
ParallelConstructExpander.mergeParallelRegions(Program.getRoot());
DumpSnapshot.dumpRoot("expanded");
System.err.println("Number of times IDFA update were triggered -- ");
for (FlowAnalysis<?> analysis : FlowAnalysis.getAllAnalyses().values()) {
System.err.println("\t For " + analysis.getAnalysisName() + ": " + analysis.autoUpdateTriggerCounter);
}
System.err.println("Total number of times nodes were processed during automated IDFA update -- ");
for (FlowAnalysis<?> analysis : FlowAnalysis.getAllAnalyses().values()) {
System.err.println("\t For " + analysis.getAnalysisName() + ": " + analysis.nodesProcessedDuringUpdate);
}
System.err.println("Time spent in forward IDFA updates -- ");
for (FlowAnalysis<?> analysis : FlowAnalysis.getAllAnalyses().values()) {
System.err.println(
"\t For " + analysis.getAnalysisName() + ": " + analysis.flowAnalysisUpdateTimer / (1e9) + "s.");
}
System.err.println("Time spent in SVE queries: " + SVEChecker.cpredaTimer / (1e9 * 1.0) + "s.");
System.err.println("Time spent in phase update: "
+ (AbstractPhase.stabilizationTime + BeginPhasePoint.phaseAnalysisTime) / (1e9 * 1.0) + "s.");
System.err.println("Time spent in inlining: " + FunctionInliner.inliningTimer / (1e9 * 1.0) + "s.");
System.err.println("Time spent in having uni-task precision in IDFA edge creation: "
+ CFGInfo.uniPrecisionTimer / (1e9 * 1.0) + "s.");
System.err.println("Number of field-sensitive queries: " + FieldSensitivity.counter);
System.err.println("Time spent in field-sensitive queries: " + FieldSensitivity.timer / (1e9 * 1.0) + "s.");
System.err.println("Time spent in generating reverse postordering of the program nodes: "
+ TraversalOrderObtainer.orderGenerationTime / (1e9 * 1.0) + "s.");
if (Program.fieldSensitive) {
DumpSnapshot.printToFile(ConstraintsGenerator.allConstraintString, Program.fileName + "_z3_queries.txt");
}
System.err.println(
"TOTAL TIME (including disk I/O time): " + (System.nanoTime() - Main.totalTime) / (1.0 * 1e9) + "s.");
DumpSnapshot.printToFile(Program.getRoot(), "imop_output.i");
DumpSnapshot.dumpPhases("final");
System.exit(0);
}
public static void testPhaseAdditions() {
FunctionDefinition main = Program.getRoot().getInfo().getMainFunction();
assert (main != null);
for (WhileStatement itStmt : Misc.getInheritedEnclosee(main, WhileStatement.class)) {
List<CriticalConstruct> critNode = Misc.getInheritedEncloseeList(itStmt, CriticalConstruct.class);
CriticalConstruct crit = critNode.get(0);
CompoundStatement compStmt = (CompoundStatement) Misc.getEnclosingBlock(crit);
int index = compStmt.getInfo().getCFGInfo().getElementList().indexOf(crit);
compStmt.getInfo().getCFGInfo().addElement(index,
FrontEnd.parseAndNormalize("if (1) {break;}", Statement.class));
}
Program.getRoot().getInfo().removeExtraScopes();
DumpSnapshot.dumpPhases("after");
System.exit(0);
}
public static void testPhaseInWhile() {
FunctionDefinition main = Program.getRoot().getInfo().getMainFunction();
assert (main != null);
for (WhileStatement itStmt : Misc.getInheritedEnclosee(main, WhileStatement.class)) {
CompoundStatement oldStmt = (CompoundStatement) itStmt.getInfo().getCFGInfo().getBody();
CompoundStatement copyStmt = FrontEnd.parseAlone(
"{if (1) {foo(); #pragma omp barrier\n} else {#pragma omp barrier\nfoo();}}",
CompoundStatement.class);
itStmt.getInfo().getCFGInfo().setBody(copyStmt);
}
DumpSnapshot.dumpPhases("after");
System.exit(0);
}
public static void testPhaseRemovals() {
FunctionDefinition main = Program.getRoot().getInfo().getMainFunction();
assert (main != null);
for (WhileStatement itStmt : Misc.getInheritedEnclosee(main, WhileStatement.class)) {
List<CriticalConstruct> critNode = Misc.getInheritedEncloseeList(itStmt, CriticalConstruct.class);
CriticalConstruct crit = critNode.get(0);
CompoundStatement compStmt = (CompoundStatement) Misc.getEnclosingBlock(crit);
int index = compStmt.getInfo().getCFGInfo().getElementList().indexOf(crit);
compStmt.getInfo().getCFGInfo().removeElement(index + 2);
}
DumpSnapshot.dumpPhases("after");
System.exit(0);
}
public static void testPhaseSwaps() {
FunctionDefinition main = Program.getRoot().getInfo().getMainFunction();
assert (main != null);
for (WhileStatement itStmt : Misc.getInheritedEnclosee(main, WhileStatement.class)) {
List<CriticalConstruct> critNode = Misc.getInheritedEncloseeList(itStmt, CriticalConstruct.class);
CriticalConstruct crit = critNode.get(0);
CompoundStatement compStmt = (CompoundStatement) Misc.getEnclosingBlock(crit);
int index = compStmt.getInfo().getCFGInfo().getElementList().indexOf(crit);
BarrierDirective barrNode = Misc.getInheritedEncloseeList(itStmt, BarrierDirective.class).get(2);
int newIndex = compStmt.getInfo().getCFGInfo().getElementList().indexOf(barrNode);
Node element = compStmt.getInfo().getCFGInfo().getElementList().get(index + 2);
compStmt.getInfo().getCFGInfo().removeElement(element);
compStmt.getInfo().getCFGInfo().addElement(newIndex + 1, element);
}
DumpSnapshot.dumpPhases("after");
System.exit(0);
}
public static void testPointerDereferenceGetter() {
for (ExpressionStatement expStmt : Misc.getInheritedEnclosee(Program.getRoot(), ExpressionStatement.class)) {
if (expStmt.getF0().present()) {
Expression exp = (Expression) expStmt.getF0().getNode();
List<CastExpression> accessReads = exp.getInfo().getPointerDereferencedExpressionReads();
List<CastExpression> accessWrites = exp.getInfo().getPointerDereferencedExpressionWrites();
if (!accessReads.isEmpty() || !accessWrites.isEmpty()) {
System.err.println("For expression " + exp);
}
for (CastExpression ae : accessReads) {
System.err.println("\tREAD:" + ae + ".");
}
for (CastExpression ae : accessWrites) {
System.err.println("\tWRITE:" + ae + ".");
}
}
}
System.exit(0);
}
private static void testReachableUses() {
FrontEnd.parseAndNormalize(
"int main() {int x; int y; x = 10; y = x; l1: x = y + 10; y = ++x + 11; x = x + 3; y = x; x = x + 1;}");
FunctionDefinition main = Program.getRoot().getInfo().getMainFunction();
ExpressionStatement stmt = (ExpressionStatement) main.getInfo().getStatementWithLabel("l1");
Cell c = stmt.getInfo().getWrites().get(0);
System.out.println(stmt.getInfo().getAllUsesForwardsExclusively(c));
CellSet set = new CellSet();
set.add(c);
System.out.println(stmt.getInfo().getFirstPossibleKillersForwardExclusively(set));
System.exit(0);
}
private static void testReversePostOrder() {
FunctionDefinition main = Program.getRoot().getInfo().getMainFunction();
if (main == null) {
System.exit(0);
}
Node beginNode = main.getInfo().getCFGInfo().getNestedCFG().getBegin();
long timer = System.nanoTime();
List<Node> reversePostOrder = TraversalOrderObtainer.obtainReversePostOrder(beginNode,
n -> n.getInfo().getCFGInfo().getInterProceduralLeafSuccessors());
long finalTime = System.nanoTime() - timer;
for (Node node : reversePostOrder) {
if (node instanceof BeginNode) {
System.err.println("B (" + node.getParent().getClass().getSimpleName() + ")");
} else if (node instanceof EndNode) {
System.err.println("E (" + node.getParent().getClass().getSimpleName() + ")");
} else {
System.err.println(node);
}
}
for (int i = 0; i < reversePostOrder.size(); i++) {
Node n1 = reversePostOrder.get(i);
for (int j = i + 1; j < reversePostOrder.size(); j++) {
Node n2 = reversePostOrder.get(j);
if (n1 == n2) {
System.err.println("WHY: " + n1 + "?");
}
}
}
System.err.println("Time spent in generating reverse postordering of the program nodes: "
+ TraversalOrderObtainer.orderGenerationTime / (1e9 * 1.0) + "s.");
System.exit(0);
}
private static void testSVEness() {
for (WhileStatement whileStmt : Misc.getInheritedEnclosee(Program.getRoot(), WhileStatement.class)) {
for (Expression exp : Misc.getExactEnclosee(whileStmt, Expression.class)) {
if (Misc.isAPredicate(exp)) {
System.out.println("\n*** Testing SVEness of " + exp);
SVEChecker.isSingleValuedPredicate(exp);
}
}
for (ExpressionStatement expStmt : Misc.getInheritedEnclosee(whileStmt, ExpressionStatement.class)) {
SVEChecker.writesSingleValue(expStmt);
}
}
System.exit(0);
}
private static void testSwapLength() {
FrontEnd.parseAndNormalize(
"int main () {int A; int B; int C; int D; \n#pragma omp parallel\n{A=B;B=C;C=D;D=A;}}");
FunctionDefinition funcDef = Program.getRoot().getInfo().getMainFunction();
assert (funcDef != null);
for (ParallelConstruct parCons : Misc.getInheritedEnclosee(funcDef, ParallelConstruct.class)) {
CompoundStatement parBody = (CompoundStatement) parCons.getInfo().getCFGInfo().getBody();
for (Node elem : parBody.getInfo().getCFGInfo().getElementList()) {
if (elem instanceof ExpressionStatement) {
ExpressionStatement expStmt = (ExpressionStatement) elem;
if (expStmt.getInfo().isCopyInstruction()) {
System.out.println("Length of swap starting at " + expStmt + " is "
+ CopyEliminator.swapLength(expStmt, parBody));
}
}
}
}
System.exit(0);
}
private static void testLists() {
totalTime = System.nanoTime();
List<Integer> list = Arrays.asList(new Integer[(int) 5e7]);
System.out.println("Time taken to construct: " + (System.nanoTime() - totalTime) / 1e9);
totalTime = System.nanoTime();
for (int i = 0; i < 5e7; i++) {
list.set(i, 100);
}
System.out.println("Time taken to insert: " + (System.nanoTime() - totalTime) / 1e9);
totalTime = System.nanoTime();
list = new ArrayList<>();
for (int i = 0; i < 5e7; i++) {
list.add(i);
}
System.out.println("Time taken to construct and add: " + (System.nanoTime() - totalTime) / 1e9);
System.exit(0);
}
private static void testUnreachables() {
FrontEnd.parseAndNormalize(
"int foo() {int p; p = 10; return p;} int main() {int x; x = 2; if (x > 3) {goto l2;}l1:while(x > 4){x = 11; foo();if(x>1){x = 10;goto l3;l2:;} x = 12; } l3: x = 14;}",
TranslationUnit.class);
WhileStatement whileStmt = (WhileStatement) Program.getRoot().getInfo().getStatementWithLabel("l1");
whileStmt.getInfo().getCFGInfo().getEndUnreachableLeafHeaders().stream().forEach(e -> {
System.out.println("E: " + e.getClass().getSimpleName());
});
whileStmt.getInfo().getCFGInfo().getBeginUnreachableLeafHeaders().stream().forEach(e -> {
System.out.println("B: " + e.getClass().getSimpleName());
});
whileStmt.getInfo().getCFGInfo().getBothBeginAndEndReachableIntraTaskLeafNodes().stream().forEach(e -> {
System.out.println(e);
});
System.exit(0);
}
public static void testValueConstraintsGenerator() {
for (ExpressionStatement expStmt : Misc.getInheritedEnclosee(Program.getRoot(), ExpressionStatement.class)) {
if (expStmt.getF0().present()) {
Expression exp = (Expression) expStmt.getF0().getNode();
}
}
System.exit(0);
}
public static void testZ3ExpressionCreator() {
List<Tokenizable> tokenList = new ArrayList<>();
Constant const1 = FrontEnd.parseAndNormalize("299.5", Constant.class);
Constant const2 = FrontEnd.parseAndNormalize("4", Constant.class);
NodeToken id1 = new NodeToken("id1");
IdOrConstToken spe1 = new IdOrConstToken(id1, null);
NodeToken id2 = new NodeToken("id2");
IdOrConstToken spe2 = new IdOrConstToken(id2, null);
IdOrConstToken spe3 = new IdOrConstToken(const1);
IdOrConstToken spe4 = new IdOrConstToken(const2);
tokenList.addAll(Arrays.asList(OperatorToken.ASSIGN, OperatorToken.MINUS, OperatorToken.MUL, spe4,
OperatorToken.PLUS, spe3, spe1, spe2, spe2));
System.out.println(tokenList);
Context c = new Context();
HashMap<String, ArithExpr> idMap = new HashMap<>();
idMap.put(id1.toString(), c.mkIntConst(id1.toString()));
idMap.put(id2.toString(), c.mkIntConst(id2.toString()));
Expr finalExpr = ConstraintsGenerator.getZ3Expression(tokenList, idMap, c);
System.out.println(finalExpr);
Solver solver = c.mkSimpleSolver();
if (finalExpr instanceof BoolExpr) {
solver.add((BoolExpr) finalExpr);
}
Status st = solver.check();
if (st == Status.SATISFIABLE) {
Model m = solver.getModel();
for (ArithExpr e : idMap.values()) {
System.out.println(e + ": " + m.getConstInterp(e));
}
} else {
System.out.println("No solution found for the system.");
}
System.exit(0);
}
public static void validateRDs() {
FunctionDefinition main = Program.getRoot().getInfo().getMainFunction();
if (main == null) {
return;
}
for (Node leaf : main.getInfo().getCFGInfo().getIntraTaskCFGLeafContents()) {
for (Definition rdDef : leaf.getInfo().getReachingDefinitions()) {
Node rd = rdDef.getDefiningNode();
if (rd instanceof Declaration) {
continue;
}
if (!rd.getInfo().isConnectedToProgram()) {
Misc.exitDueToError("Found an unstable reaching-definition state: " + rd
+ " is not connected to the program, and is yet considered to be a reaching definition for "
+ leaf);
}
}
}
}
public static class CriticalNode {
public static void connect(CriticalNode cn1, CriticalNode cn2) {
if (cn1.getSuccNodes().contains(cn2)) {
return;
}
cn1.getSuccNodes().add(cn2);
cn2.getPredNodes().add(cn1);
}
private final CriticalConstruct cc;
private CellSet accessedCells;
private List<CriticalNode> succNodes = new ArrayList<>();
private List<CriticalNode> predNodes = new ArrayList<>();
public CriticalNode(CriticalConstruct cc, CellSet accessedCells) {
this.cc = cc;
this.accessedCells = accessedCells;
System.out.println("Creating node with " + accessedCells.size() + " entries.");
}
CellSet getAccessedCells() {
return accessedCells;
}
CriticalConstruct getCriticalConstruct() {
return cc;
}
List<CriticalNode> getPredNodes() {
return predNodes;
}
List<CriticalNode> getSuccNodes() {
return succNodes;
}
}
}
