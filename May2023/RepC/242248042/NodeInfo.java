package imop.ast.info;
import imop.ast.annotation.IncompleteSemantics;
import imop.ast.annotation.Label;
import imop.ast.annotation.PragmaImop;
import imop.ast.annotation.SimpleLabel;
import imop.ast.info.cfgNodeInfo.CompoundStatementInfo;
import imop.ast.info.cfgNodeInfo.FunctionDefinitionInfo;
import imop.ast.node.external.*;
import imop.ast.node.internal.*;
import imop.deprecated.Deprecated_FlowFact;
import imop.deprecated.Deprecated_InterProceduralCFGPass;
import imop.lib.analysis.Assignment;
import imop.lib.analysis.AssignmentGetter;
import imop.lib.analysis.SVEChecker;
import imop.lib.analysis.flowanalysis.*;
import imop.lib.analysis.flowanalysis.controlflow.DominanceAnalysis;
import imop.lib.analysis.flowanalysis.controlflow.DominanceAnalysis.DominatorFlowFact;
import imop.lib.analysis.flowanalysis.controlflow.PredicateAnalysis;
import imop.lib.analysis.flowanalysis.controlflow.CrossCallPredicateAnalysis;
import imop.lib.analysis.flowanalysis.dataflow.*;
import imop.lib.analysis.flowanalysis.dataflow.CopyPropagationAnalysis.CopyPropagationFlowMap;
import imop.lib.analysis.flowanalysis.dataflow.DataDependenceForward.DataDependenceForwardFF;
import imop.lib.analysis.flowanalysis.dataflow.PointsToAnalysis.PointsToFlowMap;
import imop.lib.analysis.flowanalysis.dataflow.ReachingDefinitionAnalysis.ReachingDefinitionFlowMap;
import imop.lib.analysis.flowanalysis.generic.AnalysisName;
import imop.lib.analysis.flowanalysis.generic.CellularDataFlowAnalysis;
import imop.lib.analysis.flowanalysis.generic.ControlFlowAnalysis;
import imop.lib.analysis.flowanalysis.generic.FlowAnalysis;
import imop.lib.analysis.flowanalysis.generic.FlowAnalysis.FlowFact;
import imop.lib.analysis.mhp.AbstractPhase;
import imop.lib.analysis.mhp.incMHP.BeginPhasePoint;
import imop.lib.analysis.mhp.incMHP.NodePhaseInfo;
import imop.lib.analysis.mhp.OldLock;
import imop.lib.analysis.mhp.incMHP.Phase;
import imop.lib.analysis.mhp.yuan.BTNode;
import imop.lib.analysis.mhp.yuan.BarrierTreeConstructor;
import imop.lib.analysis.solver.BaseSyntax;
import imop.lib.analysis.solver.PointerDereferenceGetter;
import imop.lib.analysis.solver.SyntacticAccessExpression;
import imop.lib.analysis.solver.SyntacticAccessExpressionGetter;
import imop.lib.analysis.solver.tokens.Tokenizable;
import imop.lib.analysis.typesystem.*;
import imop.lib.cfg.info.CFGInfo;
import imop.lib.cfg.link.autoupdater.AutomatedUpdater;
import imop.lib.cg.CallSite;
import imop.lib.cg.CallStack;
import imop.lib.cg.NodeWithStack;
import imop.lib.getter.*;
import imop.lib.getter.StringGetter.Commentor;
import imop.lib.transform.percolate.DriverModule;
import imop.lib.transform.simplify.CompoundStatementNormalizer;
import imop.lib.util.*;
import imop.parser.FrontEnd;
import imop.parser.Program;
import imop.parser.Program.UpdateCategory;
import java.util.*;
public class NodeInfo implements Cloneable {
public static boolean readWriteDestinationsSet = false;
private Node node;
private Node copySourceNode;
private int idNumber; 
protected HashMap<AnalysisName, FlowFact> flowFactsIN;
protected HashMap<AnalysisName, FlowFact> flowFactsOUT;
protected CellSet impactedSet;
protected Map<CellularDataFlowAnalysis<?>, CellSet> accessedCellSets;
protected Map<CellularDataFlowAnalysis<?>, CellSet> readCellSets;
protected Map<CellularDataFlowAnalysis<?>, CellSet> writtenCellSets;
protected CFGInfo cfgInfo;
private NodePhaseInfo phaseInfo;
private List<ParallelConstruct> regionInfo;
private LinkedList<CallStatement> callStatements = null;
private IncompleteSemantics incomplete;
private static enum TRISTATE {
YES, NO, UNKNOWN
}
private TRISTATE reliesOnPtsTo = TRISTATE.UNKNOWN;
private TRISTATE mayWriteToPointerType = TRISTATE.UNKNOWN;
private int reversePostOrderId = -1;
private CellList readList;
private CellList writeList;
private CellSet sharedReadSet;
private CellSet sharedWriteSet;
private CellSet nonFieldReadSet;
private CellSet nonFieldWriteSet;
private CellMap<NodeSet> readDestinations;
private CellMap<NodeSet> writeDestinations;
private List<SyntacticAccessExpression> baseAccessExpressionReads;
private List<SyntacticAccessExpression> baseAccessExpressionWrites;
private List<CastExpression> pointerDereferencedExpressionReads;
private List<CastExpression> pointerDereferencedExpressionWrites;
private List<String> comments;
private List<PragmaImop> pragmaAnnotations;
private static Commentor defaultCommentor;
private static boolean livenessDone = false;
private static boolean ddfDone = false;
public static boolean rdDone = false;
private static boolean daDone = false;
private static boolean hvDone = false;
private static boolean paDone = false;
private BTNode btNode = null;
public static long callQueryTimer = 0;
private Set<Node> forwardReachableNodes = null;
private Set<Node> backwardReachableNodes = null;
private Set<Node> forwardReachableBarriers = null;
private Set<Node> backwardReachableBarriers = null;
public Set<Node> getFWRNodes() {
if (this.node instanceof EndNode && this.node.getParent() instanceof ParallelConstruct) {
return new HashSet<>();
}
populateForward();
return this.forwardReachableNodes;
}
public Set<Node> getFWRBarriers() {
if (this.node instanceof EndNode && this.node.getParent() instanceof ParallelConstruct) {
return new HashSet<>();
}
populateForward();
return this.forwardReachableBarriers;
}
public Set<Node> getBWRNodes() {
if (this.node instanceof BeginNode && this.node.getParent() instanceof ParallelConstruct) {
return new HashSet<>();
}
populateBackward();
return this.backwardReachableNodes;
}
public Set<Node> getBWRBarriers() {
if (this.node instanceof BeginNode && this.node.getParent() instanceof ParallelConstruct) {
return new HashSet<>();
}
populateBackward();
return this.backwardReachableBarriers;
}
private void populateForward() {
Set<NodeWithStack> endPoints = new HashSet<>();
Set<NodeWithStack> reachablesWithStack = new HashSet<>();
for (NodeWithStack succ : this.getNode().getInfo().getCFGInfo()
.getInterProceduralLeafSuccessors(new CallStack())) {
Set<NodeWithStack> someEndPoints = new HashSet<>();
reachablesWithStack
.addAll(CollectorVisitor.collectNodesIntraTaskForwardBarrierFreePath(succ, someEndPoints));
reachablesWithStack.add(succ);
reachablesWithStack.addAll(someEndPoints);
endPoints.addAll(someEndPoints);
}
this.forwardReachableNodes = new HashSet<>();
for (NodeWithStack nws : reachablesWithStack) {
this.forwardReachableNodes.add(nws.getNode());
}
this.forwardReachableBarriers = new HashSet<>();
for (NodeWithStack nws : endPoints) {
this.forwardReachableBarriers.add(nws.getNode());
}
}
private void populateBackward() {
Set<NodeWithStack> endPoints = new HashSet<>();
Set<NodeWithStack> reachablesWithStack = new HashSet<>();
for (NodeWithStack pred : this.getNode().getInfo().getCFGInfo()
.getInterProceduralLeafPredecessors(new CallStack())) {
Set<NodeWithStack> someEndPoints = new HashSet<>();
reachablesWithStack
.addAll(CollectorVisitor.collectNodesIntraTaskBackwardBarrierFreePath(pred, someEndPoints));
reachablesWithStack.add(pred);
reachablesWithStack.addAll(someEndPoints);
endPoints.addAll(someEndPoints);
}
this.backwardReachableNodes = new HashSet<>();
for (NodeWithStack nws : reachablesWithStack) {
this.backwardReachableNodes.add(nws.getNode());
}
this.backwardReachableBarriers = new HashSet<>();
for (NodeWithStack nws : endPoints) {
this.backwardReachableBarriers.add(nws.getNode());
}
}
private Set<Expression> jumpedPredicates;
public Set<Expression> getJumpedPredicates() {
if (jumpedPredicates == null) {
jumpedPredicates = new HashSet<>();
}
return jumpedPredicates;
}
public int getReversePostOrder() {
if (CFGInfo.isSCCStale) {
SCC.initializeSCC();
}
SCC scc = this.getCFGInfo().getSCC();
if (scc == null) {
} else {
scc.stabilizeInternalRPO();
}
return this.reversePostOrderId;
}
public void setReversePostOrderId(int i) {
this.reversePostOrderId = i;
}
@Deprecated
private boolean stabilizePointsToIfRequired() {
FlowAnalysis<?> pointsTo = FlowAnalysis.getAllAnalyses().get(AnalysisName.POINTSTO);
if (pointsTo == null || !pointsTo.stateIsInvalid()) {
return false;
}
pointsTo.markStateToBeValid();
Program.memoizeAccesses++;
pointsTo.restartAnalysisFromStoredNodes();
Program.memoizeAccesses--;
return true;
}
private boolean pointsToMayBeStale() {
FlowAnalysis<?> pointsTo = FlowAnalysis.getAllAnalyses().get(AnalysisName.POINTSTO);
if (pointsTo != null && !pointsTo.stateIsInvalid()) {
return false;
} else {
return true;
}
}
public static void resetFirstRunFlags() {
NodeInfo.ddfDone = false;
NodeInfo.daDone = false;
NodeInfo.livenessDone = false;
NodeInfo.rdDone = false;
NodeInfo.callQueryTimer = 0;
NodeInfo.defaultCommentor = null;
}
public static void checkFirstRun(AnalysisName analysisName) {
if (analysisName == AnalysisName.PSEUDO_INTER_PREDICATE_ANALYSIS
|| analysisName == AnalysisName.CROSSCALL_PREDICATE_ANALYSIS) {
if (!NodeInfo.paDone) {
NodeInfo.paDone = true;
performPredicateAnalysis();
}
} else if (analysisName == AnalysisName.LIVENESS) {
if (!NodeInfo.livenessDone) {
NodeInfo.livenessDone = true;
performLivenessAnalysis();
}
} else if (analysisName == AnalysisName.DATA_DEPENDENCE_FORWARD) {
if (!NodeInfo.ddfDone) {
NodeInfo.ddfDone = true;
performDDF();
}
} else if (analysisName == AnalysisName.REACHING_DEFINITION) {
if (!NodeInfo.rdDone) {
NodeInfo.rdDone = true;
performRDA();
}
} else if (analysisName == AnalysisName.DOMINANCE) {
if (!NodeInfo.daDone) {
NodeInfo.daDone = true;
performDominanceAnalysis();
}
} else if (analysisName == AnalysisName.HEAP_VALIDITY) {
if (!NodeInfo.hvDone) {
NodeInfo.hvDone = true;
performHeapValidityAnalysis();
}
}
}
private static void performPredicateAnalysis() {
FunctionDefinition mainFunc = Program.getRoot().getInfo().getMainFunction();
if (mainFunc == null) {
return;
}
System.err.println("Pass: Performing predicate analysis.");
long timeStart = System.nanoTime();
ControlFlowAnalysis<?> pa;
if (Program.useInterProceduralPredicateAnalysis) {
pa = new CrossCallPredicateAnalysis();
pa.run(mainFunc);
} else {
pa = new PredicateAnalysis();
pa.run(mainFunc);
}
long timeTaken = System.nanoTime() - timeStart;
SVEChecker.cpredaTimer += timeTaken;
System.err.println("\tNodes processed " + pa.nodesProcessed + " times.");
System.err.println("\tTime taken: " + timeTaken / 1.0e9 + "s.");
DumpSnapshot.dumpPredicates("");
}
private static void performLivenessAnalysis() {
FunctionDefinition mainFunc = Program.getRoot().getInfo().getMainFunction();
if (mainFunc == null) {
return;
}
System.err.println("Pass: Performing liveness analysis.");
long timeStart = System.nanoTime();
LivenessAnalysis lva = new LivenessAnalysis();
lva.run(mainFunc);
long timeTaken = System.nanoTime() - timeStart;
System.err.println("\tNodes processed " + lva.nodesProcessed + " times.");
System.err.println("\tTime taken: " + timeTaken / 1000000000.0 + "s.");
}
private static void performRDA() {
FunctionDefinition mainFunc = Program.getRoot().getInfo().getMainFunction();
if (mainFunc == null) {
return;
}
System.err.println("Pass: Performing reaching-definition analysis.");
long timeStart = System.nanoTime();
ReachingDefinitionAnalysis rda = new ReachingDefinitionAnalysis();
rda.run(mainFunc);
long timeTaken = System.nanoTime() - timeStart;
System.err.println("\tNodes processed " + rda.nodesProcessed + " times.");
System.err.println("\tTime taken: " + timeTaken / 1000000000.0 + "s.");
}
private static void performDDF() {
FunctionDefinition mainFunc = Program.getRoot().getInfo().getMainFunction();
if (mainFunc == null) {
return;
}
System.err.println("Pass: Performing data-dependence analysis.");
System.err.println("\tPerforming forward R/W analysis.");
long timeStart = System.nanoTime();
DataDependenceForward ddf = new DataDependenceForward();
ddf.run(mainFunc);
long timeTaken = System.nanoTime() - timeStart;
System.err.println("\tNodes processed (forward) " + ddf.nodesProcessed + " times.");
System.err.println("\tTime taken: " + timeTaken / 1000000000.0 + "s.");
DumpSnapshot.dumpWriteSources();
}
private static void performDominanceAnalysis() {
FunctionDefinition mainFunc = Program.getRoot().getInfo().getMainFunction();
if (mainFunc == null) {
return;
}
System.err.println("Pass: Performing dominator analysis.");
long timeStart = System.nanoTime();
DominanceAnalysis da = new DominanceAnalysis();
da.run(mainFunc);
long timeTaken = System.nanoTime() - timeStart;
System.err.println("\tNodes processed " + da.nodesProcessed + " times.");
System.err.println("\tTime taken: " + timeTaken / 1000000000.0 + "s.");
}
private static void performHeapValidityAnalysis() {
FunctionDefinition mainFunc = Program.getRoot().getInfo().getMainFunction();
if (mainFunc == null) {
return;
}
System.err.println("Pass: Performing heap validity analysis.");
long timeStart = System.nanoTime();
HeapValidityAnalysis hva = new HeapValidityAnalysis();
hva.run(mainFunc);
long timeTaken = System.nanoTime() - timeStart;
System.err.println("\tNodes processed " + hva.nodesProcessed + " times.");
System.err.println("\tTime taken: " + timeTaken / 1000000000.0 + "s.");
}
public FlowFact getIN(AnalysisName analysisName) {
return this.getIN(analysisName, null);
}
public FlowFact getIN(AnalysisName analysisName, Cell thisCell) {
ProfileSS.addRelevantChangePoint(ProfileSS.ptaSet);
checkFirstRun(analysisName);
FlowAnalysis<?> analysisHandle = FlowAnalysis.getAllAnalyses().get(analysisName);
if (analysisHandle != null) {
if (Program.idfaUpdateCategory == UpdateCategory.EGINV || Program.idfaUpdateCategory == UpdateCategory.EGUPD
|| Program.idfaUpdateCategory == UpdateCategory.CPINV
|| Program.idfaUpdateCategory == UpdateCategory.CPUPD) {
assert (!analysisHandle.stateIsInvalid());
} else if (Program.idfaUpdateCategory == UpdateCategory.LZINV) {
if (Program.concurrencyAlgorithm == Program.ConcurrencyAlgorithm.YCON
|| Program.mhpUpdateCategory == UpdateCategory.LZINV) {
if (AbstractPhase.globalMHPStale) {
AbstractPhase.globalMHPStale = false;
AutomatedUpdater.reinitMHP();
}
} else {
BeginPhasePoint.stabilizeStaleBeginPhasePoints();
}
if (analysisHandle.stateIsInvalid()) {
analysisHandle.markStateToBeValid();
AutomatedUpdater.reinitIDFA(analysisHandle);
}
} else {
assert (Program.idfaUpdateCategory == UpdateCategory.LZUPD);
if (thisCell == null) {
if (analysisHandle.stateIsInvalid()) {
if (Program.concurrencyAlgorithm == Program.ConcurrencyAlgorithm.YCON
|| Program.mhpUpdateCategory == UpdateCategory.LZINV) {
if (AbstractPhase.globalMHPStale) {
AbstractPhase.globalMHPStale = false;
AutomatedUpdater.reinitMHP();
}
} else {
BeginPhasePoint.stabilizeStaleBeginPhasePoints();
}
analysisHandle.markStateToBeValid();
if (analysisName == AnalysisName.POINTSTO) {
Program.basePointsTo = false;
Program.memoizeAccesses++;
analysisHandle.restartAnalysisFromStoredNodes();
Program.memoizeAccesses--;
} else {
analysisHandle.restartAnalysisFromStoredNodes();
}
}
} else {
if (analysisHandle.stateIsInvalid()
&& (analysisName != AnalysisName.POINTSTO || !PointsToAnalysis.isHeuristicEnabled
|| PointsToAnalysis.affectedCellsInThisEpoch.contains(thisCell))) {
if (Program.concurrencyAlgorithm == Program.ConcurrencyAlgorithm.YCON
|| Program.mhpUpdateCategory == UpdateCategory.LZINV) {
if (AbstractPhase.globalMHPStale) {
AbstractPhase.globalMHPStale = false;
AutomatedUpdater.reinitMHP();
}
} else {
BeginPhasePoint.stabilizeStaleBeginPhasePoints();
}
analysisHandle.markStateToBeValid();
if (analysisName == AnalysisName.POINTSTO) {
Program.basePointsTo = false;
Program.memoizeAccesses++;
analysisHandle.restartAnalysisFromStoredNodes();
Program.memoizeAccesses--;
} else {
analysisHandle.restartAnalysisFromStoredNodes();
}
}
}
}
}
if (flowFactsIN == null) {
flowFactsIN = new HashMap<>();
}
return flowFactsIN.get(analysisName);
}
public FlowFact getCurrentIN(AnalysisName analysisName) {
checkFirstRun(analysisName);
if (flowFactsIN == null) {
flowFactsIN = new HashMap<>();
}
return flowFactsIN.get(analysisName);
}
public FlowFact getCurrentOUT(AnalysisName analysisName) {
checkFirstRun(analysisName);
if (flowFactsOUT == null) {
flowFactsOUT = new HashMap<>();
}
return flowFactsOUT.get(analysisName);
}
public void setIN(AnalysisName analysisName, FlowFact flowFact) {
if (flowFactsIN == null) {
flowFactsIN = new HashMap<>();
}
FlowFact oldIN = flowFactsIN.get(analysisName);
if (oldIN == flowFact) {
return;
}
if (analysisName == AnalysisName.POINTSTO) {
if (this.mayRelyOnPtsTo()) {
if (oldIN == null || !oldIN.isEqualTo(flowFact)) {
this.invalidateAccessLists();
}
}
}
flowFactsIN.put(analysisName, flowFact);
return;
}
public FlowFact getOUT(AnalysisName analysisName) {
ProfileSS.addRelevantChangePoint(ProfileSS.ptaSet);
checkFirstRun(analysisName);
FlowAnalysis<?> analysisHandle = FlowAnalysis.getAllAnalyses().get(analysisName);
if (analysisHandle != null) {
if (Program.idfaUpdateCategory == UpdateCategory.EGINV || Program.idfaUpdateCategory == UpdateCategory.EGUPD
|| Program.idfaUpdateCategory == UpdateCategory.CPINV
|| Program.idfaUpdateCategory == UpdateCategory.CPUPD) {
assert (!analysisHandle.stateIsInvalid());
} else if (Program.idfaUpdateCategory == UpdateCategory.LZINV) {
if (Program.concurrencyAlgorithm == Program.ConcurrencyAlgorithm.YCON
|| Program.mhpUpdateCategory == UpdateCategory.LZINV) {
if (AbstractPhase.globalMHPStale) {
AbstractPhase.globalMHPStale = false;
AutomatedUpdater.reinitMHP();
}
} else {
BeginPhasePoint.stabilizeStaleBeginPhasePoints();
}
if (analysisHandle.stateIsInvalid()) {
analysisHandle.markStateToBeValid();
AutomatedUpdater.reinitIDFA(analysisHandle);
}
} else {
assert (Program.idfaUpdateCategory == UpdateCategory.LZUPD);
if (analysisHandle.stateIsInvalid()) {
if (Program.concurrencyAlgorithm == Program.ConcurrencyAlgorithm.YCON
|| Program.mhpUpdateCategory == UpdateCategory.LZINV) {
if (AbstractPhase.globalMHPStale) {
AbstractPhase.globalMHPStale = false;
AutomatedUpdater.reinitMHP();
}
} else {
BeginPhasePoint.stabilizeStaleBeginPhasePoints();
}
analysisHandle.markStateToBeValid();
if (analysisName == AnalysisName.POINTSTO) {
Program.basePointsTo = false;
Program.memoizeAccesses++;
analysisHandle.restartAnalysisFromStoredNodes();
Program.memoizeAccesses--;
} else {
analysisHandle.restartAnalysisFromStoredNodes();
}
}
}
}
if (flowFactsOUT == null) {
flowFactsOUT = new HashMap<>();
}
return flowFactsOUT.get(analysisName);
}
public FlowFact getStaleOUT(AnalysisName analysisName) {
if (flowFactsOUT == null) {
flowFactsOUT = new HashMap<>();
}
return flowFactsOUT.get(analysisName);
}
public void setOUT(AnalysisName analysisName, FlowFact flowFact) {
if (flowFactsOUT == null) {
flowFactsOUT = new HashMap<>();
}
FlowFact oldOUT = flowFactsOUT.get(analysisName);
if (oldOUT == flowFact) {
return;
}
if (analysisName == AnalysisName.POINTSTO) {
if (oldOUT == null || !oldOUT.isEqualTo(flowFact)) {
this.invalidateAccessLists();
}
}
flowFactsOUT.put(analysisName, flowFact);
return;
}
public CellSet getImpactedSet() {
if (impactedSet == null) {
this.impactedSet = new CellSet();
}
return this.impactedSet;
}
public boolean hasAccessedCellSetsFor(CellularDataFlowAnalysis<?> analysis) {
if (this.accessedCellSets == null) {
return false;
}
return this.accessedCellSets.keySet().contains(analysis);
}
public boolean hasReadCellSetsFor(CellularDataFlowAnalysis<?> analysis) {
if (this.readCellSets == null) {
return false;
}
return this.readCellSets.keySet().contains(analysis);
}
public boolean hasWrittenCellSetsFor(CellularDataFlowAnalysis<?> analysis) {
if (this.writtenCellSets == null) {
return false;
}
return this.writtenCellSets.keySet().contains(analysis);
}
public CellSet getAccessedCellSets(AnalysisName analysisName) {
if (this.accessedCellSets == null) {
return new CellSet();
}
for (CellularDataFlowAnalysis<?> analysis : this.accessedCellSets.keySet()) {
if (analysis.getAnalysisName().equals(analysisName)) {
return this.accessedCellSets.get(analysis);
}
}
return new CellSet();
}
public CellSet getReadCellSets(AnalysisName analysisName) {
if (this.readCellSets == null) {
return new CellSet();
}
for (CellularDataFlowAnalysis<?> analysis : this.readCellSets.keySet()) {
if (analysis.getAnalysisName().equals(analysisName)) {
return this.readCellSets.get(analysis);
}
}
return new CellSet();
}
public CellSet getWrittenCellSets(AnalysisName analysisName) {
if (this.writtenCellSets == null) {
return new CellSet();
}
for (CellularDataFlowAnalysis<?> analysis : this.writtenCellSets.keySet()) {
if (analysis.getAnalysisName().equals(analysisName)) {
return this.writtenCellSets.get(analysis);
}
}
return new CellSet();
}
public Set<CellularDataFlowAnalysis<?>> getAnalysesWithAccessedCells() {
Set<CellularDataFlowAnalysis<?>> retSet = new HashSet<>();
if (this.accessedCellSets == null) {
return retSet;
}
return this.accessedCellSets.keySet();
}
public Set<CellularDataFlowAnalysis<?>> getAnalysesWithReadCells() {
Set<CellularDataFlowAnalysis<?>> retSet = new HashSet<>();
if (this.readCellSets == null) {
return retSet;
}
return this.readCellSets.keySet();
}
public Set<CellularDataFlowAnalysis<?>> getAnalysesWithWrittenCells() {
Set<CellularDataFlowAnalysis<?>> retSet = new HashSet<>();
if (this.writtenCellSets == null) {
return retSet;
}
return this.writtenCellSets.keySet();
}
public CellSet getAccessedCellSets(CellularDataFlowAnalysis<?> analysis) {
if (accessedCellSets == null) {
accessedCellSets = new HashMap<>();
}
CellSet retSet = accessedCellSets.get(analysis);
if (retSet == null) {
retSet = new CellSet();
accessedCellSets.put(analysis, retSet);
}
return retSet;
}
public CellSet getReadCellSets(CellularDataFlowAnalysis<?> analysis) {
if (readCellSets == null) {
readCellSets = new HashMap<>();
}
CellSet retSet = readCellSets.get(analysis);
if (retSet == null) {
retSet = new CellSet();
readCellSets.put(analysis, retSet);
}
return retSet;
}
public CellSet getWrittenCellSets(CellularDataFlowAnalysis<?> analysis) {
if (writtenCellSets == null) {
writtenCellSets = new HashMap<>();
}
CellSet retSet = writtenCellSets.get(analysis);
if (retSet == null) {
retSet = new CellSet();
writtenCellSets.put(analysis, retSet);
}
return retSet;
}
public void clearAccessedCellSets(CellularDataFlowAnalysis<?> analysis) {
if (accessedCellSets == null) {
return;
}
CellSet retSet = accessedCellSets.get(analysis);
if (retSet == null) {
return;
}
retSet.clear();
return;
}
public void clearReadCellSets(CellularDataFlowAnalysis<?> analysis) {
if (readCellSets == null) {
return;
}
CellSet retSet = readCellSets.get(analysis);
if (retSet == null) {
return;
}
retSet.clear();
return;
}
public void clearWrittenCellSets(CellularDataFlowAnalysis<?> analysis) {
if (writtenCellSets == null) {
return;
}
CellSet retSet = writtenCellSets.get(analysis);
if (retSet == null) {
return;
}
retSet.clear();
return;
}
public void removeAnalysisInformation(AnalysisName analysisName) {
if (flowFactsIN != null) {
flowFactsIN.remove(analysisName);
}
if (flowFactsOUT != null) {
flowFactsOUT.remove(analysisName);
}
}
public void removeAllAnalysisInformation() {
if (flowFactsIN != null) {
flowFactsIN.clear();
}
if (flowFactsOUT != null) {
flowFactsOUT.clear();
}
}
public NodeInfo(Node owner) {
setNode(owner);
setIdNumber(getNode().hashCode()); 
if (Misc.isCFGLeafNode(owner)) {
deprecated_flowFactsIN = new HashMap<>();
deprecated_flowFactsOUT = new HashMap<>();
deprecated_parallelFlowFactsIN = new HashMap<>();
deprecated_parallelFlowFactsOUT = new HashMap<>();
}
}
public void setCopySourceNode(Node copySourceNode) {
this.copySourceNode = copySourceNode;
}
@Override
public Object clone() throws CloneNotSupportedException {
return super.clone();
}
public IncompleteSemantics getIncompleteSemantics() {
if (incomplete == null) {
incomplete = new IncompleteSemantics(getNode());
}
return incomplete;
}
public CFGInfo getCFGInfo() {
if (cfgInfo == null) {
if (copySourceNode != null) {
this.cfgInfo = copySourceNode.getInfo().getCFGInfo().getCopy(this.getNode());
} else {
this.cfgInfo = new CFGInfo(getNode());
}
}
return cfgInfo;
}
public NodePhaseInfo getNodePhaseInfo() {
if (Misc.isCFGNonLeafNode(this.getNode())) {
BeginNode beginNode = this.getCFGInfo().getNestedCFG().getBegin();
return beginNode.getInfo().getNodePhaseInfo();
}
if (phaseInfo == null) {
phaseInfo = new NodePhaseInfo(getNode());
}
return phaseInfo;
}
public NodePhaseInfo readNodePhaseInfo() {
return phaseInfo;
}
public List<ParallelConstruct> getRegionInfo() {
if (regionInfo == null) {
regionInfo = new ArrayList<>();
}
return regionInfo;
}
public List<ParallelConstruct> readRegionInfo() {
return regionInfo;
}
public boolean isRunnableInRegion(ParallelConstruct par) {
if (regionInfo == null) {
return false;
}
for (ParallelConstruct present : regionInfo) {
if (present == par) {
return true;
}
}
return false;
}
public DataSharingAttribute getSharingAttribute(Cell cell) {
assert (cell != null);
if (cell instanceof HeapCell) {
return DataSharingAttribute.SHARED;
} else if (cell instanceof FreeVariable) {
return DataSharingAttribute.SHARED;
} else if (cell instanceof AddressCell) {
return this.getSharingAttribute(((AddressCell) cell).getPointedElement());
} else if (cell instanceof FieldCell) {
return this.getSharingAttribute(((FieldCell) cell).getAggregateElement());
} else if (!(cell instanceof Symbol)) {
if (cell == Cell.getNullCell()) {
return DataSharingAttribute.PRIVATE; 
}
return DataSharingAttribute.SHARED;
}
Symbol sym = (Symbol) cell;
if (sym == Cell.genericCell) {
return DataSharingAttribute.SHARED;
}
if (Program.getRoot().getInfo().getThreadPrivateList().values().contains(sym)) {
return DataSharingAttribute.THREADPRIVATE;
}
Node encloser = getNode();
while (encloser != null) {
if (encloser instanceof CompoundStatement) {
CompoundStatementInfo compStmtInfo = ((CompoundStatementInfo) encloser.getInfo());
HashMap<String, Symbol> symbolMap = compStmtInfo.getSymbolTable();
if (symbolMap.containsKey(sym.getName())) {
if (sym.isStatic()) {
return DataSharingAttribute.SHARED;
} else {
return DataSharingAttribute.PRIVATE;
}
}
} else if (encloser instanceof FunctionDefinition) {
FunctionDefinitionInfo funcDefInfo = ((FunctionDefinitionInfo) encloser.getInfo());
HashMap<String, Symbol> symbolMap = funcDefInfo.getSymbolTable();
if (symbolMap.containsKey(sym.getName())) {
if (sym.isStatic()) {
return DataSharingAttribute.SHARED;
} else {
return DataSharingAttribute.PRIVATE;
}
}
} else if (encloser instanceof TranslationUnit) {
RootInfo rootInfo = (RootInfo) encloser.getInfo();
HashMap<String, Symbol> symbolMap = rootInfo.getSymbolTable();
if (symbolMap.containsKey(sym.getName())) {
return DataSharingAttribute.SHARED;
} else {
return DataSharingAttribute.SHARED;
}
} else if (encloser instanceof OmpConstruct) {
if (encloser instanceof ForConstruct) {
NodeToken loopIteratorToken = ((ForConstruct) encloser).getF2().getF2().getF0();
Cell iteratorCell = Misc.getSymbolOrFreeEntry(loopIteratorToken);
if (!(iteratorCell instanceof Symbol)) {
return DataSharingAttribute.SHARED;
} else {
Symbol iteratorSymbol = (Symbol) iteratorCell;
if (iteratorSymbol == sym) {
return DataSharingAttribute.PRIVATE;
}
}
} else if (encloser instanceof ParallelForConstruct) {
assert (false);
} else if (encloser instanceof ParallelConstruct || encloser instanceof TaskConstruct) {
List<OmpClause> ompClauseList = ((OmpConstructInfo) encloser.getInfo()).getOmpClauseList();
for (OmpClause clause : ompClauseList) {
if (clause instanceof OmpSharedClause) {
OmpSharedClause sharedClause = (OmpSharedClause) clause;
VariableList varList = sharedClause.getF2();
Cell cellVar = Misc.getSymbolOrFreeEntry(varList.getF0());
if (!(cellVar instanceof Symbol)) {
return DataSharingAttribute.SHARED;
} else {
Symbol symVar = (Symbol) cellVar;
if (symVar == sym) {
return DataSharingAttribute.SHARED;
}
}
for (Node nodeChoice : varList.getF1().getNodes()) {
cellVar = Misc.getSymbolOrFreeEntry(
((NodeToken) ((NodeSequence) nodeChoice).getNodes().get(1)));
if (!(cellVar instanceof Symbol)) {
return DataSharingAttribute.SHARED;
} else {
Symbol symVar = (Symbol) cellVar;
if (symVar == sym) {
return DataSharingAttribute.SHARED;
}
}
}
} else if (clause instanceof OmpFirstPrivateClause) {
OmpFirstPrivateClause privateClause = (OmpFirstPrivateClause) clause;
VariableList varList = privateClause.getF2();
Cell cellVar = Misc.getSymbolOrFreeEntry(varList.getF0());
if (!(cellVar instanceof Symbol)) {
return DataSharingAttribute.SHARED;
} else {
Symbol symVar = (Symbol) cellVar;
if (symVar == sym) {
return DataSharingAttribute.PRIVATE;
}
}
for (Node nodeChoice : varList.getF1().getNodes()) {
cellVar = Misc.getSymbolOrFreeEntry(
(NodeToken) ((NodeSequence) nodeChoice).getNodes().get(1));
if (!(cellVar instanceof Symbol)) {
return DataSharingAttribute.SHARED;
} else {
Symbol symVar = (Symbol) cellVar;
if (symVar == sym) {
return DataSharingAttribute.PRIVATE;
}
}
}
} else if (clause instanceof OmpLastPrivateClause) {
OmpLastPrivateClause privateClause = (OmpLastPrivateClause) clause;
VariableList varList = privateClause.getF2();
Cell cellVar = Misc.getSymbolOrFreeEntry(varList.getF0());
if (!(cellVar instanceof Symbol)) {
return DataSharingAttribute.SHARED;
} else {
Symbol symVar = (Symbol) cellVar;
if (symVar == sym) {
return DataSharingAttribute.PRIVATE;
}
}
for (Node nodeChoice : varList.getF1().getNodes()) {
cellVar = Misc.getSymbolOrFreeEntry(
(NodeToken) ((NodeSequence) nodeChoice).getNodes().get(1));
if (!(cellVar instanceof Symbol)) {
return DataSharingAttribute.SHARED;
} else {
Symbol symVar = (Symbol) cellVar;
if (symVar == sym) {
return DataSharingAttribute.PRIVATE;
}
}
}
} else if (clause instanceof OmpPrivateClause) {
OmpPrivateClause privateClause = (OmpPrivateClause) clause;
VariableList varList = privateClause.getF2();
Cell cellVar = Misc.getSymbolOrFreeEntry(varList.getF0());
if (!(cellVar instanceof Symbol)) {
return DataSharingAttribute.SHARED;
} else {
Symbol symVar = (Symbol) cellVar;
if (symVar == sym) {
return DataSharingAttribute.PRIVATE;
}
}
for (Node nodeChoice : varList.getF1().getNodes()) {
cellVar = Misc.getSymbolOrFreeEntry(
(NodeToken) ((NodeSequence) nodeChoice).getNodes().get(1));
if (!(cellVar instanceof Symbol)) {
return DataSharingAttribute.SHARED;
} else {
Symbol symVar = (Symbol) cellVar;
if (symVar == sym) {
return DataSharingAttribute.PRIVATE;
}
}
}
}
}
for (OmpClause clause : ompClauseList) {
if (clause instanceof OmpDfltSharedClause) {
return DataSharingAttribute.SHARED;
}
}
if (encloser instanceof ParallelConstruct) {
return DataSharingAttribute.SHARED;
}
}
}
encloser = encloser.getParent();
}
return DataSharingAttribute.SHARED;
}
public boolean mayRelyOnPtsTo() {
long timer = System.nanoTime();
Node node = this.getNode();
if (Misc.isCFGLeafNode(node)) {
if (this.reliesOnPtsTo == TRISTATE.UNKNOWN) {
this.reliesOnPtsTo = CellAccessGetter.mayRelyOnPointsTo(node) ? TRISTATE.YES : TRISTATE.NO;
}
DriverModule.mayRelyPTATimer += System.nanoTime() - timer;
return this.reliesOnPtsTo == TRISTATE.YES;
} else {
boolean val = CellAccessGetter.mayRelyOnPointsTo(node);
DriverModule.mayRelyPTATimer += System.nanoTime() - timer;
return val;
}
}
private boolean mayRelyOnPtsToForSymbols() {
Node node = this.getNode();
if (Misc.isCFGLeafNode(node)) {
if (this.reliesOnPtsTo == TRISTATE.UNKNOWN) {
this.reliesOnPtsTo = CellAccessGetter.mayRelyOnPointsToForSymbols(node) ? TRISTATE.YES : TRISTATE.NO;
}
return this.reliesOnPtsTo == TRISTATE.YES;
} else {
return CellAccessGetter.mayRelyOnPointsToForSymbols(node);
}
}
public CellList getReads() {
if (Misc.isCFGLeafNode(this.getNode())) {
if (this.readList == null) {
readList = CellAccessGetter.getReads(this.getNode());
} else {
boolean mayRelyOnPointsTo = this.mayRelyOnPtsTo();
if (mayRelyOnPointsTo) {
if (pointsToMayBeStale() || Program.memoizeAccesses > 0) {
readList = CellAccessGetter.getReads(this.getNode());
}
}
}
return this.readList;
} else if (Misc.isCFGNonLeafNode(this.getNode())) {
CellList nonLeafReadList = new CellList();
for (Node leafContent : node.getInfo().getCFGInfo().getIntraTaskCFGLeafContents()) {
CellList leafReadList = leafContent.getInfo().getReads();
nonLeafReadList.addAll(leafReadList);
}
return nonLeafReadList;
} else {
return CellAccessGetter.getReads(this.getNode());
}
}
public CellList getWrites() {
if (Misc.isCFGLeafNode(this.getNode())) {
if (this.writeList == null) {
writeList = CellAccessGetter.getWrites(this.getNode());
} else {
boolean mayRelyOnPointsTo = this.mayRelyOnPtsTo();
if (mayRelyOnPointsTo) {
if (pointsToMayBeStale() || Program.memoizeAccesses > 0) {
writeList = CellAccessGetter.getWrites(this.getNode());
}
}
}
return this.writeList;
} else if (Misc.isCFGNonLeafNode(this.getNode())) {
CellList nonLeafWriteList = new CellList();
for (Node leafContent : node.getInfo().getCFGInfo().getIntraTaskCFGLeafContents()) {
CellList leafWriteList = leafContent.getInfo().getWrites();
nonLeafWriteList.addAll(leafWriteList);
}
return nonLeafWriteList;
} else {
return CellAccessGetter.getWrites(this.getNode());
}
}
public boolean mayWrite() {
return CellAccessGetter.mayWrite(this.getNode());
}
public boolean mayUpdatePointsTo() {
return CellAccessGetter.mayUpdatePointsTo(this.getNode());
}
public CellSet getSharedReads() {
if (Misc.isCFGLeafNode(this.getNode())) {
if (this.sharedReadSet == null) {
sharedReadSet = CellAccessGetter.getSharedReads(this.getNode());
} else {
boolean mayRelyOnPointsTo = this.mayRelyOnPtsTo();
if (mayRelyOnPointsTo) {
if (pointsToMayBeStale() || Program.memoizeAccesses > 0) {
sharedReadSet = CellAccessGetter.getSharedReads(this.getNode());
}
}
}
return this.sharedReadSet;
} else if (Misc.isCFGNonLeafNode(this.getNode())) {
CellSet nonLeafSharedReadList = new CellSet();
for (Node leafContent : node.getInfo().getCFGInfo().getIntraTaskCFGLeafContents()) {
CellSet leafSharedReadList = leafContent.getInfo().getSharedReads();
nonLeafSharedReadList.addAll(leafSharedReadList);
}
return nonLeafSharedReadList;
} else {
return CellAccessGetter.getSharedReads(this.getNode());
}
}
public CellSet getSharedWrites() {
if (Misc.isCFGLeafNode(this.getNode())) {
if (this.sharedWriteSet == null) {
sharedWriteSet = CellAccessGetter.getSharedWrites(this.getNode());
} else {
boolean mayRelyOnPointsTo = this.mayRelyOnPtsTo();
if (mayRelyOnPointsTo) {
if (pointsToMayBeStale() || Program.memoizeAccesses > 0) {
sharedWriteSet = CellAccessGetter.getSharedWrites(this.getNode());
}
}
}
return this.sharedWriteSet;
} else if (Misc.isCFGNonLeafNode(this.getNode())) {
CellSet nonLeafSharedWriteList = new CellSet();
for (Node leafContent : node.getInfo().getCFGInfo().getIntraTaskCFGLeafContents()) {
CellSet leafSharedWriteList = leafContent.getInfo().getSharedWrites();
nonLeafSharedWriteList.addAll(leafSharedWriteList);
}
return nonLeafSharedWriteList;
} else {
return CellAccessGetter.getSharedWrites(this.getNode());
}
}
public CellSet getSymbolReads() {
CellList cellList = this.readList;
if (Misc.isCFGLeafNode(this.getNode())) {
if (cellList == null) {
cellList = CellAccessGetter.getSymbolReads(this.getNode());
} else {
boolean mayRelyOnPointsTo = this.mayRelyOnPtsToForSymbols();
if (mayRelyOnPointsTo) {
if (pointsToMayBeStale() || Program.memoizeAccesses > 0) {
cellList = CellAccessGetter.getSymbolReads(this.getNode());
}
}
}
} else if (Misc.isCFGNonLeafNode(this.getNode())) {
cellList = new CellList();
for (Node leafContent : node.getInfo().getCFGInfo().getIntraTaskCFGLeafContents()) {
CellSet leafSymbolReadList = leafContent.getInfo().getSymbolReads();
cellList.addAll(leafSymbolReadList);
}
} else {
cellList = CellAccessGetter.getSymbolReads(this.getNode());
}
CellSet symbolSet = new CellSet();
for (Cell cell : cellList) {
if (cell instanceof Symbol) {
symbolSet.add(cell);
}
}
return symbolSet;
}
public CellSet getSymbolWrites() {
CellList cellList = this.writeList;
if (Misc.isCFGLeafNode(this.getNode())) {
if (cellList == null) {
cellList = CellAccessGetter.getSymbolWrites(this.getNode());
} else {
boolean mayRelyOnPointsTo = this.mayRelyOnPtsToForSymbols();
if (mayRelyOnPointsTo) {
if (pointsToMayBeStale() || Program.memoizeAccesses > 0) {
cellList = CellAccessGetter.getSymbolWrites(this.getNode());
}
}
}
} else if (Misc.isCFGNonLeafNode(this.getNode())) {
cellList = new CellList();
for (Node leafContent : node.getInfo().getCFGInfo().getIntraTaskCFGLeafContents()) {
CellSet leafSymbolWriteList = leafContent.getInfo().getSymbolWrites();
cellList.addAll(leafSymbolWriteList);
}
} else {
cellList = CellAccessGetter.getSymbolWrites(this.getNode());
}
CellSet symbolSet = new CellSet();
for (Cell cell : cellList) {
if (cell instanceof Symbol) {
symbolSet.add(cell);
}
}
return symbolSet;
}
public CellSet getSymbolAccesses() {
CellSet retSet = new CellSet();
retSet.addAll(this.getSymbolReads());
retSet.addAll(this.getSymbolWrites());
return retSet;
}
public CellSet getNonFieldSharedReads() {
if (Misc.isCFGLeafNode(this.getNode())) {
boolean update = false;
if (this.nonFieldReadSet == null) {
update = true;
} else {
boolean mayRelyOnPointsTo = this.mayRelyOnPtsTo();
if (mayRelyOnPointsTo) {
if (pointsToMayBeStale() || Program.memoizeAccesses > 0) {
update = true;
}
}
}
if (update) {
CellSet reads = this.getSharedReads();
if (reads.isUniversal()) {
this.nonFieldReadSet = reads;
} else if (!reads.getReadOnlyInternal().stream()
.anyMatch(c -> c instanceof FieldCell || c instanceof HeapCell)) {
this.nonFieldReadSet = reads;
} else {
this.nonFieldReadSet = new CellSet();
for (Cell cell : reads) {
if (cell instanceof FieldCell || cell instanceof HeapCell) {
continue;
}
this.nonFieldReadSet.add(cell);
}
}
}
return this.nonFieldReadSet;
} else {
CellSet reads = this.getSharedReads();
CellSet newSet;
if (reads.isUniversal()) {
newSet = reads;
} else if (!reads.getReadOnlyInternal().stream()
.anyMatch(c -> c instanceof FieldCell || c instanceof HeapCell)) {
newSet = reads;
} else {
newSet = new CellSet();
for (Cell cell : reads) {
if (cell instanceof FieldCell || cell instanceof HeapCell) {
continue;
}
newSet.add(cell);
}
}
return newSet;
}
}
public CellSet getNonFieldSharedWrites() {
if (Misc.isCFGLeafNode(this.getNode())) {
boolean update = false;
if (this.nonFieldWriteSet == null) {
update = true;
} else {
boolean mayRelyOnPointsTo = this.mayRelyOnPtsTo();
if (mayRelyOnPointsTo) {
if (pointsToMayBeStale() || Program.memoizeAccesses > 0) {
update = true;
}
}
}
if (update) {
CellSet writes = this.getSharedWrites();
if (writes.isUniversal()) {
this.nonFieldWriteSet = writes;
} else if (!writes.getReadOnlyInternal().stream()
.anyMatch(c -> c instanceof FieldCell || c instanceof HeapCell)) {
this.nonFieldWriteSet = writes;
} else {
this.nonFieldWriteSet = new CellSet();
for (Cell cell : writes) {
if (cell instanceof FieldCell || cell instanceof HeapCell) {
continue;
}
this.nonFieldWriteSet.add(cell);
}
}
}
return this.nonFieldWriteSet;
} else {
CellSet writes = this.getSharedWrites();
CellSet newSet;
if (writes.isUniversal()) {
newSet = writes;
} else if (!writes.getReadOnlyInternal().stream()
.anyMatch(c -> c instanceof FieldCell || c instanceof HeapCell)) {
newSet = writes;
} else {
newSet = new CellSet();
for (Cell cell : writes) {
if (cell instanceof FieldCell || cell instanceof HeapCell) {
continue;
}
newSet.add(cell);
}
}
return newSet;
}
}
public void invalidateAccessLists() {
this.readList = null;
this.writeList = null;
this.sharedReadSet = null;
this.sharedWriteSet = null;
this.nonFieldReadSet = null;
this.nonFieldWriteSet = null;
}
public CellSet getAccesses() {
CellSet accessSet = new CellSet();
accessSet.addAll(this.getReads());
accessSet.addAll(this.getWrites());
return accessSet;
}
public List<SyntacticAccessExpression> getBaseAccessExpressionReads() {
Node node = this.getNode();
if (node instanceof Expression || Misc.isCFGLeafNode(node)) {
if (baseAccessExpressionReads == null) {
baseAccessExpressionReads = SyntacticAccessExpressionGetter.getBaseAccessExpressionReads(node);
}
return baseAccessExpressionReads;
} else {
return SyntacticAccessExpressionGetter.getBaseAccessExpressionReads(node);
}
}
public List<SyntacticAccessExpression> getBaseAccessExpressionWrites() {
Node node = this.getNode();
if (node instanceof Expression || Misc.isCFGLeafNode(node)) {
if (baseAccessExpressionWrites == null) {
baseAccessExpressionWrites = SyntacticAccessExpressionGetter.getBaseAccessExpressionWrites(node);
}
return baseAccessExpressionWrites;
} else {
return SyntacticAccessExpressionGetter.getBaseAccessExpressionWrites(node);
}
}
public List<CastExpression> getPointerDereferencedExpressionReads() {
Node node = this.getNode();
if (node instanceof Expression || Misc.isCFGLeafNode(node)) {
if (pointerDereferencedExpressionReads == null) {
pointerDereferencedExpressionReads = PointerDereferenceGetter
.getPointerDereferencedCastExpressionReads(node);
}
return pointerDereferencedExpressionReads;
} else {
return PointerDereferenceGetter.getPointerDereferencedCastExpressionReads(node);
}
}
public List<CastExpression> getPointerDereferencedExpressionWrites() {
Node node = this.getNode();
if (node instanceof Expression || Misc.isCFGLeafNode(node)) {
if (pointerDereferencedExpressionWrites == null) {
pointerDereferencedExpressionWrites = PointerDereferenceGetter
.getPointerDereferencedCastExpressionWrites(node);
}
return pointerDereferencedExpressionWrites;
} else {
return PointerDereferenceGetter.getPointerDereferencedCastExpressionWrites(node);
}
}
public CellSet getInternalAccesses() {
CellSet internalAccessSet = new CellSet();
internalAccessSet.addAll(this.getReads());
internalAccessSet.addAll(this.getWrites());
return internalAccessSet;
}
public CellSet getUsedCells() {
return UsedCellsGetter.getUsedCells(this.getNode());
}
public Set<Type> getUsedTypes() {
Node baseNode = this.getNode();
Set<Type> usedTypes = new HashSet<>();
for (Scopeable scopeNode : baseNode.getInfo().getLexicallyEnclosedScopesInclusive()) {
if (scopeNode instanceof TranslationUnit) {
TranslationUnit scope = (TranslationUnit) scopeNode;
for (Symbol sym : scope.getInfo().getSymbolTable().values()) {
Type symType = sym.getType();
if (symType instanceof StructType || symType instanceof UnionType || symType instanceof EnumType) {
usedTypes.addAll(symType.getAllTypes());
}
}
for (Typedef tDef : scope.getInfo().getTypedefTable().values()) {
Type tDefType = tDef.getType();
if (tDefType instanceof StructType || tDefType instanceof UnionType
|| tDefType instanceof EnumType) {
usedTypes.addAll(tDefType.getAllTypes());
}
}
} else if (scopeNode instanceof FunctionDefinition) {
FunctionDefinition scope = (FunctionDefinition) scopeNode;
for (Symbol sym : scope.getInfo().getSymbolTable().values()) {
Type symType = sym.getType();
if (symType instanceof StructType || symType instanceof UnionType || symType instanceof EnumType) {
usedTypes.addAll(symType.getAllTypes());
}
}
usedTypes.addAll(scope.getInfo().getReturnType().getAllTypes());
} else if (scopeNode instanceof CompoundStatement) {
CompoundStatement scope = (CompoundStatement) scopeNode;
for (Symbol sym : scope.getInfo().getSymbolTable().values()) {
Type symType = sym.getType();
if (symType instanceof StructType || symType instanceof UnionType || symType instanceof EnumType) {
usedTypes.addAll(symType.getAllTypes());
}
}
for (Typedef tDef : scope.getInfo().getTypedefTable().values()) {
Type tDefType = tDef.getType();
if (tDefType instanceof StructType || tDefType instanceof UnionType
|| tDefType instanceof EnumType) {
usedTypes.addAll(tDefType.getAllTypes());
}
}
}
}
UsedTypeGetter utg = new UsedTypeGetter();
this.getNode().accept(utg);
usedTypes.addAll(utg.usedTypes);
return usedTypes;
}
public Set<Typedef> getUsedTypedefs() {
UsedTypedefGetter utg = new UsedTypedefGetter();
this.getNode().accept(utg);
return utg.usedTypedefs;
}
public void makeSymbolsFreeInRWList(Set<String> freeNames) {
Node node = this.getNode();
if (Misc.isCFGLeafNode(node)) {
this.writeList.replaceAll((x) -> {
if (x instanceof Symbol) {
Symbol sym = (Symbol) x;
if (freeNames.contains(sym.getName())) {
return new FreeVariable(sym.getName());
} else {
return x;
}
} else {
return x;
}
});
this.readList.replaceAll((x) -> {
if (x instanceof Symbol) {
Symbol sym = (Symbol) x;
if (freeNames.contains(sym.getName())) {
return new FreeVariable(sym.getName());
} else {
return x;
}
} else {
return x;
}
});
} else {
Misc.warnDueToLackOfFeature("Handling free variables in the read/write lists for non-leaf/non-CFG nodes.",
this.getNode());
return;
}
}
private TRISTATE hasShared = TRISTATE.UNKNOWN;
public boolean hasSharedAccesses() {
if (this.hasShared == TRISTATE.UNKNOWN) {
boolean val = !this.getSharedReads().isEmpty() || !this.getSharedWrites().isEmpty();
this.hasShared = val ? TRISTATE.YES : TRISTATE.NO;
}
return this.hasShared == TRISTATE.YES;
}
public boolean mayInterfereWith(CellCollection reads, CellCollection writes) {
for (Cell c : this.getSharedReads()) {
if (writes.contains(c)) {
return true;
}
}
for (Cell c : this.getSharedWrites()) {
if (reads.contains(c)) {
return true;
}
if (writes.contains(c)) {
return true;
}
}
return false;
}
public CellSet getSharedAccesses() {
return Misc.setUnion(this.getSharedReads(), this.getSharedWrites());
}
public boolean isUpdateNode() {
return Misc.doIntersect(this.getReads(), this.getWrites());
}
public CellSet getAllCellsAtNodeExclusively() {
if (this.getNode() instanceof Scopeable) {
Node encloser = this.getNode().getParent();
if (encloser == null) {
return new CellSet();
}
return encloser.getInfo().getAllCellsAtNode();
} else {
return this.getAllCellsAtNode();
}
}
public Set<String> getAllSymbolNamesAtNodeExclusively() {
Set<String> cellNames = new HashSet<>();
CellSet cellSet = this.getAllCellsAtNodeExclusively();
if (cellSet.isUniversal()) {
return cellNames;
} else {
cellSet.applyAllExpanded(c -> {
if (c instanceof Symbol) {
Symbol sym = (Symbol) c;
cellNames.add(sym.getName());
}
});
}
return cellNames;
}
public CellSet getAllCellsAtNode() {
Node encloser = (Node) Misc.getEnclosingBlock(this.getNode());
if (encloser == null) {
return new CellSet();
}
return encloser.getInfo().getAllCellsAtNode();
}
public Set<String> getAllSymbolNamesAtNodes() {
Set<String> cellNames = new HashSet<>();
CellSet cellSet = this.getAllCellsAtNode();
if (cellSet.isUniversal()) {
return cellNames;
} else {
for (Cell cell : cellSet) {
if (cell instanceof Symbol) {
Symbol sym = (Symbol) cell;
cellNames.add(sym.getName());
}
}
}
return cellNames;
}
public CellSet getSharedCellsAtNode() {
Node encloser = (Node) Misc.getEnclosingBlock(this.getNode());
if (encloser == null) {
return new CellSet();
}
return encloser.getInfo().getSharedCellsAtNode();
}
public List<Typedef> getAllTypedefListAtNode() {
Node encloser = (Node) Misc.getEnclosingBlock(this.getNode());
if (encloser == null) {
return new ArrayList<>();
}
return encloser.getInfo().getAllTypedefListAtNode();
}
public Set<String> getAllTypedefNameListAtNode() {
Set<String> typedefNameList = new HashSet<>();
for (Typedef td : this.getAllTypedefListAtNode()) {
typedefNameList.add(td.getTypedefName());
}
return typedefNameList;
}
public List<Type> getAllTypeListAtNode() {
Node encloser = (Node) Misc.getEnclosingBlock(this.getNode());
if (encloser == null) {
return new ArrayList<>();
}
return encloser.getInfo().getAllTypeListAtNode();
}
public Set<String> getAllTypeNameListAtNode() {
List<Type> typeAtStmt = this.getAllTypeListAtNode();
Set<String> typeNameAtStmt = new HashSet<>();
for (Type t : typeAtStmt) {
if (t instanceof StructType) {
typeNameAtStmt.add(((StructType) t).getName());
} else if (t instanceof UnionType) {
typeNameAtStmt.add(((UnionType) t).getTag());
} else if (t instanceof EnumType) {
typeNameAtStmt.add(((EnumType) t).getTag());
}
}
return typeNameAtStmt;
}
public boolean hasBarrierInAST() {
BarrierGetter barGetter = new BarrierGetter();
getNode().accept(barGetter);
if (barGetter.barrierList.size() != 0) {
return true;
}
return false;
}
public boolean hasBarrierInCFG() {
if (this.hasBarrierInAST()) {
return true;
}
for (CallStatement cS : this.getLexicallyEnclosedCallStatements()) {
for (FunctionDefinition funcDef : cS.getInfo().getCalledDefinitions()) {
Set<FunctionDefinition> inProcessFDSet = new HashSet<>();
inProcessFDSet.add(funcDef);
if (funcDef.getInfo().hasBarrierInCFGVisited(inProcessFDSet)) {
return true;
}
}
}
return false;
}
protected boolean hasBarrierInCFGVisited(Set<FunctionDefinition> inProcessFDSet) {
if (this.hasBarrierInAST()) {
return true;
}
for (CallStatement cS : this.getLexicallyEnclosedCallStatements()) {
for (FunctionDefinition funcDef : cS.getInfo().getCalledDefinitions()) {
if (inProcessFDSet.contains(funcDef)) {
continue;
}
inProcessFDSet.add(funcDef);
if (funcDef.getInfo().hasBarrierInCFGVisited(inProcessFDSet)) {
return true;
}
}
}
return false;
}
public boolean removeLexicallyEnclosedCallStatement(CallStatement callStmt) {
Node node = this.getNode();
if (!Misc.isCFGNode(node)) {
return false;
}
if (callStatements == null) {
return false;
}
return callStatements.removeIf(c -> c == callStmt);
}
public List<CallStatement> getLexicallyEnclosedCallStatements() {
Node node = this.getNode();
long thisTimer = System.nanoTime();
if (Misc.isCFGNode(node)) {
if (callStatements == null) {
callStatements = new LinkedList<>();
for (CallStatement expNode : Misc.getInheritedEnclosee(getNode(), CallStatement.class)) {
callStatements.add(expNode);
}
}
NodeInfo.callQueryTimer += (System.nanoTime() - thisTimer);
return callStatements;
} else if (node instanceof TranslationUnit) {
List<CallStatement> nonCachedCalls = new LinkedList<>();
RootInfo rootInfo = (RootInfo) node.getInfo();
for (FunctionDefinition func : rootInfo.getAllFunctionDefinitions()) {
nonCachedCalls.addAll(func.getInfo().getLexicallyEnclosedCallStatements());
}
NodeInfo.callQueryTimer += (System.nanoTime() - thisTimer);
return nonCachedCalls;
} else {
List<CallStatement> nonCachedCalls = new LinkedList<>();
for (Node cfgContent : node.getInfo().getCFGInfo().getLexicalCFGContents()) {
if (cfgContent instanceof CallStatement) {
nonCachedCalls.add((CallStatement) cfgContent);
}
}
NodeInfo.callQueryTimer += (System.nanoTime() - thisTimer);
return nonCachedCalls;
}
}
public boolean hasLexicallyEnclosedCallStatements() {
this.getLexicallyEnclosedCallStatements();
if (callStatements.size() > 0) {
return true;
} else {
return false;
}
}
public Set<FunctionDefinition> getReachableCallGraphNodes() {
Set<FunctionDefinition> returnSet = new HashSet<>();
Node baseNode = this.getNode();
if (baseNode instanceof FunctionDefinition) {
returnSet.add((FunctionDefinition) baseNode);
}
Set<FunctionDefinition> endSet = new HashSet<>();
Set<FunctionDefinition> startSet = this.getCallGraphSuccessors();
returnSet.addAll(CollectorVisitor.collectNodeSetInGenericGraph(startSet, endSet, (f) -> false,
(f) -> f.getInfo().getCallGraphSuccessors()));
returnSet.addAll(startSet);
returnSet.addAll(endSet);
return returnSet;
}
public Set<FunctionDefinition> getCallGraphSuccessors() {
Set<FunctionDefinition> calledFDs = new HashSet<>();
for (CallStatement cs : this.getLexicallyEnclosedCallStatements()) {
calledFDs.addAll(cs.getInfo().getCalledDefinitions());
}
return calledFDs;
}
private Set<CallStatement> reachableCallSites;
private int localCounter = -1;
public Set<CallStatement> getReachableCallStatementsInclusive() {
if (reachableCallSites == null || localCounter != AutomatedUpdater.reachableCounter) {
Set<CallStatement> callStmts = new HashSet<>();
Set<CallStatement> startSet = new HashSet<>(this.getLexicallyEnclosedCallStatements());
Set<CallStatement> endSet = new HashSet<>();
callStmts.addAll(CollectorVisitor.collectNodeSetInGenericGraph(startSet, endSet, (cs) -> false, (cs) -> {
Set<CallStatement> nextSet = new HashSet<>();
for (FunctionDefinition fd : cs.getInfo().getCalledDefinitions()) {
nextSet.addAll(fd.getInfo().getLexicallyEnclosedCallStatements());
}
return nextSet;
}));
callStmts.addAll(startSet);
callStmts.addAll(endSet);
this.reachableCallSites = callStmts;
localCounter = AutomatedUpdater.reachableCounter;
}
return reachableCallSites;
}
public Node getOuterMostEncloser() {
List<Node> outerList = (CollectorVisitor.collectNodeListInGenericGraph(this.node, null, Objects::isNull,
(n) -> {
List<Node> neighbourSet = new LinkedList<>();
neighbourSet.add(n.getParent());
return neighbourSet;
}));
Node node = outerList.get(outerList.size() - 1);
return node;
}
public Node getOuterMostNonLeafEncloser() {
List<Node> nonLeafEnclosingPathExclusive = this.node.getInfo().getNonLeafNestingPathExclusive();
if (nonLeafEnclosingPathExclusive.isEmpty()) {
return null;
} else {
return nonLeafEnclosingPathExclusive.get(nonLeafEnclosingPathExclusive.size() - 1);
}
}
public Scopeable getOuterMostScopeExclusive() {
List<Node> nonLeafEnclosingPathExclusive = this.node.getInfo().getNonLeafNestingPathExclusive();
nonLeafEnclosingPathExclusive = Misc.reverseList(nonLeafEnclosingPathExclusive);
for (Node n : nonLeafEnclosingPathExclusive) {
if (n instanceof Scopeable) {
return (Scopeable) n;
}
}
return null;
}
public Set<Node> getAllLexicalNonLeafEnclosersExclusive() {
Set<Node> allEnclosers = CollectorVisitor.collectNodeSetInGenericGraph(this.node, null, (n) -> (n == null),
(n) -> {
Set<Node> neighbourSet = new HashSet<>();
Node encloser = Misc.getEnclosingCFGNonLeafNode(n);
if (encloser != null) {
neighbourSet.add(encloser);
}
return neighbourSet;
});
return allEnclosers;
}
public List<Node> getNonLeafNestingPathExclusive() {
List<Node> nonLeafEnclosingPathExclusive = new LinkedList<>();
nonLeafEnclosingPathExclusive = CollectorVisitor.collectNodeListInGenericGraph(this.node, null,
(n) -> (n == null), (n) -> {
List<Node> neighbourSet = new LinkedList<>();
neighbourSet.add(Misc.getEnclosingCFGNonLeafNode(n));
return neighbourSet;
});
return nonLeafEnclosingPathExclusive;
}
public Set<Node> getCFGNestingNonLeafNodes() {
Set<Node> allEnclosers = CollectorVisitor.collectNodeSetInGenericGraph(this.node, null, (n) -> (n == null),
(n) -> {
Set<Node> neighbourSet = new HashSet<>();
if (n instanceof FunctionDefinition) {
FunctionDefinition func = (FunctionDefinition) n;
for (CallStatement callSt : func.getInfo().getCallersOfThis()) {
neighbourSet.add(callSt);
Node encloser = Misc.getEnclosingCFGNonLeafNode(callSt);
if (encloser != null) {
neighbourSet.add(encloser);
}
}
} else {
Node encloser = Misc.getEnclosingCFGNonLeafNode(n);
if (encloser != null) {
neighbourSet.add(encloser);
}
}
return neighbourSet;
});
return allEnclosers;
}
public boolean isControlConfined() {
Set<Node> cfgContents = this.getCFGInfo().getLexicalCFGContents();
for (Object labeledObject : cfgContents.stream()
.filter(n -> (n instanceof Statement && ((Statement) n).getInfo().hasLabelAnnotations())).toArray()) {
Statement labeledStmt = (Statement) labeledObject;
for (Node pred : labeledStmt.getInfo().getCFGInfo().getPredecessors()) {
if (!cfgContents.contains(pred)) {
return false;
}
}
if (labeledStmt.getInfo().getIncompleteSemantics().hasIncompleteEdges()) {
return false;
}
}
for (Node constituent : cfgContents) {
if (constituent instanceof JumpStatement) {
for (Node succ : constituent.getInfo().getCFGInfo().getSuccessors()) {
if (!cfgContents.contains(succ)) {
return false;
}
}
if (constituent.getInfo().getIncompleteSemantics().hasIncompleteEdges()) {
return false;
}
}
}
return true;
}
public Set<Node> getInJumpSources() {
Set<Node> inJumpSources = new HashSet<>();
Set<Node> cfgContents = this.getCFGInfo().getLexicalCFGContents();
for (Object labeledObject : cfgContents.stream()
.filter(n -> (n instanceof Statement && ((Statement) n).getInfo().hasLabelAnnotations())).toArray()) {
Statement labeledStmt = (Statement) labeledObject;
for (Node pred : labeledStmt.getInfo().getCFGInfo().getPredecessors()) {
if (!cfgContents.contains(pred)) {
inJumpSources.add(pred);
}
}
}
return inJumpSources;
}
public Set<Node> getInJumpDestinations() {
Set<Node> inJumpDestinations = new HashSet<>();
Set<Node> cfgContents = this.getCFGInfo().getLexicalCFGContents();
for (Node ijs : this.getInJumpSources()) {
for (Node succ : ijs.getInfo().getCFGInfo().getInterProceduralLeafSuccessors()) {
if (cfgContents.contains(succ)) {
inJumpDestinations.add(succ);
}
}
}
return inJumpDestinations;
}
public Set<Node> getOutJumpDestinations() {
Set<Node> outJumpDestinations = new HashSet<>();
Set<Node> cfgContents = this.getCFGInfo().getLexicalCFGContents();
for (Node constituent : cfgContents) {
if (constituent instanceof JumpStatement) {
for (Node succ : constituent.getInfo().getCFGInfo().getSuccessors()) {
if (!cfgContents.contains(succ)) {
outJumpDestinations.add(succ);
}
}
}
}
return outJumpDestinations;
}
public Set<Node> getOutJumpSources() {
Set<Node> outJumpSources = new HashSet<>();
Set<Node> cfgContents = this.getCFGInfo().getLexicalCFGContents();
for (Node ojd : this.getOutJumpDestinations()) {
for (Node pred : ojd.getInfo().getCFGInfo().getInterProceduralLeafPredecessors()) {
if (cfgContents.contains(pred)) {
outJumpSources.add(pred);
}
}
}
return outJumpSources;
}
public Set<Node> getEntryNodes() {
Set<Node> entryNodes = new HashSet<>();
if (Misc.isCFGLeafNode(this.getNode())) {
entryNodes.add(this.getNode());
} else if (Misc.isCFGNonLeafNode(this.getNode())) {
entryNodes.add(this.getCFGInfo().getNestedCFG().getBegin());
entryNodes.addAll(this.getInJumpDestinations());
} else {
}
return entryNodes;
}
public Set<Node> getExitNodes() {
Set<Node> exitNodes = new HashSet<>();
if (Misc.isCFGLeafNode(this.getNode())) {
exitNodes.add(this.getNode());
} else if (Misc.isCFGNonLeafNode(this.getNode())) {
exitNodes.add(this.getCFGInfo().getNestedCFG().getEnd());
exitNodes.addAll(this.getOutJumpSources());
} else {
}
return exitNodes;
}
public void removeExtraScopes() {
CompoundStatementNormalizer.removeExtraScopes(this.getNode());
}
public void printNode() {
System.out.println(this.getString());
}
public String getString(List<Commentor> commentorList) {
return StringGetter.getString(this.getNode(), commentorList);
}
public String getString() {
return this.getString(new ArrayList<>());
}
public String getDebugString() {
List<Commentor> commentors = new ArrayList<>();
commentors.add((n) -> {
String tempStr = "";
FlowFact flow;
flow = n.getInfo().getCurrentOUT(AnalysisName.POINTSTO);
if (n instanceof BeginNode) {
tempStr += "[BeginNode of " + n.getParent().getClass().getSimpleName() + "]";
} else if (n instanceof EndNode) {
tempStr += "[EndNode of " + n.getParent().getClass().getSimpleName() + "]";
} else if (n instanceof ParameterDeclaration) {
tempStr += "[Parameter: " + n + "]";
} else if (n instanceof PreCallNode) {
tempStr += "[PRECALL NODE] ";
} else if (n instanceof PostCallNode) {
tempStr += "[POSTCALL NODE] ";
}
if (flow != null) {
tempStr += "OUT: " + flow.getString();
}
return tempStr;
});
Node node = this.getNode();
if (!Misc.isCFGNode(node)) {
return this.getString(commentors);
}
Node outerMostNonLeafEncloser = node.getInfo().getOuterMostNonLeafEncloser();
if (outerMostNonLeafEncloser == null) {
return this.getString(commentors);
} else {
int size = this.getComments().size();
this.getComments().add("################################################");
String tempStr = outerMostNonLeafEncloser.getInfo().getString(commentors);
this.getComments().remove(size);
return tempStr;
}
}
public Statement getStatementWithLabel(String l) {
for (Node tempNode : this.getCFGInfo().getLexicalCFGContents()) {
if (tempNode instanceof Statement) {
for (Label tempLabel : ((Statement) tempNode).getInfo().getLabelAnnotations()) {
if (tempLabel instanceof SimpleLabel) {
if (((SimpleLabel) tempLabel).getLabelName().equals(l)) {
return (Statement) tempNode;
}
}
}
}
}
return null;
}
public CellMap<Cell> getCopyMap() {
CellMap<Cell> returnSet = new CellMap<>();
FunctionDefinition mainFunc = Program.getRoot().getInfo().getMainFunction();
if (mainFunc == null) {
Misc.warnDueToLackOfFeature("Cannot run copy propagation pass on a program without main().", null);
return returnSet;
}
if (!FlowAnalysis.getAllAnalyses().keySet().contains(AnalysisName.COPYPROPAGATION)) {
System.err.println("Pass: Copy propagation analysis.");
long timeStart = System.nanoTime();
CopyPropagationAnalysis cpa = new CopyPropagationAnalysis();
cpa.run(mainFunc);
long timeTaken = System.nanoTime() - timeStart;
System.err.println("\tNodes processed " + cpa.nodesProcessed + " times.");
System.err.println("\tTime taken: " + timeTaken / 1000000000.0 + "s.");
}
CopyPropagationFlowMap cpf = (CopyPropagationFlowMap) this.getIN(AnalysisName.COPYPROPAGATION);
if (cpf == null) {
return returnSet;
}
returnSet = new ExtensibleCellMap<>(cpf.getFlowMap());
return returnSet;
}
public CellMap<Cell> getRelevantCopyMap() {
CellMap<Cell> retMap = new CellMap<>();
CellMap<Cell> fullMap = this.getCopyMap();
CellSet relevantSymbols = this.getAllCellsAtNode();
for (Cell key : fullMap.nonGenericKeySet()) {
if (!relevantSymbols.contains(key)) {
continue;
}
Cell rhs = fullMap.get(key);
if (key == rhs) {
continue;
}
if (rhs == Cell.genericCell || !relevantSymbols.contains(rhs)) {
continue;
}
retMap.put(key, rhs);
}
return retMap;
}
public Set<Definition> getReachingDefinitions(Cell cell) {
Set<Definition> returnSet = new HashSet<>();
ReachingDefinitionFlowMap rdf = (ReachingDefinitionFlowMap) this.getIN(AnalysisName.REACHING_DEFINITION);
if (rdf == null) {
return returnSet;
}
return rdf.getFlowMap().get(cell);
}
public Set<Definition> getReachingDefinitions() {
Set<Definition> reachingDefinitions = new HashSet<>();
ReachingDefinitionFlowMap rdf = (ReachingDefinitionFlowMap) this.getIN(AnalysisName.REACHING_DEFINITION);
CellSet cellsHere = this.getAllCellsAtNode();
if (rdf == null) {
return reachingDefinitions;
}
for (Cell key : rdf.getFlowMap().nonGenericKeySet()) {
if (!cellsHere.contains(key)) {
continue;
}
reachingDefinitions.addAll(rdf.getFlowMap().get(key));
}
if (rdf.getFlowMap().isUniversal()) {
reachingDefinitions = new HashSet<>(reachingDefinitions);
reachingDefinitions.addAll(rdf.getFlowMap().get(Cell.genericCell));
}
return reachingDefinitions;
}
private static boolean printedVal = false;
public Set<Node> getDominators() {
if (!printedVal) {
printedVal = true;
DumpSnapshot.dumpRoot("beforeDom");
}
DominatorFlowFact da = (DominatorFlowFact) this.getIN(AnalysisName.DOMINANCE);
if (da == null) {
return new HashSet<>();
}
if (da.dominators == null) {
return new HashSet<>();
}
Set<Node> retSet = new HashSet<>(da.dominators);
retSet.add(this.getNode());
return retSet;
}
public Set<Node> getPhaseSensitiveReachingDefinitions(Phase ph, Set<Node> phaseInsensitiveReachingDefinitions) {
return phaseInsensitiveReachingDefinitions;
}
public Set<Node> getFlowSources() {
Set<Node> retSet = new HashSet<>();
DataDependenceForwardFF ff = (DataDependenceForwardFF) this.getIN(AnalysisName.DATA_DEPENDENCE_FORWARD);
if (ff == null || ff.writeSources == null) {
return retSet;
}
this.getReads().applyAllExpanded((sym) -> {
NodeSet nodeSet = ff.writeSources.get(sym);
if (nodeSet != null) {
Set<Node> nodesForSym = nodeSet.getReadOnlySet();
if (nodesForSym != null) {
retSet.addAll(nodesForSym);
}
}
});
return retSet;
}
public Set<Node> getFlowDestinations() {
Set<Node> retSet = new HashSet<>();
CellMap<NodeSet> fDMap = this.getReadDestinations();
this.getWrites().applyAllExpanded(sym -> {
NodeSet nodeSet = fDMap.get(sym);
if (nodeSet != null) {
Set<Node> nodesForSym = nodeSet.getReadOnlySet();
if (nodesForSym != null) {
retSet.addAll(nodesForSym);
}
}
});
return retSet;
}
public Set<Node> getAntiSources() {
Set<Node> retSet = new HashSet<>();
if (Misc.doIntersect(this.getReads(), this.getWrites())) {
retSet.add(this.getNode());
}
DataDependenceForwardFF ff = (DataDependenceForwardFF) this.getIN(AnalysisName.DATA_DEPENDENCE_FORWARD);
if (ff == null || ff.readSources == null) {
return retSet;
}
this.getWrites().applyAllExpanded(sym -> {
NodeSet nodeSet = ff.readSources.get(sym);
if (nodeSet != null) {
Set<Node> nodesForSym = nodeSet.getReadOnlySet();
if (nodesForSym != null) {
retSet.addAll(nodesForSym);
}
}
});
return retSet;
}
public Set<Node> getAntiDestinations() {
Set<Node> retSet = new HashSet<>();
CellMap<NodeSet> aDMap = this.getWriteDestinations();
this.getReads().applyAllExpanded(sym -> {
NodeSet nodeSet = aDMap.get(sym);
if (nodeSet != null) {
Set<Node> nodesForSym = nodeSet.getReadOnlySet();
if (nodesForSym != null) {
retSet.addAll(nodesForSym);
}
}
});
return retSet;
}
public Set<Node> getOutputSources() {
Set<Node> retSet = new HashSet<>();
DataDependenceForwardFF ff = (DataDependenceForwardFF) this.getIN(AnalysisName.DATA_DEPENDENCE_FORWARD);
if (ff == null || ff.writeSources == null) {
return retSet;
}
this.getWrites().applyAllExpanded(sym -> {
NodeSet nodeSet = ff.writeSources.get(sym);
if (nodeSet != null) {
Set<Node> nodesForSym = nodeSet.getReadOnlySet();
if (nodesForSym != null) {
retSet.addAll(nodesForSym);
}
}
});
return retSet;
}
public Set<Node> getOutputDestinations() {
Set<Node> retSet = new HashSet<>();
CellMap<NodeSet> oDMap = this.getWriteDestinations();
this.getWrites().applyAllExpanded(sym -> {
NodeSet nodeSet = oDMap.get(sym);
if (nodeSet != null) {
Set<Node> nodesForSym = nodeSet.getReadOnlySet();
if (nodesForSym != null) {
retSet.addAll(nodesForSym);
}
}
});
return retSet;
}
public CellMap<NodeSet> getReadDestinations() {
if (!NodeInfo.ddfDone) {
NodeInfo.ddfDone = true;
performDDF();
}
if (!NodeInfo.readWriteDestinationsSet) {
NodeInfo.setReadWriteDestinations();
}
if (this.readDestinations == null) {
CellMap<NodeSet> newCellMap = new CellMap<>();
this.readDestinations = newCellMap;
}
return readDestinations;
}
public CellMap<NodeSet> getWriteDestinations() {
if (!NodeInfo.ddfDone) {
NodeInfo.ddfDone = true;
performDDF();
}
if (!NodeInfo.readWriteDestinationsSet) {
NodeInfo.setReadWriteDestinations();
}
if (this.writeDestinations == null) {
CellMap<NodeSet> newCellMap = new CellMap<>();
this.writeDestinations = newCellMap;
}
return writeDestinations;
}
public static void setReadWriteDestinations() {
NodeInfo.readWriteDestinationsSet = true;
for (Node n1 : Program.getRoot().getInfo().getAllLeafNodesInTheProgram()) {
DataDependenceForwardFF n1FF = (DataDependenceForwardFF) n1.getInfo()
.getIN(AnalysisName.DATA_DEPENDENCE_FORWARD);
if (n1FF == null) {
continue;
}
CellList n1Reads = n1.getInfo().getReads();
if (n1Reads.isUniversal()) {
if (n1FF.writeSources != null) {
if (n1FF.writeSources.isUniversal()) {
for (Node n2 : n1FF.writeSources.get(Cell.genericCell).getReadOnlySet()) {
if (n2.getInfo().getReadDestinations().isUniversal()) {
NodeInfo.addNodeToMap(n2.getInfo().getReadDestinations(), Cell.genericCell, n1);
} else {
for (Cell cn2 : n2.getInfo().getReadDestinations().keySetExpanded()) {
NodeInfo.addNodeToMap(n2.getInfo().getReadDestinations(), cn2, n1);
}
}
}
} else {
for (Cell c : n1FF.writeSources.keySetExpanded()) {
for (Node n2 : n1FF.writeSources.get(c).getReadOnlySet()) {
NodeInfo.addNodeToMap(n2.getInfo().getReadDestinations(), c, n1);
}
}
}
}
} else {
for (Cell read : n1Reads) {
if (n1FF.writeSources != null && n1FF.writeSources.containsKey(read)) {
for (Node n2 : n1FF.writeSources.get(read).getReadOnlySet()) {
NodeInfo.addNodeToMap(n2.getInfo().getReadDestinations(), read, n1);
}
}
}
}
CellList n1Writes = n1.getInfo().getWrites();
if (n1Writes.isUniversal()) {
if (n1FF.readSources != null) {
if (n1FF.readSources.isUniversal()) {
for (Node n2 : n1FF.readSources.get(Cell.genericCell).getReadOnlySet()) {
if (n2.getInfo().getWriteDestinations().isUniversal()) {
NodeInfo.addNodeToMap(n2.getInfo().getWriteDestinations(), Cell.genericCell, n1);
} else {
for (Cell cn2 : n2.getInfo().getWriteDestinations().keySetExpanded()) {
NodeInfo.addNodeToMap(n2.getInfo().getWriteDestinations(), cn2, n1);
}
}
}
} else {
for (Cell c : n1FF.readSources.keySetExpanded()) {
for (Node n2 : n1FF.readSources.get(c).getReadOnlySet()) {
NodeInfo.addNodeToMap(n2.getInfo().getWriteDestinations(), c, n1);
}
}
}
}
if (n1FF.writeSources != null) {
if (n1FF.writeSources.isUniversal()) {
for (Node n2 : n1FF.writeSources.get(Cell.genericCell).getReadOnlySet()) {
if (n2.getInfo().getWriteDestinations().isUniversal()) {
NodeInfo.addNodeToMap(n2.getInfo().getWriteDestinations(), Cell.genericCell, n1);
} else {
for (Cell cn2 : n2.getInfo().getWriteDestinations().keySetExpanded()) {
NodeInfo.addNodeToMap(n2.getInfo().getWriteDestinations(), cn2, n1);
}
}
}
} else {
for (Cell c : n1FF.writeSources.keySetExpanded()) {
for (Node n2 : n1FF.writeSources.get(c).getReadOnlySet()) {
NodeInfo.addNodeToMap(n2.getInfo().getWriteDestinations(), c, n1);
}
}
}
}
} else {
for (Cell write : n1Writes) {
if (n1FF.readSources != null && n1FF.readSources.containsKey(write)) {
for (Node n2 : n1FF.readSources.get(write).getReadOnlySet()) {
NodeInfo.addNodeToMap(n2.getInfo().getWriteDestinations(), write, n1);
}
}
if (n1FF.writeSources != null && n1FF.writeSources.containsKey(write)) {
for (Node n2 : n1FF.writeSources.get(write).getReadOnlySet()) {
NodeInfo.addNodeToMap(n2.getInfo().getWriteDestinations(), write, n1);
}
}
}
}
}
}
public Set<Node> getFirstPossibleKillersForwardExclusively(CellSet atStake) {
final Set<Node> nodeSet = new HashSet<>();
Node leafNode = this.getNode();
NodeWithStack startPoint = new NodeWithStack(leafNode, new CallStack());
CollectorVisitor.collectNodeSetInGenericGraph(startPoint, null, (n) -> {
CellSet writesOfN = new CellSet(n.getNode().getInfo().getWrites());
if (Misc.doIntersect(writesOfN, atStake)) {
nodeSet.add(n.getNode());
return true;
} else {
return false;
}
}, (n) -> n.getNode().getInfo().getCFGInfo().getInterTaskLeafSuccessorNodesForVariables(n.getCallStack(),
atStake, Program.sveSensitive));
return nodeSet;
}
public Set<Node> getAllUsesForwardsExclusively(Cell w) {
Node leafNode = this.getNode();
assert (Misc.isCFGLeafNode(leafNode));
CellSet focusSet = new CellSet();
focusSet.add(w);
NodeWithStack startPoint = new NodeWithStack(leafNode, new CallStack());
Set<Node> readSet = new HashSet<>();
CollectorVisitor.collectNodeSetInGenericGraph(startPoint, null, (n) -> {
CellSet readsOfN = new CellSet(n.getNode().getInfo().getReads());
if (Misc.doIntersect(readsOfN, focusSet)) {
readSet.add(n.getNode());
}
CellSet writesOfN = new CellSet(n.getNode().getInfo().getWrites());
if (writesOfN.size() == 1 && writesOfN.contains(w)) {
return true;
} else {
return false;
}
}, n -> n.getNode().getInfo().getCFGInfo().getInterTaskLeafSuccessorNodesForVariables(n.getCallStack(),
focusSet, Program.sveSensitive));
return readSet;
}
public boolean isCellLiveOut(Cell w) {
return this.hasUsesForwardsExclusively(w);
}
public boolean hasUsesForwardsExclusively(Cell w) {
Node leafNode = this.getNode();
assert (Misc.isCFGLeafNode(leafNode));
CellSet focusSet = new CellSet();
focusSet.add(w);
NodeWithStack startPoint = new NodeWithStack(leafNode, new CallStack());
Set<Node> readSet = new HashSet<>();
CollectorVisitor.collectNodeSetInGenericGraph(startPoint, null, (n) -> {
CellSet readsOfN = new CellSet(n.getNode().getInfo().getReads());
if (Misc.doIntersect(readsOfN, focusSet)) {
readSet.add(n.getNode());
}
CellSet writesOfN = new CellSet(n.getNode().getInfo().getWrites());
if (writesOfN.size() == 1 && writesOfN.contains(w)) {
return true;
} else {
return false;
}
}, n -> n.getNode().getInfo().getCFGInfo().getInterTaskLeafSuccessorNodesForVariables(n.getCallStack(),
focusSet, Program.sveSensitive));
if (readSet.isEmpty()) {
return false;
}
return true;
}
public Set<Node> getAllUsesBackwardsExclusively(Cell w) {
Node leafNode = this.getNode();
assert (Misc.isCFGLeafNode(leafNode));
CellSet focusSet = new CellSet();
focusSet.add(w);
NodeWithStack startPoint = new NodeWithStack(leafNode, new CallStack());
Set<Node> readSet = new HashSet<>();
CollectorVisitor.collectNodeSetInGenericGraph(startPoint, null, (n) -> {
CellSet writesOfN = new CellSet(n.getNode().getInfo().getWrites());
if (writesOfN.size() == 1 && writesOfN.contains(w)) {
return true;
} else {
CellSet readsOfN = new CellSet(n.getNode().getInfo().getReads());
if (Misc.doIntersect(readsOfN, focusSet)) {
readSet.add(n.getNode());
}
return false;
}
}, n -> n.getNode().getInfo().getCFGInfo().getInterTaskLeafPredecessorNodesForVariables(n.getCallStack(),
focusSet, Program.sveSensitive));
return readSet;
}
public boolean hasUsesBackwardsExclusively(Cell w) {
Node leafNode = this.getNode();
assert (Misc.isCFGLeafNode(leafNode));
CellSet focusSet = new CellSet();
focusSet.add(w);
NodeWithStack startPoint = new NodeWithStack(leafNode, new CallStack());
Set<Node> readSet = new HashSet<>();
CollectorVisitor.collectNodeSetInGenericGraph(startPoint, null, (n) -> {
CellSet writesOfN = new CellSet(n.getNode().getInfo().getWrites());
if (writesOfN.size() == 1 && writesOfN.contains(w)) {
return true;
} else {
CellSet readsOfN = new CellSet(n.getNode().getInfo().getReads());
if (Misc.doIntersect(readsOfN, focusSet)) {
readSet.add(n.getNode());
}
return false;
}
}, n -> n.getNode().getInfo().getCFGInfo().getInterTaskLeafPredecessorNodesForVariables(n.getCallStack(),
focusSet, Program.sveSensitive));
return !readSet.isEmpty();
}
private static void addNodeToMap(CellMap<NodeSet> map, Cell cell, Node node) {
NodeSet nS = map.get(cell);
if (nS != null) {
nS.addNode(node);
return;
} else {
Set<Node> nodeSet = new HashSet<>();
nodeSet.add(node);
nS = new NodeSet(nodeSet);
map.put(cell, nS);
}
}
public List<PragmaImop> getPragmaAnnotations() {
if (!Misc.isCFGNode(this.node)) {
return null;
}
if (this.pragmaAnnotations == null) {
this.pragmaAnnotations = new ArrayList<>();
}
return this.pragmaAnnotations;
}
public boolean isSCOPPed(Set<Node> nodeSet) {
Node node = this.getNode();
Node encloser = null;
if (node instanceof Expression) {
Expression exp = (Expression) this.getNode();
if (!Misc.isAPredicate(exp)) {
return false;
} else {
encloser = Misc.getEnclosingCFGNonLeafNode(exp);
}
} else if (node instanceof OmpForCondition) {
encloser = Misc.getEnclosingCFGNonLeafNode(node);
} else {
assert (false);
return false;
}
if (encloser == null) {
return false;
}
if (!encloser.getInfo().isControlConfined()) {
return false;
}
Set<Node> contents = encloser.getInfo().getCFGInfo().getIntraTaskCFGLeafContents();
Set<Node> leafContents = new HashSet<>();
for (Node n : nodeSet) {
leafContents.addAll(n.getInfo().getCFGInfo().getIntraTaskCFGLeafContents());
}
if (Misc.doIntersect(contents, leafContents)) {
return false;
}
return true;
}
public boolean isSCOPPed(Node tempNode) {
Set<Node> tempSet = new HashSet<>();
tempSet.add(tempNode);
return this.isSCOPPed(tempSet);
}
public Set<Scopeable> getLexicallyEnclosedScopesInclusive() {
Set<Scopeable> retSet = new HashSet<>();
Set<Class<? extends Node>> classSet = new HashSet<>();
classSet.add(TranslationUnit.class);
classSet.add(FunctionDefinition.class);
classSet.add(CompoundStatement.class);
for (Node scopeNode : Misc.getExactEnclosee(this.getNode(), classSet)) {
retSet.add((Scopeable) scopeNode);
}
return retSet;
}
public void removeUnusedElements() {
removeUnusedVariables();
boolean changed;
do {
boolean internal;
changed = false;
do {
internal = false;
internal |= this.removeUnusedTypedefs();
changed |= internal;
} while (internal);
do {
internal = false;
internal |= this.removeUnusedTypes();
changed |= internal;
} while (internal);
} while (changed);
}
public void removeUnusedVariables() {
Node baseNode = getNode();
CellSet usedCells = baseNode.getInfo().getUsedCells();
assert (!usedCells.isUniversal());
System.err.print("\tDeleting the declarations for the following unused variables: ");
boolean deletedAny = false;
for (Scopeable scopeNode : baseNode.getInfo().getLexicallyEnclosedScopesInclusive()) {
if (scopeNode instanceof TranslationUnit) {
TranslationUnit scope = (TranslationUnit) scopeNode;
Collection<Symbol> symbolSet = new HashSet<>(scope.getInfo().getSymbolTable().values());
for (Symbol sym : symbolSet) {
if (!usedCells.contains(sym)) {
Node declaringNode = sym.getDeclaringNode();
if (declaringNode instanceof Declaration) {
ElementsOfTranslation elemOfTranslation = Misc.getEnclosingNode(declaringNode,
ElementsOfTranslation.class);
scope.getF0().removeNode(elemOfTranslation);
scope.getInfo().removeDeclarationEffects((Declaration) declaringNode);
System.err.print(sym.getName() + "; ");
}
}
}
} else if (scopeNode instanceof FunctionDefinition) {
continue;
} else if (scopeNode instanceof CompoundStatement) {
CompoundStatement scope = (CompoundStatement) scopeNode;
Collection<Symbol> symbolSet = new HashSet<>(scope.getInfo().getSymbolTable().values());
for (Symbol sym : symbolSet) {
if (!usedCells.contains(sym)) {
Declaration decl = (Declaration) sym.getDeclaringNode();
Initializer init = decl.getInfo().getInitializer();
if (init != null) {
Statement stmt = FrontEnd.parseAlone(init.toString() + ";", Statement.class);
int index = scope.getInfo().getCFGInfo().getElementList().indexOf(decl);
scope.getInfo().getCFGInfo().addStatement(index, stmt);
}
scope.getInfo().getCFGInfo().removeDeclaration(decl);
System.err.print(sym.getName() + "; ");
}
}
}
}
if (!deletedAny) {
System.err.println("<none>");
} else {
System.err.println();
}
}
public boolean removeUnusedTypes() {
Node baseNode = getNode();
boolean changed = false;
Set<Type> usedTypes = baseNode.getInfo().getUsedTypes();
System.err.print("\tDeleting the declarations for the following unused types: ");
for (Scopeable scopeNode : baseNode.getInfo().getLexicallyEnclosedScopesInclusive()) {
if (scopeNode instanceof TranslationUnit) {
TranslationUnit scope = (TranslationUnit) scopeNode;
Collection<Type> typeSet = new HashSet<>(scope.getInfo().getTypeTable().values());
for (Type type : typeSet) {
Declaration declaringNode = null;
if (type instanceof StructType) {
declaringNode = ((StructType) type).getDeclaringNode();
} else if (type instanceof UnionType) {
declaringNode = ((UnionType) type).getDeclaringNode();
} else if (type instanceof EnumType) {
declaringNode = ((EnumType) type).getDeclaringNode();
}
if (declaringNode != null) {
if (type.isComplete()) {
if (!usedTypes.contains(type)) {
ElementsOfTranslation elemOfTranslation = Misc.getEnclosingNode(declaringNode,
ElementsOfTranslation.class);
scope.getF0().removeNode(elemOfTranslation);
scope.getInfo().removeTypeDeclarationEffects(declaringNode);
System.err.print(type);
changed = true;
}
}
}
}
} else if (scopeNode instanceof FunctionDefinition) {
continue;
} else if (scopeNode instanceof CompoundStatement) {
CompoundStatement scope = (CompoundStatement) scopeNode;
Collection<Type> typeSet = new HashSet<>(scope.getInfo().getTypeTable().values());
for (Type type : typeSet) {
Declaration declaringNode = null;
if (type instanceof StructType) {
declaringNode = ((StructType) type).getDeclaringNode();
} else if (type instanceof UnionType) {
declaringNode = ((UnionType) type).getDeclaringNode();
} else if (type instanceof EnumType) {
declaringNode = ((EnumType) type).getDeclaringNode();
}
if (declaringNode != null) {
if (type.isComplete()) {
if (!usedTypes.contains(type)) {
scope.getInfo().getCFGInfo().removeDeclaration(declaringNode);
System.err.print(type);
changed = true;
}
}
}
}
}
}
if (changed) {
System.err.println();
} else {
System.err.println("<none>");
}
return changed;
}
public boolean removeUnusedTypedefs() {
Node baseNode = getNode();
boolean changed = false;
Set<Typedef> usedTypedefs = baseNode.getInfo().getUsedTypedefs();
System.err.print("\tDeleting the declarations for the following unused typedefs: ");
for (Scopeable scopeNode : baseNode.getInfo().getLexicallyEnclosedScopesInclusive()) {
if (scopeNode instanceof TranslationUnit) {
TranslationUnit scope = (TranslationUnit) scopeNode;
Collection<Typedef> typedefSet = new HashSet<>(scope.getInfo().getTypedefTable().values());
for (Typedef tDef : typedefSet) {
if (!usedTypedefs.contains(tDef)) {
Declaration definingNode = tDef.getDefiningNode();
ElementsOfTranslation elemOfTranslation = Misc.getEnclosingNode(definingNode,
ElementsOfTranslation.class);
scope.getF0().removeNode(elemOfTranslation);
scope.getInfo().removeDeclarationEffects(definingNode);
System.err.print(tDef.getTypedefName() + "; ");
changed = true;
}
}
} else if (scopeNode instanceof FunctionDefinition) {
continue;
} else if (scopeNode instanceof CompoundStatement) {
CompoundStatement scope = (CompoundStatement) scopeNode;
Collection<Typedef> typedefSet = new HashSet<>(scope.getInfo().getTypedefTable().values());
for (Typedef tDef : typedefSet) {
if (!usedTypedefs.contains(tDef)) {
Declaration decl = tDef.getDefiningNode();
scope.getInfo().getCFGInfo().removeDeclaration(decl);
System.err.print(tDef.getTypedefName() + "; ");
changed = true;
}
}
}
}
if (changed) {
System.err.println();
} else {
System.err.println("<none>");
}
return changed;
}
public List<Assignment> getLexicalAssignments() {
return AssignmentGetter.getLexicalAssignments(this.getNode());
}
public List<Assignment> getInterProceduralAssignments() {
return AssignmentGetter.getInterProceduralAssignments(this.getNode());
}
public String getNodeReplacedString(Node changeSource, String replacementString) {
return StringGetter.getNodeReplacedString(this.getNode(), changeSource, replacementString);
}
public List<String> getComments() {
if (this.getNode() instanceof BeginNode || this.getNode() instanceof EndNode) {
return this.getNode().getParent().getInfo().getComments();
}
if (comments == null) {
comments = new ArrayList<>();
}
return comments;
}
public static Commentor getDefaultCommentor() {
if (defaultCommentor == null) {
defaultCommentor = ((n) -> {
String comments = "";
for (String comment : n.getInfo().getComments()) {
comments += comment + "\n";
}
return comments;
});
}
return defaultCommentor;
}
public boolean isConnectedToProgram() {
Node encloser = Misc.getEnclosingNode(this.getNode(), TranslationUnit.class);
if (encloser == null || encloser != Program.getRoot()) {
return false;
}
return true;
}
public int getIdNumber() {
return idNumber;
}
public void setIdNumber(int idNumber) {
this.idNumber = idNumber;
}
public Node getNode() {
return node;
}
public void setNode(Node node) {
this.node = node;
}
public Cell getCleanWrite() {
Node n = getNode();
n = Misc.getCFGNodeFor(n);
if (Misc.isCFGNonLeafNode(n)) {
return null;
}
CellList writeList = n.getInfo().getWrites();
CellList readList = n.getInfo().getReads();
if (writeList.size() != 1) {
return null;
}
if (Misc.doIntersect(writeList, readList)) {
return null;
}
return writeList.get(0);
}
public boolean isCleanWrite() {
Node n = getNode();
n = Misc.getCFGNodeFor(n);
if (Misc.isCFGNonLeafNode(n)) {
return false;
}
CellList writeList = n.getInfo().getWrites();
CellList readList = n.getInfo().getReads();
if (writeList.size() != 1) {
return false;
}
if (Misc.doIntersect(writeList, readList)) {
return false;
}
return true;
}
public BTNode getBTNode() {
if (this.btNode == null) {
this.btNode = BarrierTreeConstructor.getBarrierTreeFor(this.getNode());
}
return this.btNode;
}
@Deprecated
private List<OldLock> lockSet;
@Deprecated
public void setInfo(NodeInfo oldInfo) {
this.setIdNumber(oldInfo.getIdNumber());
if (Misc.isCFGNode(oldInfo.getNode())) {
if (oldInfo.deprecated_flowFactsIN != null) {
for (AnalysisName analysis : oldInfo.deprecated_flowFactsIN.keySet()) {
this.setFlowFactIN(analysis, oldInfo.getFlowFactIN(analysis, null).getCopy());
}
}
if (oldInfo.deprecated_flowFactsOUT != null) {
for (AnalysisName analysis : oldInfo.deprecated_flowFactsOUT.keySet()) {
this.setFlowFactOUT(analysis, oldInfo.getFlowFactOUT(analysis, null).getCopy());
}
}
}
if (oldInfo.cfgInfo != null) {
this.cfgInfo = oldInfo.cfgInfo.getCopy(this.getNode());
}
if (phaseInfo != null) {
this.phaseInfo = oldInfo.phaseInfo.getCopy(this.getNode());
}
this.regionInfo = oldInfo.regionInfo;
this.reachingDefinitions = oldInfo.reachingDefinitions;
this.usesInDU = oldInfo.usesInDU;
this.defsInUD = oldInfo.defsInUD;
this.liveOut = oldInfo.liveOut;
this.flowEdgeDestList = oldInfo.flowEdgeDestList;
this.flowEdgeSrcList = oldInfo.flowEdgeSrcList;
this.antiEdgeDestList = oldInfo.antiEdgeDestList;
this.antiEdgeSrcList = oldInfo.antiEdgeSrcList;
this.outputEdgeDestList = oldInfo.outputEdgeDestList;
this.outputEdgeSrcList = oldInfo.outputEdgeSrcList;
this.callStatements = oldInfo.callStatements;
this.incomplete = oldInfo.incomplete;
this.lockSet = oldInfo.lockSet;
}
@Deprecated
public List<OldLock> getLockSet() {
if (lockSet != null) {
return lockSet;
}
lockSet = new ArrayList<>();
return lockSet;
}
@Deprecated
public boolean deprecated_isControlConfined() {
Set<Node> cfgContents = this.getCFGInfo().getLexicalCFGContents();
for (Node constituent : cfgContents) {
if (constituent instanceof JumpStatement) {
for (Node succ : constituent.getInfo().getCFGInfo().getSuccessors()) {
if (!cfgContents.contains(succ)) {
return false;
}
}
}
}
Set<LabeledStatement> astContents = Misc.getExactEnclosee(getNode(), LabeledStatement.class);
for (Node labeledStmt : astContents) {
Node cfgNode = Misc.getInternalFirstCFGNode(labeledStmt);
for (Node pred : cfgNode.getInfo().getCFGInfo().getPredBlocks()) {
if (!cfgContents.contains(pred)) {
return false;
}
}
}
return true;
}
@Deprecated
public List<Definition> deprecated_getDefinitionList() {
return null;
}
@Deprecated
private HashMap<AnalysisName, Deprecated_FlowFact> deprecated_flowFactsIN;
@Deprecated
private HashMap<AnalysisName, Deprecated_FlowFact> deprecated_flowFactsOUT;
@Deprecated
private HashMap<AnalysisName, Deprecated_FlowFact> deprecated_parallelFlowFactsIN;
@Deprecated
private HashMap<AnalysisName, Deprecated_FlowFact> deprecated_parallelFlowFactsOUT;
@Deprecated
private Set<Definition> reachingDefinitions;
@Deprecated
private CellMap<Set<Node>> usesInDU;
@Deprecated
private CellMap<Set<Node>> defsInUD;
@Deprecated
private HashMap<Node, CellSet> liveOut;
@Deprecated
private CellMap<Set<Node>> flowEdgeDestList;
@Deprecated
private CellMap<Set<Node>> flowEdgeSrcList;
@Deprecated
private CellMap<Set<Node>> antiEdgeDestList;
@Deprecated
private CellMap<Set<Node>> antiEdgeSrcList;
@Deprecated
private CellMap<Set<Node>> outputEdgeDestList;
@Deprecated
private CellMap<Set<Node>> outputEdgeSrcList;
@Deprecated
private List<CallSite> calledFunctions_old = null;
@Deprecated
public boolean hasParallelFlowFactIN(AnalysisName analysisName) {
if (deprecated_parallelFlowFactsIN.containsKey(analysisName)) {
return true;
} else {
return false;
}
}
@Deprecated
public boolean hasParalleleFlowFactOUT(AnalysisName analysisName) {
if (deprecated_parallelFlowFactsOUT.containsKey(analysisName)) {
return true;
} else {
return false;
}
}
@Deprecated
public boolean hasFlowFactIN(AnalysisName analysisName) {
if (deprecated_flowFactsIN.containsKey(analysisName)) {
return true;
} else {
return false;
}
}
@Deprecated
public boolean hasFlowFactOUT(AnalysisName analysisName) {
if (deprecated_flowFactsOUT.containsKey(analysisName)) {
return true;
} else {
return false;
}
}
@Deprecated
public Deprecated_FlowFact getParallelFlowFactIN(AnalysisName analysisName,
Deprecated_InterProceduralCFGPass<? extends Deprecated_FlowFact> pass) {
if (deprecated_parallelFlowFactsIN == null) {
deprecated_parallelFlowFactsIN = new HashMap<>();
}
if (!deprecated_parallelFlowFactsIN.containsKey(analysisName)) {
deprecated_parallelFlowFactsIN.put(analysisName, pass.getTop());
}
return deprecated_parallelFlowFactsIN.get(analysisName);
}
@Deprecated
public Deprecated_FlowFact getParallelFlowFactOUT(AnalysisName analysisName,
Deprecated_InterProceduralCFGPass<? extends Deprecated_FlowFact> pass) {
if (deprecated_parallelFlowFactsOUT == null) {
deprecated_parallelFlowFactsOUT = new HashMap<>();
}
if (!deprecated_parallelFlowFactsOUT.containsKey(analysisName)) {
deprecated_parallelFlowFactsOUT.put(analysisName, pass.getTop());
}
return deprecated_parallelFlowFactsOUT.get(analysisName);
}
@Deprecated
public Deprecated_FlowFact getFlowFactIN(AnalysisName analysisName,
Deprecated_InterProceduralCFGPass<? extends Deprecated_FlowFact> pass) {
if (deprecated_flowFactsIN == null) {
deprecated_flowFactsIN = new HashMap<>();
}
if (!deprecated_flowFactsIN.containsKey(analysisName)) {
if (this.copySourceNode != null) {
Deprecated_FlowFact copiedFlowFactIN = copySourceNode.getInfo().getFlowFactIN(analysisName, pass)
.getCopy();
this.deprecated_flowFactsIN.put(analysisName, copiedFlowFactIN);
} else {
deprecated_flowFactsIN.put(analysisName, pass.getTop());
}
}
return deprecated_flowFactsIN.get(analysisName);
}
@Deprecated
public Deprecated_FlowFact getFlowFactIN(AnalysisName analysisName) {
if (deprecated_flowFactsIN == null) {
deprecated_flowFactsIN = new HashMap<>();
}
if (!deprecated_flowFactsIN.containsKey(analysisName)) {
if (this.copySourceNode != null) {
Deprecated_FlowFact copiedFlowFactIN = copySourceNode.getInfo().getFlowFactIN(analysisName).getCopy();
this.deprecated_flowFactsIN.put(analysisName, copiedFlowFactIN);
}
}
return deprecated_flowFactsIN.get(analysisName);
}
@Deprecated
public Deprecated_FlowFact getFlowFactOUT(AnalysisName analysisName,
Deprecated_InterProceduralCFGPass<? extends Deprecated_FlowFact> pass) {
if (deprecated_flowFactsOUT == null) {
deprecated_flowFactsOUT = new HashMap<>();
}
if (!deprecated_flowFactsOUT.containsKey(analysisName)) {
if (this.copySourceNode != null) {
Deprecated_FlowFact copiedFlowFactOUT = copySourceNode.getInfo().getFlowFactOUT(analysisName, pass)
.getCopy();
this.deprecated_flowFactsOUT.put(analysisName, copiedFlowFactOUT);
} else {
deprecated_flowFactsOUT.put(analysisName, pass.getTop());
}
}
return deprecated_flowFactsOUT.get(analysisName);
}
@Deprecated
public Deprecated_FlowFact getFlowFactOUT(AnalysisName analysisName) {
if (deprecated_flowFactsOUT == null) {
deprecated_flowFactsOUT = new HashMap<>();
}
if (!deprecated_flowFactsOUT.containsKey(analysisName)) {
if (this.copySourceNode != null) {
Deprecated_FlowFact copiedFlowFactOUT = copySourceNode.getInfo().getFlowFactOUT(analysisName).getCopy();
this.deprecated_flowFactsOUT.put(analysisName, copiedFlowFactOUT);
}
}
return deprecated_flowFactsOUT.get(analysisName);
}
@Deprecated
public <F extends Deprecated_FlowFact> void setParallelFlowFactIN(AnalysisName analysisName, F flowFact) {
if (deprecated_parallelFlowFactsIN == null) {
deprecated_parallelFlowFactsIN = new HashMap<>();
}
deprecated_parallelFlowFactsIN.put(analysisName, flowFact);
}
@Deprecated
public <F extends Deprecated_FlowFact> void setParallelFlowFactOUT(AnalysisName analysis, F flowFact) {
if (deprecated_parallelFlowFactsOUT == null) {
deprecated_parallelFlowFactsOUT = new HashMap<>();
}
deprecated_parallelFlowFactsOUT.put(analysis, flowFact);
}
@Deprecated
public <F extends Deprecated_FlowFact> void setFlowFactIN(AnalysisName analysisName, F flowFact) {
if (deprecated_flowFactsIN == null) {
deprecated_flowFactsIN = new HashMap<>();
}
deprecated_flowFactsIN.put(analysisName, flowFact);
}
@Deprecated
public <F extends Deprecated_FlowFact> void setFlowFactOUT(AnalysisName analysis, F flowFact) {
if (deprecated_flowFactsOUT == null) {
deprecated_flowFactsOUT = new HashMap<>();
}
deprecated_flowFactsOUT.put(analysis, flowFact);
}
@Deprecated
public void removeParallelFlowFact(AnalysisName type) {
if (deprecated_parallelFlowFactsOUT == null) {
return;
}
deprecated_parallelFlowFactsOUT.remove(type);
if (deprecated_parallelFlowFactsIN == null) {
return;
}
deprecated_parallelFlowFactsIN.remove(type);
}
@Deprecated
public void removeFlowFact(AnalysisName type) {
if (deprecated_flowFactsOUT == null) {
return;
}
deprecated_flowFactsOUT.remove(type);
}
@Deprecated
public Set<Definition> deprecated_getReachingDefinitions() {
if (reachingDefinitions == null) {
reachingDefinitions = new HashSet<>();
}
return reachingDefinitions;
}
@Deprecated
public Set<Definition> deprecated_readReachingDefinitions() {
return reachingDefinitions;
}
@Deprecated
public CellMap<Set<Node>> getUsesInDU() {
if (usesInDU == null) {
usesInDU = new CellMap<>();
}
return usesInDU;
}
@Deprecated
public CellMap<Set<Node>> readUsesInDU() {
return usesInDU;
}
@Deprecated
public CellMap<Set<Node>> getDefsInUD() {
if (defsInUD == null) {
defsInUD = new CellMap<>();
}
return defsInUD;
}
@Deprecated
public CellMap<Set<Node>> readDefsInUD() {
return defsInUD;
}
@Deprecated
public HashMap<Node, CellSet> getLiveOut() {
if (liveOut == null) {
liveOut = new HashMap<>();
}
return liveOut;
}
@Deprecated
public CellSet getLiveOutCells() {
CellSet liveCells = new CellSet();
for (Node keySucc : getLiveOut().keySet()) {
liveCells = Misc.setUnion(liveCells, getLiveOut().get(keySucc));
}
return liveCells;
}
@Deprecated
public CellMap<Set<Node>> getFlowEdgeDestList() {
if (flowEdgeDestList == null) {
flowEdgeDestList = new CellMap<>();
}
return flowEdgeDestList;
}
@Deprecated
public CellMap<Set<Node>> readFlowEdgeDestList() {
return flowEdgeDestList;
}
@Deprecated
public CellMap<Set<Node>> getFlowEdgeSrcList() {
if (flowEdgeSrcList == null) {
flowEdgeSrcList = new CellMap<>();
}
return flowEdgeSrcList;
}
@Deprecated
public CellMap<Set<Node>> readFlowEdgeSrcList() {
return flowEdgeSrcList;
}
@Deprecated
public CellMap<Set<Node>> getAntiEdgeDestList() {
if (antiEdgeDestList == null) {
antiEdgeDestList = new CellMap<>();
}
return antiEdgeDestList;
}
@Deprecated
public CellMap<Set<Node>> readAntiEdgeDestList() {
return antiEdgeDestList;
}
@Deprecated
public CellMap<Set<Node>> getAntiEdgeSrcList() {
if (antiEdgeSrcList == null) {
antiEdgeSrcList = new CellMap<>();
}
return antiEdgeSrcList;
}
@Deprecated
public CellMap<Set<Node>> readAntiEdgeSrcList() {
return antiEdgeSrcList;
}
@Deprecated
public CellMap<Set<Node>> getOutputEdgeDestList() {
if (outputEdgeDestList == null) {
outputEdgeDestList = new CellMap<>();
}
return outputEdgeDestList;
}
@Deprecated
public CellMap<Set<Node>> readOutputEdgeDestList() {
return outputEdgeDestList;
}
@Deprecated
public CellMap<Set<Node>> getOutputEdgeSrcList() {
if (outputEdgeSrcList == null) {
outputEdgeSrcList = new CellMap<>();
}
return outputEdgeSrcList;
}
@Deprecated
public CellMap<Set<Node>> readOutputEdgeSrcList() {
return outputEdgeSrcList;
}
@Deprecated
public List<CallSite> getCallSites() {
if (calledFunctions_old == null) {
CallSiteGetter callGetter = new CallSiteGetter();
getNode().accept(callGetter);
calledFunctions_old = callGetter.callSiteList;
}
return calledFunctions_old;
}
@Deprecated
public boolean hasCallSites() {
if (getCallSites().isEmpty()) {
return false;
}
return true;
}
@Deprecated
public void resetCallSites() {
calledFunctions_old = null;
getCallSites();
}
}
