package imop.lib.transform;
import imop.ast.node.external.*;
import imop.lib.analysis.typesystem.SignedLongLongIntType;
import imop.lib.builder.Builder;
import imop.lib.cg.NodeWithStack;
import imop.lib.transform.updater.InsertImmediatePredecessor;
import imop.lib.util.Misc;
import imop.parser.FrontEnd;
import imop.parser.Program;
import java.util.Set;
public class BarrierCounterInstrumentation {
public static void insertBarrierCounters() {
String totalCounterName = Builder.getNewTempName("totalBarrCounter");
Declaration counterDeclaration = SignedLongLongIntType.type().getDeclaration(totalCounterName);
Builder.addDeclarationToGlobals(counterDeclaration);
FunctionDefinition mainFunction = Program.getRoot().getInfo().getMainFunction();
ExpressionStatement totalInit = FrontEnd.parseAndNormalize(totalCounterName + " = 0;",
ExpressionStatement.class);
if (mainFunction == null) {
Misc.exitDueToError("Cannot insert counters for a program with no main function.");
}
mainFunction.getInfo().getCFGInfo().getBody().getInfo().getCFGInfo().addElement(0, totalInit);
for (ParallelConstruct parCons : Misc.getInheritedEnclosee(Program.getRoot(), ParallelConstruct.class)) {
String counterName = Builder.getNewTempName("barrCounter");
counterDeclaration = SignedLongLongIntType.type().getDeclaration(counterName);
Builder.addDeclarationToGlobals(counterDeclaration);
CompoundStatement parBody = (CompoundStatement) parCons.getInfo().getCFGInfo().getBody();
Statement counterInit = FrontEnd.parseAndNormalize(
"#pragma omp master\n{" + counterName + " = 2" + ";" + totalCounterName + " += 2;}",
Statement.class);
parBody.getInfo().getCFGInfo().addElement(0, counterInit);
for (NodeWithStack cfgNodeWithStack : parBody.getInfo().getCFGInfo()
.getIntraTaskCFGLeafContentsOfSameParLevel()) {
Node cfgNode = cfgNodeWithStack.getNode();
if (cfgNode instanceof BarrierDirective) {
BarrierDirective barrier = (BarrierDirective) cfgNode;
Statement counterAdd = FrontEnd.parseAndNormalize(
"#pragma omp master\n{" + counterName + "++;" + totalCounterName + "++;}", Statement.class);
InsertImmediatePredecessor.insert(barrier, counterAdd);
}
}
String printCode = "#pragma omp master\n{printf(\"The master thread encountered %d barriers.\\n\", "
+ counterName + ");}";
Statement counterPrint = FrontEnd.parseAndNormalize(printCode, Statement.class);
parBody.getInfo().getCFGInfo().addAtLast(counterPrint);
}
Set<ReturnStatement> allReturns = Misc.getExactEnclosee(mainFunction, ReturnStatement.class);
if (allReturns == null || allReturns.isEmpty()) {
Statement totalBarriers = FrontEnd.parseAndNormalize(
"printf(\"TOTAL NUMBER OF BARRIERS ENCOUNTERED: %d.\\n\", " + totalCounterName + ");",
Statement.class);
mainFunction.getInfo().getCFGInfo().getBody().getInfo().getCFGInfo().addAtLast(totalBarriers);
} else {
for (ReturnStatement retStmt : allReturns) {
Statement totalBarriers = FrontEnd.parseAndNormalize(
"printf(\"TOTAL NUMBER OF BARRIERS ENCOUNTERED: %d.\\n\", " + totalCounterName + ");",
Statement.class);
InsertImmediatePredecessor.insert(retStmt, totalBarriers);
}
}
}
}
