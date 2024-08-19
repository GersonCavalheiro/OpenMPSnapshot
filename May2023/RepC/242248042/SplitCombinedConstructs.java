package imop.lib.transform.simplify;
import imop.ast.node.external.*;
import imop.ast.node.internal.*;
import imop.lib.transform.BasicTransform;
import imop.lib.transform.updater.sideeffect.NodeUpdated;
import imop.lib.transform.updater.sideeffect.SideEffect;
import imop.lib.transform.updater.sideeffect.SyntacticConstraint;
import imop.lib.util.Misc;
import imop.parser.FrontEnd;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
public class SplitCombinedConstructs {
public static void splitCombinedConstructsWithin(Node node) {
Set<Class<? extends Node>> classList = new HashSet<>();
classList.add(ParallelForConstruct.class);
classList.add(ParallelSectionsConstruct.class);
for (Node combinedNode : Misc.getExactPostOrderEnclosee(node, classList)) {
if (combinedNode == node) {
continue;
}
if (combinedNode instanceof ParallelForConstruct) {
SplitCombinedConstructs.splitThisCombinedConstruct((ParallelForConstruct) combinedNode);
} else if (combinedNode instanceof ParallelSectionsConstruct) {
SplitCombinedConstructs.splitThisCombinedConstruct((ParallelSectionsConstruct) combinedNode);
}
}
}
public static List<SideEffect> splitCombinedConstructForTheStatement(Statement stmt) {
List<SideEffect> sideEffects = new ArrayList<>();
if (!(stmt.getStmtF0().getChoice() instanceof OmpConstruct)) {
return sideEffects;
}
OmpConstruct ompCons = (OmpConstruct) stmt.getStmtF0().getChoice();
Node combinedNode = ompCons.getOmpConsF0().getChoice();
if (combinedNode instanceof ParallelForConstruct) {
return SplitCombinedConstructs.splitThisCombinedConstruct((ParallelForConstruct) combinedNode);
} else if (combinedNode instanceof ParallelSectionsConstruct) {
return SplitCombinedConstructs.splitThisCombinedConstruct((ParallelSectionsConstruct) combinedNode);
} else {
return sideEffects;
}
}
public static List<SideEffect> splitThisCombinedConstruct(ParallelForConstruct parFor) {
List<SideEffect> sideEffectList = new ArrayList<>();
OmpConstruct ompCons = Misc.getEnclosingNode(parFor, OmpConstruct.class);
if (ompCons == null) {
sideEffectList.add(new SyntacticConstraint(parFor,
"Cannot simplify this parallel-for construct as there is no enclosing OmpConstruct for it."));
return sideEffectList;
}
StringBuilder forSpecificClauses = new StringBuilder();
StringBuilder parSpecificClauses = new StringBuilder();
List<OmpClause> clauseList = Misc.getClauseList(parFor);
for (OmpClause clause : clauseList) {
if (clause instanceof IfClause || clause instanceof NumThreadsClause || clause instanceof OmpPrivateClause
|| clause instanceof OmpFirstPrivateClause || clause instanceof OmpLastPrivateClause
|| clause instanceof OmpSharedClause || clause instanceof OmpCopyinClause
|| clause instanceof OmpDfltSharedClause || clause instanceof OmpDfltNoneClause) {
parSpecificClauses.append(" " + clause + " ");
} else {
forSpecificClauses.append(" " + clause + " ");
}
}
OmpConstruct parConsOmpConstruct = FrontEnd.parseAndNormalize(
"#pragma omp parallel " + parSpecificClauses.toString() + "\n {}", OmpConstruct.class);
try {
boolean couldSwap = BasicTransform.crudeReplaceOldWithNew(ompCons, parConsOmpConstruct);
if (!couldSwap) {
return sideEffectList;
}
} catch (IllegalArgumentException e) {
e.printStackTrace();
} catch (IllegalAccessException e) {
e.printStackTrace();
}
ParallelConstruct parCons = (ParallelConstruct) Misc.getCFGNodeFor(parConsOmpConstruct);
OmpForInitExpression initExp = parFor.getF5().getF2();
OmpForCondition condExp = parFor.getF5().getF4();
OmpForReinitExpression reinitExp = parFor.getF5().getF6();
Statement forBody = parFor.getF6();
ForConstruct forCons = FrontEnd.parseAndNormalize("#pragma omp for " + forSpecificClauses.toString()
+ "\n for (" + initExp + "; " + condExp + "; " + reinitExp + ") \n {}", ForConstruct.class);
Statement forConsStmtWrapper = Misc.getStatementWrapper(forCons);
parCons.setParConsF2(forConsStmtWrapper);
Statement forBodyStmtWrapper = Misc.getStatementWrapper(forBody);
forCons.setF3(forBodyStmtWrapper);
sideEffectList.add(
new NodeUpdated(parCons, "This parallel construct was created upon splitting of a par-for construct."));
return sideEffectList;
}
public static List<SideEffect> splitThisCombinedConstruct(ParallelSectionsConstruct parSec) {
List<SideEffect> sideEffectList = new ArrayList<>();
OmpConstruct ompCons = Misc.getEnclosingNode(parSec, OmpConstruct.class);
if (ompCons == null) {
sideEffectList.add(new SyntacticConstraint(parSec,
"Cannot simplify this parallel-sections construct as there is no enclosing OmpConstruct for it."));
return sideEffectList;
}
StringBuilder secSpecificClauses = new StringBuilder();
StringBuilder parSpecificClauses = new StringBuilder();
List<OmpClause> clauseList = Misc.getClauseList(parSec);
for (OmpClause clause : clauseList) {
if (clause instanceof IfClause || clause instanceof NumThreadsClause || clause instanceof OmpPrivateClause
|| clause instanceof OmpFirstPrivateClause || clause instanceof OmpLastPrivateClause
|| clause instanceof OmpSharedClause || clause instanceof OmpCopyinClause
|| clause instanceof OmpDfltSharedClause || clause instanceof OmpDfltNoneClause
) {
parSpecificClauses.append(" " + clause.getInfo().getString() + " ");
} else {
secSpecificClauses.append(" " + clause.getInfo().getString() + " ");
}
}
OmpConstruct parConsOmpConstruct = FrontEnd.parseAndNormalize(
"#pragma omp parallel " + parSpecificClauses.toString() + "\n {}", OmpConstruct.class);
try {
boolean couldSwap = BasicTransform.crudeReplaceOldWithNew(ompCons, parConsOmpConstruct);
if (!couldSwap) {
return sideEffectList;
}
} catch (IllegalArgumentException e) {
e.printStackTrace();
} catch (IllegalAccessException e) {
e.printStackTrace();
}
ParallelConstruct parCons = (ParallelConstruct) Misc.getCFGNodeFor(parConsOmpConstruct);
SectionsConstruct secCons = FrontEnd.parseAndNormalize(
"\n#pragma omp sections " + secSpecificClauses + "\n" + "{}", SectionsConstruct.class);
Statement secConsStmtWrapper = Misc.getStatementWrapper(secCons);
parCons.setParConsF2(secConsStmtWrapper);
secCons.setF4(parSec.getF5());
sideEffectList.add(new NodeUpdated(parCons,
"This parallel construct was created upon splitting of a par-sections construct."));
return sideEffectList;
}
}
