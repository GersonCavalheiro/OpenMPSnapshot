package imop.ast.node.external;
public class OmpPragma extends Node {
{
classId = 211;
}
private static final long serialVersionUID = -5513477653523572872L;
private NodeToken f0;
private NodeToken f1;
private NodeToken f2;
public OmpPragma(NodeToken n0, NodeToken n1, NodeToken n2) {
n0.setParent(this);
n1.setParent(this);
n2.setParent(this);
setF0(n0);
setF1(n1);
setF2(n2);
}
public OmpPragma() {
setF0(new NodeToken("#"));
getF0().setParent(this);
setF1(new NodeToken("pragma"));
getF1().setParent(this);
setF2(new NodeToken("omp"));
getF2().setParent(this);
}
@Override
public void accept(imop.baseVisitor.Visitor v) {
v.visit(this);
}
@Override
public <R, A> R accept(imop.baseVisitor.GJVisitor<R, A> v, A argu) {
return v.visit(this, argu);
}
@Override
public <R> R accept(imop.baseVisitor.GJNoArguVisitor<R> v) {
return v.visit(this);
}
@Override
public <A> void accept(imop.baseVisitor.GJVoidVisitor<A> v, A argu) {
v.visit(this, argu);
}
public NodeToken getF0() {
return f0;
}
public void setF0(NodeToken f0) {
f0.setParent(this);
this.f0 = f0;
}
public NodeToken getF1() {
return f1;
}
public void setF1(NodeToken f1) {
f1.setParent(this);
this.f1 = f1;
}
public NodeToken getF2() {
return f2;
}
public void setF2(NodeToken f2) {
f2.setParent(this);
this.f2 = f2;
}
}
