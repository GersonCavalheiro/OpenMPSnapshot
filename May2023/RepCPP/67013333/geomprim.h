#pragma once

FORD distanceSquare(FORD onex, FORD oney, FORD twox, FORD twoy) {
FORD dx = onex - twox;
FORD dy = oney - twoy;
FORD dsq = dx * dx + dy * dy;
return dsq;
}

FORD distanceSquare(unsigned one, unsigned two, FORD *nodex, FORD *nodey) {
return distanceSquare(nodex[one], nodey[one], nodex[two], nodey[two]);
}

bool angleOB(Mesh &mesh, unsigned a, unsigned b, unsigned c) {
FORD vax = mesh.nodex[a] - mesh.nodex[c];
FORD vay = mesh.nodey[a] - mesh.nodey[c];
FORD vbx = mesh.nodex[b] - mesh.nodex[c];
FORD vby = mesh.nodey[b] - mesh.nodey[c];
FORD dp = vax * vbx + vay * vby; 

if (dp < 0) 
return true;

return false;
}

bool angleLT(Mesh &mesh, unsigned a, unsigned b, unsigned c) {
FORD vax = mesh.nodex[a] - mesh.nodex[c];
FORD vay = mesh.nodey[a] - mesh.nodey[c];
FORD vbx = mesh.nodex[b] - mesh.nodex[c];
FORD vby = mesh.nodey[b] - mesh.nodey[c];
FORD dp = vax * vbx + vay * vby; 

if (dp < 0) {
return false;
} else {
FORD dsqaacurr = distanceSquare(a, c, mesh.nodex, mesh.nodey);
FORD dsqbbcurr = distanceSquare(b, c, mesh.nodex, mesh.nodey);
FORD c = dp * (1.0f/sqrtf(dsqaacurr*dsqbbcurr));
if (c > cos(MINANGLE * (PI / 180))) {
return true;
}
}

return false;
}


FORD gincircle (FORD ax, FORD ay, FORD bx, FORD by, FORD cx, FORD cy, FORD px, FORD py) {
FORD apx, bpx, cpx, apy, bpy, cpy;
FORD bpxcpy, cpxbpy, cpxapy, apxcpy, apxbpy, bpxapy;
FORD alift, blift, clift, det;

apx = ax - px;
bpx = bx - px;
cpx = cx - px;

apy = ay - py;
bpy = by - py;
cpy = cy - py;

bpxcpy = bpx * cpy;
cpxbpy = cpx * bpy;
alift = apx * apx + apy * apy;

cpxapy = cpx * apy;
apxcpy = apx * cpy;
blift = bpx * bpx + bpy * bpy;

apxbpy = apx * bpy;
bpxapy = bpx * apy;
clift = cpx * cpx + cpy * cpy;

det = alift * (bpxcpy - cpxbpy) + blift * (cpxapy - apxcpy) + clift * (apxbpy - bpxapy);

return det;
}

FORD counterclockwise(FORD pax, FORD pay, FORD pbx, FORD pby, FORD pcx, FORD pcy) {
FORD detleft, detright, det;

detleft = (pax - pcx) * (pby - pcy);
detright = (pay - pcy) * (pbx - pcx);
det = detleft - detright;

return det;
}

void circumcenter(FORD Ax, FORD Ay, FORD Bx, FORD By, FORD Cx, FORD Cy, FORD &CCx, FORD &CCy) {
FORD D;
D = 2 * (Ax * (By - Cy) + Bx * (Cy - Ay) + Cx * (Ay - By));
CCx = ((Ax*Ax + Ay*Ay)*(By - Cy) + (Bx*Bx + By*By)*(Cy - Ay) + (Cx*Cx + Cy*Cy)*(Ay - By))/D;
CCy = ((Ax*Ax + Ay*Ay)*(Cx - Bx) + (Bx*Bx + By*By)*(Ax - Cx) + (Cx*Cx + Cy*Cy)*(Bx - Ax))/D;
}

