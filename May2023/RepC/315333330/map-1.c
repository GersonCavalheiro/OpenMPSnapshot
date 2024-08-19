extern int a[][10], a2[][10];
int b[10], c[10][2], d[10], e[10], f[10];
int b2[10], c2[10][2], d2[10], e2[10], f2[10];
int k[10], l[10], m[10], n[10], o;
int *p;
int **q;
int r[4][4][4][4][4];
extern struct s s1;
extern struct s s2[1]; 
int t[10];
#pragma omp threadprivate (t)
#pragma omp declare target
void bar (int *);
#pragma omp end declare target
void
foo (int g[3][10], int h[4][8], int i[2][10], int j[][9],
int g2[3][10], int h2[4][8], int i2[2][10], int j2[][9])
{
#pragma omp target map(to: bar[2:5]) 
;
#pragma omp target map(from: t[2:5]) 
;
#pragma omp target map(tofrom: k[0.5:]) 
;
#pragma omp target map(from: l[:7.5f]) 
;
#pragma omp target map(to: m[p:]) 
;
#pragma omp target map(tofrom: n[:p]) 
;
#pragma omp target map(to: o[2:5]) 
;
#pragma omp target map(alloc: s1) 
;
#pragma omp target map(alloc: s2) 
;
#pragma omp target map(to: a[:][:]) 
bar (&a[0][0]); 
#pragma omp target map(tofrom: b[-1:]) 
bar (b);
#pragma omp target map(tofrom: c[:-3][:]) 
bar (&c[0][0]);
#pragma omp target map(from: d[11:]) 
bar (d);
#pragma omp target map(to: e[:11]) 
bar (e);
#pragma omp target map(to: f[1:10]) 
bar (f);
#pragma omp target map(from: g[:][0:10]) 
bar (&g[0][0]);
#pragma omp target map(from: h[2:1][-1:]) 
bar (&h[0][0]);
#pragma omp target map(tofrom: h[:1][:-3]) 
bar (&h[0][0]);
#pragma omp target map(i[:1][11:]) 
bar (&i[0][0]);
#pragma omp target map(from: j[3:1][:10]) 
bar (&j[0][0]);
#pragma omp target map(to: j[30:1][5:5]) 
bar (&j[0][0]);
#pragma omp target map(to: a2[:1][2:4])
bar (&a2[0][0]);
#pragma omp target map(a2[3:5][:])
bar (&a2[0][0]);
#pragma omp target map(to: a2[3:5][:10])
bar (&a2[0][0]);
#pragma omp target map(tofrom: b2[0:])
bar (b2);
#pragma omp target map(tofrom: c2[:3][:])
bar (&c2[0][0]);
#pragma omp target map(from: d2[9:])
bar (d2);
#pragma omp target map(to: e2[:10])
bar (e2);
#pragma omp target map(to: f2[1:9])
bar (f2);
#pragma omp target map(g2[:1][2:4])
bar (&g2[0][0]);
#pragma omp target map(from: h2[2:2][0:])
bar (&h2[0][0]);
#pragma omp target map(tofrom: h2[:1][:3])
bar (&h2[0][0]);
#pragma omp target map(to: i2[:1][9:])
bar (&i2[0][0]);
#pragma omp target map(from: j2[3:4][:9])
bar (&j2[0][0]);
#pragma omp target map(to: j2[30:1][5:4])
bar (&j2[0][0]);
#pragma omp target map(q[1:2])
;
#pragma omp target map(tofrom: q[3:5][:10]) 
;
#pragma omp target map(r[3:][2:1][1:2])
;
#pragma omp target map(r[3:][2:1][1:2][:][0:4])
;
#pragma omp target map(r[3:][2:1][1:2][1:][0:4]) 
;
#pragma omp target map(r[3:][2:1][1:2][:3][0:4]) 
;
#pragma omp target map(r[3:][2:1][1:2][:][1:]) 
;
#pragma omp target map(r[3:][2:1][1:2][:][:3]) 
;
}
