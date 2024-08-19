#ifndef __LULESH
#define __LULESH

#include <math.h>
#include <vector>

#define NT 2

/
class Domain {

public:

Domain(Int_t numRanks, Index_t colLoc,
Index_t rowLoc, Index_t planeLoc,
Index_t nx, Int_t tp, Int_t nr, Int_t balance, Int_t cost);


void AllocateNodePersistent(Int_t numNode) 
{
m_x.resize(numNode);  
m_y.resize(numNode);
m_z.resize(numNode);

m_xd.resize(numNode); 
m_yd.resize(numNode);
m_zd.resize(numNode);

m_xdd.resize(numNode); 
m_ydd.resize(numNode);
m_zdd.resize(numNode);

m_fx.resize(numNode);  
m_fy.resize(numNode);
m_fz.resize(numNode);

m_nodalMass.resize(numNode);  
}

void AllocateElemPersistent(Int_t numElem) 
{
m_nodelist.resize(8*numElem);

m_lxim.resize(numElem);
m_lxip.resize(numElem);
m_letam.resize(numElem);
m_letap.resize(numElem);
m_lzetam.resize(numElem);
m_lzetap.resize(numElem);

m_elemBC.resize(numElem);

m_e.resize(numElem);
m_p.resize(numElem);

m_q.resize(numElem);
m_ql.resize(numElem);
m_qq.resize(numElem);

m_v.resize(numElem);

m_volo.resize(numElem);
m_delv.resize(numElem);
m_vdov.resize(numElem);

m_arealg.resize(numElem);

m_ss.resize(numElem);

m_elemMass.resize(numElem);

m_elemRep.resize(numElem);

m_elemElem.resize(numElem);
}

void AllocateGradients(Int_t numElem, Int_t allElem)
{
m_delx_xi.resize(numElem) ;
m_delx_eta.resize(numElem) ;
m_delx_zeta.resize(numElem) ;

m_delv_xi.resize(allElem) ;
m_delv_eta.resize(allElem);
m_delv_zeta.resize(allElem) ;
}

void DeallocateGradients()
{
m_delx_zeta.clear() ;
m_delx_eta.clear() ;
m_delx_xi.clear() ;

m_delv_zeta.clear() ;
m_delv_eta.clear() ;
m_delv_xi.clear() ;
}

void AllocateStrains(Int_t numElem)
{
m_dxx.resize(numElem) ;
m_dyy.resize(numElem) ;
m_dzz.resize(numElem) ;
}

void DeallocateStrains()
{
m_dzz.clear() ;
m_dyy.clear() ;
m_dxx.clear() ;
}



Real_t& x(Index_t idx)    { return m_x[idx] ; }
Real_t& y(Index_t idx)    { return m_y[idx] ; }
Real_t& z(Index_t idx)    { return m_z[idx] ; }

Real_t& xd(Index_t idx)   { return m_xd[idx] ; }
Real_t& yd(Index_t idx)   { return m_yd[idx] ; }
Real_t& zd(Index_t idx)   { return m_zd[idx] ; }

Real_t& xdd(Index_t idx)  { return m_xdd[idx] ; }
Real_t& ydd(Index_t idx)  { return m_ydd[idx] ; }
Real_t& zdd(Index_t idx)  { return m_zdd[idx] ; }

Real_t& fx(Index_t idx)   { return m_fx[idx] ; }
Real_t& fy(Index_t idx)   { return m_fy[idx] ; }
Real_t& fz(Index_t idx)   { return m_fz[idx] ; }

Real_t& nodalMass(Index_t idx) { return m_nodalMass[idx] ; }

Index_t symmX(Index_t idx) { return m_symmX[idx] ; }
Index_t symmY(Index_t idx) { return m_symmY[idx] ; }
Index_t symmZ(Index_t idx) { return m_symmZ[idx] ; }
bool symmXempty()          { return m_symmX.empty(); }
bool symmYempty()          { return m_symmY.empty(); }
bool symmZempty()          { return m_symmZ.empty(); }

Index_t&  regElemSize(Index_t idx) { return m_regElemSize[idx] ; }
Index_t&  regNumList(Index_t idx) { return m_regNumList[idx] ; }
Index_t*  regNumList()            { return &m_regNumList[0] ; }
Index_t*  regElemlist(Int_t r)    { return m_regElemlist[r] ; }
Index_t&  regElemlist(Int_t r, Index_t idx) { return m_regElemlist[r][idx] ; }

Index_t*  nodelist(Index_t idx)    { return &m_nodelist[Index_t(8)*idx] ; }

Index_t&  lxim(Index_t idx) { return m_lxim[idx] ; }
Index_t&  lxip(Index_t idx) { return m_lxip[idx] ; }
Index_t&  letam(Index_t idx) { return m_letam[idx] ; }
Index_t&  letap(Index_t idx) { return m_letap[idx] ; }
Index_t&  lzetam(Index_t idx) { return m_lzetam[idx] ; }
Index_t&  lzetap(Index_t idx) { return m_lzetap[idx] ; }

Int_t&  elemBC(Index_t idx) { return m_elemBC[idx] ; }

Real_t& dxx(Index_t idx)  { return m_dxx[idx] ; }
Real_t& dyy(Index_t idx)  { return m_dyy[idx] ; }
Real_t& dzz(Index_t idx)  { return m_dzz[idx] ; }

Real_t& delv_xi(Index_t idx)    { return m_delv_xi[idx] ; }
Real_t& delv_eta(Index_t idx)   { return m_delv_eta[idx] ; }
Real_t& delv_zeta(Index_t idx)  { return m_delv_zeta[idx] ; }

Real_t& delx_xi(Index_t idx)    { return m_delx_xi[idx] ; }
Real_t& delx_eta(Index_t idx)   { return m_delx_eta[idx] ; }
Real_t& delx_zeta(Index_t idx)  { return m_delx_zeta[idx] ; }

Real_t& e(Index_t idx)          { return m_e[idx] ; }

Real_t& p(Index_t idx)          { return m_p[idx] ; }

Real_t& q(Index_t idx)          { return m_q[idx] ; }

Real_t& ql(Index_t idx)         { return m_ql[idx] ; }
Real_t& qq(Index_t idx)         { return m_qq[idx] ; }

Real_t& v(Index_t idx)          { return m_v[idx] ; }
Real_t& delv(Index_t idx)       { return m_delv[idx] ; }

Real_t& volo(Index_t idx)       { return m_volo[idx] ; }

Real_t& vdov(Index_t idx)       { return m_vdov[idx] ; }

Real_t& arealg(Index_t idx)     { return m_arealg[idx] ; }

Real_t& ss(Index_t idx)         { return m_ss[idx] ; }

Real_t& elemMass(Index_t idx)  { return m_elemMass[idx] ; }

Index_t& elemRep(Index_t idx)  { return m_elemRep[idx] ; }

Index_t& elemElem(Index_t idx)  { return m_elemElem[idx] ; }

Index_t nodeElemCount(Index_t idx)
{ return m_nodeElemStart[idx+1] - m_nodeElemStart[idx] ; }

Index_t *nodeElemCornerList(Index_t idx)
{ return &m_nodeElemCornerList[m_nodeElemStart[idx]] ; }


Real_t u_cut() const               { return m_u_cut ; }
Real_t e_cut() const               { return m_e_cut ; }
Real_t p_cut() const               { return m_p_cut ; }
Real_t q_cut() const               { return m_q_cut ; }
Real_t v_cut() const               { return m_v_cut ; }

Real_t hgcoef() const              { return m_hgcoef ; }
Real_t qstop() const               { return m_qstop ; }
Real_t monoq_max_slope() const     { return m_monoq_max_slope ; }
Real_t monoq_limiter_mult() const  { return m_monoq_limiter_mult ; }
Real_t ss4o3() const               { return m_ss4o3 ; }
Real_t qlc_monoq() const           { return m_qlc_monoq ; }
Real_t qqc_monoq() const           { return m_qqc_monoq ; }
Real_t qqc() const                 { return m_qqc ; }

Real_t eosvmax() const             { return m_eosvmax ; }
Real_t eosvmin() const             { return m_eosvmin ; }
Real_t pmin() const                { return m_pmin ; }
Real_t emin() const                { return m_emin ; }
Real_t dvovmax() const             { return m_dvovmax ; }
Real_t refdens() const             { return m_refdens ; }

Real_t& time()                 { return m_time ; }
Real_t& deltatime()            { return m_deltatime ; }
Real_t& deltatimemultlb()      { return m_deltatimemultlb ; }
Real_t& deltatimemultub()      { return m_deltatimemultub ; }
Real_t& stoptime()             { return m_stoptime ; }
Real_t& dtcourant()            { return m_dtcourant ; }
Real_t& dthydro()              { return m_dthydro ; }
Real_t& dtmax()                { return m_dtmax ; }
Real_t& dtfixed()              { return m_dtfixed ; }

Int_t&  cycle()                { return m_cycle ; }
Index_t&  numRanks()           { return m_numRanks ; }

Index_t&  colLoc()             { return m_colLoc ; }
Index_t&  rowLoc()             { return m_rowLoc ; }
Index_t&  planeLoc()           { return m_planeLoc ; }
Index_t&  tp()                 { return m_tp ; }

Index_t&  sizeX()              { return m_sizeX ; }
Index_t&  sizeY()              { return m_sizeY ; }
Index_t&  sizeZ()              { return m_sizeZ ; }
Index_t&  numReg()             { return m_numReg ; }
Int_t&  cost()             { return m_cost ; }
Index_t&  numElem()            { return m_numElem ; }
Index_t&  numNode()            { return m_numNode ; }

Index_t&  maxPlaneSize()       { return m_maxPlaneSize ; }
Index_t&  maxEdgeSize()        { return m_maxEdgeSize ; }


void BuildMesh(Int_t nx, Int_t edgeNodes, Int_t edgeElems);
void SetupThreadSupportStructures();
void CreateRegionIndexSets(Int_t nreg, Int_t balance);
void SetupCommBuffers(Int_t edgeNodes);
void SetupSymmetryPlanes(Int_t edgeNodes);
void SetupElementConnectivities(Int_t edgeElems);
void SetupBoundaryConditions(Int_t edgeElems);



std::vector<Real_t> m_x ;  
std::vector<Real_t> m_y ;
std::vector<Real_t> m_z ;

std::vector<Real_t> m_xd ; 
std::vector<Real_t> m_yd ;
std::vector<Real_t> m_zd ;

std::vector<Real_t> m_xdd ; 
std::vector<Real_t> m_ydd ;
std::vector<Real_t> m_zdd ;

std::vector<Real_t> m_fx ;  
std::vector<Real_t> m_fy ;
std::vector<Real_t> m_fz ;

std::vector<Real_t> m_nodalMass ;  

std::vector<Index_t> m_symmX ;  
std::vector<Index_t> m_symmY ;
std::vector<Index_t> m_symmZ ;


Int_t    m_numReg ;
Int_t    m_cost; 
Index_t *m_regElemSize ;   
Index_t *m_regNumList ;    
Index_t **m_regElemlist ;  

std::vector<Index_t>  m_nodelist ;     

std::vector<Index_t>  m_lxim ;  
std::vector<Index_t>  m_lxip ;
std::vector<Index_t>  m_letam ;
std::vector<Index_t>  m_letap ;
std::vector<Index_t>  m_lzetam ;
std::vector<Index_t>  m_lzetap ;

std::vector<Int_t>    m_elemBC ;  

std::vector<Real_t> m_dxx ;  
std::vector<Real_t> m_dyy ;
std::vector<Real_t> m_dzz ;

std::vector<Real_t> m_delv_xi ;    
std::vector<Real_t> m_delv_eta ;
std::vector<Real_t> m_delv_zeta ;

std::vector<Real_t> m_delx_xi ;    
std::vector<Real_t> m_delx_eta ;
std::vector<Real_t> m_delx_zeta ;

std::vector<Real_t> m_e ;   

std::vector<Real_t> m_p ;   
std::vector<Real_t> m_q ;   
std::vector<Real_t> m_ql ;  
std::vector<Real_t> m_qq ;  

std::vector<Real_t> m_v ;     
std::vector<Real_t> m_volo ;  
std::vector<Real_t> m_vnew ;  
std::vector<Real_t> m_delv ;  
std::vector<Real_t> m_vdov ;  

std::vector<Real_t> m_arealg ;  

std::vector<Real_t> m_ss ;      

std::vector<Real_t> m_elemMass ;  

std::vector<Index_t> m_elemRep ;  

std::vector<Index_t> m_elemElem ;  

const Real_t  m_e_cut ;             
const Real_t  m_p_cut ;             
const Real_t  m_q_cut ;             
const Real_t  m_v_cut ;             
const Real_t  m_u_cut ;             


const Real_t  m_hgcoef ;            
const Real_t  m_ss4o3 ;
const Real_t  m_qstop ;             
const Real_t  m_monoq_max_slope ;
const Real_t  m_monoq_limiter_mult ;
const Real_t  m_qlc_monoq ;         
const Real_t  m_qqc_monoq ;         
const Real_t  m_qqc ;
const Real_t  m_eosvmax ;
const Real_t  m_eosvmin ;
const Real_t  m_pmin ;              
const Real_t  m_emin ;              
const Real_t  m_dvovmax ;           
const Real_t  m_refdens ;           

Real_t  m_dtcourant ;         
Real_t  m_dthydro ;           
Int_t   m_cycle ;             
Real_t  m_dtfixed ;           
Real_t  m_time ;              
Real_t  m_deltatime ;         
Real_t  m_deltatimemultlb ;
Real_t  m_deltatimemultub ;
Real_t  m_dtmax ;             
Real_t  m_stoptime ;          


Int_t   m_numRanks ;

Index_t m_colLoc ;
Index_t m_rowLoc ;
Index_t m_planeLoc ;
Index_t m_tp ;

Index_t m_sizeX ;
Index_t m_sizeY ;
Index_t m_sizeZ ;
Index_t m_numElem ;
Index_t m_numNode ;

Index_t m_maxPlaneSize ;
Index_t m_maxEdgeSize ;

Index_t *m_nodeElemStart ;
Index_t *m_nodeElemCornerList ;

Index_t m_rowMin, m_rowMax;
Index_t m_colMin, m_colMax;
Index_t m_planeMin, m_planeMax ;

} ;

typedef Real_t &(Domain::* Domain_member )(Index_t) ;

struct cmdLineOpts {
Int_t its; 
Int_t nx;  
Int_t numReg; 
Int_t numFiles; 
Int_t showProg; 
Int_t quiet; 
Int_t viz; 
Int_t cost; 
Int_t balance; 
Int_t iteration_cap; 
};




Real_t CalcElemVolume( const Real_t x[8],
const Real_t y[8],
const Real_t z[8]);

void ParseCommandLineOptions(int argc, char *argv[],
Int_t myRank, struct cmdLineOpts *opts);
void VerifyAndWriteFinalOutput(Real_t elapsed_time,
Domain& locDom,
Int_t nx,
Int_t numRanks);

void DumpToVisit(Domain& domain, int numFiles, int myRank, int numRanks);

void InitMeshDecomp(Int_t numRanks, Int_t myRank,
Int_t *col, Int_t *row, Int_t *plane, Int_t *side);

#endif
