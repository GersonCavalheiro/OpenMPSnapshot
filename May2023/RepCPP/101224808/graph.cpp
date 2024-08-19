#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <climits>
#include <omp.h>
#include "graph.hpp"

graph::graph(options* tmpOpt){

pOpt = tmpOpt;

if (pOpt->isOpenMPCores()) numCores = pOpt->getNumberOfCores();
else numCores = omp_get_max_threads();

std::cout << "Number of cores : " << numCores << std::endl;
}

graph::~graph(){

delete Pl;
delete Pc;

}

void graph::processInputPropertyGraph(){

auto t1 = std::chrono::high_resolution_clock::now();

readAttribs(pOpt->getSourceAttribsFile());
readEdgeList(pOpt->getSourceGraphFile());
printBasicStats();

auto t2 = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1);
auto durationPreProcess = duration;
std::cout << "Source dataset read time : " << (float)duration.count()/1000.0 << " seconds." << std::endl;

if (pOpt->isAugment()){

t1 = std::chrono::high_resolution_clock::now();

augmentNodeLabels();

t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1);
durationPreProcess += duration;
std::cout << "Label augmentation time : " << (float)duration.count()/1000.0 << " seconds." << std::endl;
}

t1 = std::chrono::high_resolution_clock::now();

createNodeCategoryDistribution();
createEdgeCategoryDistribution();

t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1);
durationPreProcess += duration;
std::cout << "Distributions build time : " << (float)duration.count()/1000.0 << " seconds." << std::endl;
std::cout << "Total preprocessing time : " << (float)durationPreProcess.count()/1000.0 << " seconds." << std::endl;

}

void graph::outputTargetPropertyGraph(){

auto t1 = std::chrono::high_resolution_clock::now();

createTargetNodes();

auto t2 = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1);
auto durationGeneration = duration;
std::cout << "Target dataset node set generation time : " << (float)duration.count()/1000.0 << " seconds." << std::endl;

t1 = std::chrono::high_resolution_clock::now();

createTargetEdges();

t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1);
durationGeneration += duration;
std::cout << "Target dataset edge set generation time : " << (float)duration.count()/1000.0 << " seconds." << std::endl;
std::cout << "Total generation time : " << (float)durationGeneration.count()/1000.0 << " seconds." << std::endl;

t1 = std::chrono::high_resolution_clock::now();

writeTargetEdgeList(pOpt->getTargetGraphFile());

t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1);
auto durationOutput = duration;
std::cout << "Target dataset edge list file output time : " << (float)duration.count()/1000.0 << " seconds." << std::endl;

t1 = std::chrono::high_resolution_clock::now();

writeTargetNodeAttributes(pOpt->getTargetAttribsFile());

t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1);
durationOutput += duration;
std::cout << "Target dataset node attribute file output time : " << (float)duration.count()/1000.0 << " seconds." << std::endl;
std::cout << "Total file output time : " << (float)durationOutput.count()/1000.0 << " seconds." << std::endl;

}

void graph::readAttribs(std::string fName){

std::fstream attrFile(fName, std::ios_base::in);

ul nodeIdx = 0;
ul nAttr = pOpt->getNumAttribs();

if (attrFile.good()){

std::vector<ul> thisNodeAttribs(nAttr,0);
if (pOpt->isAttributeInputExplicit())   { 
while (attrFile >> nodeIdx) {
addNode(nodeIdx);
for (size_t idx = 0; idx < (nAttr); ++idx){ attrFile >> thisNodeAttribs[idx];   }
mSrcNodeAttr[nodeIdx] = thisNodeAttribs; 
mSrcNodeDegree[nodeIdx] = 0; 
}
}
else    { 
while (attrFile >> thisNodeAttribs[0]) {
addNode(nodeIdx);
for (size_t idx = 1; idx < (nAttr); ++idx){ attrFile >> thisNodeAttribs[idx];   }
mSrcNodeAttr[nodeIdx] = thisNodeAttribs; 
mSrcNodeDegree[nodeIdx] = 0; 
++nodeIdx;
}
}
}
else{

std::cerr << "Unable to read from the input attributes file. Please check." << std::endl;
exit(EXIT_FAILURE);
}
}

void graph::readEdgeList(std::string fName){

std::fstream eList(fName, std::ios_base::in);
if (eList.good()){
ul src = 0, tar = 0;
while (eList >> src >> tar){
addEdge(src,tar);
mSrcNodeDegree[src] += 1;
mSrcNodeDegree[tar] += 1;
}

}
else{
std::cerr << "Unable to read from the input graph file. Please check." << std::endl;
exit(EXIT_FAILURE);
}

}

void graph::addNode(ul node){

nodesSrc.insert(node);
}

void graph::addEdge(ul src, ul tar){

edge thisEdge = std::make_tuple(src,tar);
edgesSrc.insert(thisEdge);

}

void graph::printBasicStats(){

std::cout << "Number of nodes in the source dataset : " << nodesSrc.size() << std::endl;
std::cout << "Number of edges in the source dataset : " << edgesSrc.size() << std::endl;

}

void graph::createNodeCategoryDistribution(){

ul nAttr = pOpt->getNumAttribs();
if (pOpt->isAugment()){
nAttr = nAttr + 1;
}

std::vector<ul> vAttribC = pOpt->getAttribCardinalities();
if (pOpt->isAugment()){
vAttribC.push_back(pOpt->getAugLabelCardinality());
}


std::vector<ul> vData;

auto itr1 = mSrcNodeAttr.begin();

for (; itr1 != mSrcNodeAttr.end(); ++itr1){

ul lCat = utils::convertToDecimal(itr1->second,vAttribC,nAttr);
mSrcNodeAttrDec[itr1->first] = lCat;
vData.push_back(lCat);
}

Pl = new distributions((unsigned int)pOpt->getRandomSeed());
ul lAttribC = pOpt->getTotalAttributeCardinality();
if (pOpt->isAugment()){
lAttribC = lAttribC*pOpt->getAugLabelCardinality();
}

Pl->setDataAndOptions(vData,lAttribC,pOpt->getNodeProbThreshold());
Pl->constructDist();

std::cout << "Node category distribution construction completed." << std::endl;

}

void graph::createEdgeCategoryDistribution(){

auto itr1 = edgesSrc.begin();
ul lAttribC = pOpt->getTotalAttributeCardinality();
if (pOpt->isAugment()){
lAttribC = lAttribC*pOpt->getAugLabelCardinality();
}


std::vector<ul> vData;

for (; itr1 != edgesSrc.end(); ++itr1){

edge thisEdge = *itr1;
ul node1Cat = mSrcNodeAttrDec[std::get<0>(thisEdge)];
ul node2Cat = mSrcNodeAttrDec[std::get<1>(thisEdge)];

vData.push_back(node1Cat + lAttribC*node2Cat);
}

Pc = new distributions((unsigned int)Pl->generateUniformRandomInt(UINT_MAX));
Pc->setDataAndOptions(vData, (lAttribC*lAttribC), pOpt->getEdgeProbThreshold());
Pc->constructDist();

std::cout << "Edge category distribution construction completed." << std::endl;

}

void graph::createTargetNodes(){

std::vector<ul> vSampledCats = Pl->generateSamples(pOpt->getNumTargetGraphNodes());

ul nAttribs = pOpt->getNumAttribs();
if (pOpt->isAugment()){
nAttribs = nAttribs + 1;
}
std::vector<ul> vAttribC = pOpt->getAttribCardinalities();
if (pOpt->isAugment()){
vAttribC.push_back(pOpt->getAugLabelCardinality());
}


for (size_t idx = 0; idx < pOpt->getNumTargetGraphNodes(); ++idx){
mTarNodeAttr[idx] = utils::convertToTuple(vSampledCats[idx],vAttribC, nAttribs);
(mCatToNodeList[vSampledCats[idx]]).push_back(idx);
}

for (auto itr1 = mCatToNodeList.begin(); itr1 != mCatToNodeList.end(); ++itr1){
mCatCounts[itr1->first] = (itr1->second).size();
}



}

void graph::createTargetEdges(){

std::vector<distributions*> vDistPtrs;
for (int idx=0; idx < numCores; ++idx){
distributions* tmpDist = new distributions(Pc,(unsigned int)Pc->generateUniformRandomInt(UINT_MAX));
vDistPtrs.push_back(tmpDist);
}

ul nAttribs = pOpt->getNumAttribs();
if (pOpt->isAugment()){
nAttribs = nAttribs + 1;
}
std::vector<ul> vAttribC = pOpt->getAttribCardinalities();
if (pOpt->isAugment()){
vAttribC.push_back(pOpt->getAugLabelCardinality());
}

ul lCat = pOpt->getTotalAttributeCardinality();
if (pOpt->isAugment()){
lCat = lCat*pOpt->getAugLabelCardinality();
}

vEdgesTar.resize(pOpt->getNumTargetGraphEdges());

omp_set_num_threads(numCores);
#pragma omp parallel shared(vEdgesTar)
{
ul numEdgesPerCore = pOpt->getNumTargetGraphEdges()/numCores;
std::vector<ul> vSampledCats = (vDistPtrs[omp_get_thread_num()])->generateSamples(numEdgesPerCore);
std::cout << "# Target edges processed (by this core) : " << vSampledCats.size() << std::endl;
std::ldiv_t dv{};

for (size_t idx = 0; idx < numEdgesPerCore; ++idx)
{
ul edgeCat = vSampledCats[idx];

dv = std::ldiv(edgeCat,lCat);
ul node1cat = dv.rem;
ul node2cat = dv.quot;

ul nodeIdx1, nodeIdx2;

if (mCatCounts.count(node1cat) > 0){
nodeIdx1= mCatToNodeList[node1cat][(vDistPtrs[omp_get_thread_num()])->generateUniformRandomInt(mCatCounts[node1cat]-1)];
}
else continue;

if (mCatCounts.count(node2cat) > 0){
nodeIdx2 = mCatToNodeList[node2cat][(vDistPtrs[omp_get_thread_num()])->generateUniformRandomInt(mCatCounts[node2cat]-1)];
}
else continue;

if (nodeIdx1 != nodeIdx2)   {
vEdgesTar[idx+omp_get_thread_num()*numEdgesPerCore] = (std::make_tuple(nodeIdx1,nodeIdx2));
}
}
}


for (auto it = vDistPtrs.begin() ; it != vDistPtrs.end(); ++it)
{
delete (*it);
}
vDistPtrs.clear();


}

void graph::writeTargetEdgeList(std::string fName){

std::stringstream ss;
for (size_t idx = 0; idx < vEdgesTar.size(); ++idx){
ss << std::get<0>(vEdgesTar[idx]) << "\t" << std::get<1>(vEdgesTar[idx]) << "\n";
}

std::ofstream oFs;

oFs.open(fName.c_str(),std::ofstream::out);
oFs << ss.rdbuf();
oFs.close();

}

void graph::writeTargetNodeAttributes(std::string fName){

std::stringstream ss;
ul nAttribs = pOpt->getNumAttribs();

for (auto itr1 = mTarNodeAttr.begin(); itr1 != mTarNodeAttr.end(); ++itr1){
ss << itr1->first << "\t";
for (size_t idx = 0; idx < nAttribs; ++idx){
ss << (itr1->second)[idx] << " ";
}
ss << "\n";
}

std::ofstream oFs;

oFs.open(fName.c_str(),std::ofstream::out);
oFs << ss.rdbuf();
oFs.close();

}

void graph::augmentNodeLabels(){

std::vector<std::string> vQuartDesc = {"Min","25%","50%","75%","Max"};
for (auto itr1 = mSrcNodeDegree.begin(); itr1 != mSrcNodeDegree.end(); ++itr1){
vDegree.push_back(itr1->second);
}
std::sort(vDegree.begin(),vDegree.end());

std::cout << "Displaying all quartiles for the degree distribution (min, 25%, 50%, 75%, max)" << std::endl;
std::vector<double> vQuarts = utils::computeQuartiles(vDegree);
for (size_t idx=0; idx < vQuarts.size(); ++idx){
std::cout << vQuartDesc[idx] << " : " << vQuarts[idx] << std::endl;
}
std::cout << std::endl;

std::string sAug = pOpt->getAugmentationType();
if (pOpt->isAugment()){
ul minDeg = vDegree.front();
ul maxDeg = vDegree.back();
std::cout << "Min and Max degrees : " << minDeg << " " << maxDeg << std::endl;


if (sAug.compare(std::string("Linear")) == 0) {
double spacing = ((double)(maxDeg+1.00)-(double)minDeg)/((double)pOpt->getAugLabelCardinality());
double offset = (double)minDeg;
for (auto itr1 = mSrcNodeDegree.begin(); itr1 != mSrcNodeDegree.end(); ++itr1 ){
ul augLabel = (floor(((double)itr1->second - offset)/spacing));
(mSrcNodeAttr[itr1->first]).push_back(augLabel);
}
}
else if (sAug.compare(std::string("Logarithmic")) == 0) {
double spacing = (log2((double)maxDeg+1.00)-log2((double)minDeg))/((double)pOpt->getAugLabelCardinality());
double offset = log2((double)minDeg);
for (auto itr1 = mSrcNodeDegree.begin(); itr1 != mSrcNodeDegree.end(); ++itr1 ){
ul augLabel = (floor((log2((double)itr1->second) - offset)/spacing));
(mSrcNodeAttr[itr1->first]).push_back(augLabel);
}
}
}
}
