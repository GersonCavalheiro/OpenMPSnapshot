#include <iostream>
#include <fstream>
#include <exception>
#include <random>
#include <omp.h>
#include <cmath>
#include "CmdArgs.hpp"
#include "PuzzleReader.hpp"
#include "Timing.hpp"

static int solveOneFile(const CmdArgs &args, std::string filename, std::istream &puzzlefile) {
Timing tm1;
std::cout.precision(3);
if (filename.empty()) filename = "from stdin";
else filename = "\"" + filename + "\"";
Puzzle puzzle;
try {
PuzzleReader reader;
puzzle = reader.read(puzzlefile);
}
catch (std::exception& x) {
std::cerr << "Error: Cannot parse puzzle file " << filename << ". Reason:"
<< std::endl;
std::cerr << x.what() << std::endl;
return 1;
}
std::cout << "parse time=" << std::fixed << tm1.getRunTime() << "ms" << std::endl;
int i = 0;
for (Polyomino &po : puzzle.polyominoes) {
i += 1;
po.generateTransforms(puzzle.grid);
}
puzzle.board.sortCoords();
puzzle.buildDlxRows();
puzzle.buildDlxColumns();
puzzle.buildDlxCells();
std::cout << "build time=" << tm1.getRunTime() << "ms"<<std::endl;
if (args.reduction) {
puzzle.reduce();
std::cout << "reduce time=" << tm1.getRunTime() << "ms"<<std::endl;
}

puzzle.targetLevel = args.parallelLevel;
puzzle.saveSolution = true;
puzzle.dlxSolve();
std::vector<std::vector<int>> subproblems = puzzle.solutions;
std::cout << "create subproblem time=" << tm1.getRunTime() << "ms"<<std::endl;
int numSolution = 0;
puzzle.targetLevel = -1;  
puzzle.saveSolution = args.saveSolution;
puzzle.attempts = 0;
puzzle.solutions.clear();
puzzle.numSolution = 0;
int numThreads = omp_get_max_threads();
std::vector<int> prefixSum(numThreads);
std::vector<int> oneSolution;
int rndChoose;
int percent = 0, complete = 0;
std::default_random_engine gen = std::default_random_engine(1234);
for (int i = 1; i < subproblems.size(); i++) {
std::uniform_int_distribution<int> dis(0, i);
int r = dis(gen);
if (r != i) std::swap(subproblems[i], subproblems[r]);
}
std::cout << "Number of subproblems: " << subproblems.size() << '\n';
Timing timeall;
#pragma omp parallel firstprivate(puzzle, tm1)
{
#pragma omp master
if (!args.percent)
std::cout << "copy time=" << tm1.getRunTime() << "ms"<<std::endl;

std::vector<int> removedRowCount(args.parallelLevel);
int tid = omp_get_thread_num();

#pragma omp for schedule(dynamic, 1) reduction(+:numSolution)
for (int i = 0; i < subproblems.size(); i++) {
puzzle.dlxCounter = 0;
if (subproblems[i].size() != args.parallelLevel) {
if (args.saveSolution)
puzzle.solutions.push_back(subproblems[i]);
numSolution += 1;
continue;
}

for (int j = 0; j < args.parallelLevel; j++)
removedRowCount[j] = puzzle.enterBranch(subproblems[i][j]);

int nRows = puzzle.numRows;
puzzle.dlxSolve();

for (int j = args.parallelLevel; j > 0; j--)
puzzle.leaveBranch(removedRowCount[j-1]);

numSolution += puzzle.numSolution;
double runtime = tm1.getRunTime();
if (args.info) {
#pragma omp critical
std::cout << "thread "<<omp_get_thread_num()<<" solve in " << runtime << "ms "
<< nRows << " rows "
<< puzzle.numSolution << " solution "
<< puzzle.attempts << " attempts "
<< puzzle.dlxCounter << " unlinks"
<< std::endl;
}
if (args.percent) {
#pragma omp critical
{
complete += 1;
int finish = complete * 100.0 / subproblems.size();
if (finish > percent) {
double t = timeall.getRunTime(true);
std::cerr << "progress: " <<finish << "% remaining time: " << t * 0.1 / finish - t * 0.001 << "s";
std::cerr << std::string(15, ' ') << "\r";
percent = finish;
}
}
}
puzzle.attempts = 0;
}

prefixSum[tid] = puzzle.solutions.size();
if (args.saveSolution && numSolution > 0) {
#pragma omp barrier
{}
#pragma omp single
{
for (int i = 1; i < numThreads; i++) {
prefixSum[i] += prefixSum[i-1];
}
for (int i = numThreads-1; i > 0; i--) {
prefixSum[i] = prefixSum[i-1];
}
prefixSum[0] = 0;
std::random_device rd;
std::default_random_engine gen = std::default_random_engine(rd());
std::uniform_int_distribution<int> dis(1, numSolution);
rndChoose = dis(gen) - 1;
}
if (rndChoose >= prefixSum[tid] && rndChoose < prefixSum[tid] + puzzle.solutions.size()) {
oneSolution = puzzle.solutions[rndChoose - prefixSum[tid]];
}
}
}
if (args.percent) std::cerr << std::string(70, ' ') << "\r";
std::cout << "number of solutions = " << numSolution << '\n';
if (args.saveSolution) {
std::map<std::pair<int, int>, int> pieceMap;
std::vector<int> numPoly(puzzle.polyominoes.size());
for (int i = 0; i < oneSolution.size(); i++) {
int rowid = oneSolution[i];
DlxRow row = puzzle.dlx.rows[rowid];
int id = row.polyomino;
int polymul = ++numPoly[id];
std::pair<int, int> thispiece = {id, polymul};
pieceMap[thispiece] = rowid;
}
for (auto cp : pieceMap) {
auto pid = cp.first;
int i = cp.second;
DlxRow row = puzzle.dlx.rows[i];
Polyomino piece = puzzle.polyominoes[pid.first];
if (piece.names.empty()) {
std::cout << "piece " << pid.first+1 << " #" << pid.second << ":";
}
else if (pid.second <= piece.names.size()) {
std::cout << "piece " << piece.names[pid.second-1] << ":";
}
else {
std::cout << "piece " << piece.names[0] << " #" << pid.second << ":";
}
Shape sh = puzzle.polyominoes[pid.first].transforms[row.transform];
sh.sortCoords();
for (Coord coord : sh.coords) {
Coord pos = coord + row.position;
std::cout << " (" << pos.x << "," << pos.y << "," << pos.z << "," << pos.w
<< ")";
}
std::cout << '\n';
}
}
std::cout << "solve time=" << tm1.getRunTime() << "ms"<<std::endl;
return 0;
}

int main(int argc, char *argv[]) {
CmdArgs args;
if (!args.parseCmdLine(argc, argv)) return 1;
if (args.help) {
args.showHelp();
return 0;
}
if (args.ver) {
args.showVersion();
return 0;
}
if (args.numThreads > 0) {
omp_set_num_threads(args.numThreads);
}
if (args.parallelLevel == 0 && args.numThreads != 1) {
std::cerr << "Warning: Multithreading is not possible when --parallel-level is 0." << '\n';
std::cerr << "Consider setting --parallel-level to 25% of piece count." << '\n';
}

int lastError = 1;
std::cout << std::fixed;
for (std::string filename : args.filenames) {
#ifdef USE_GPU
extern void ss();
ss();
#endif
lastError = 0;
if (args.info) {
std::cout << "FILE=" << filename << "\n";
}
std::ifstream puzzlefile(filename);
if (!puzzlefile) {
std::cerr << "Error: File \"" << filename << "\" not found."
<< std::endl;
lastError = 1;
continue;
}
Timing tm;
lastError = solveOneFile(args, filename, puzzlefile);
std::cout << "total solve time: " << tm.getRunTime() << "ms\n";
}
if (args.filenames.empty()) {
if (args.info) {
std::cout << "FILE=stdin\n";
}
Timing tm;
lastError = solveOneFile(args, "", std::cin);
std::cout << "solve time: " << tm.getRunTime() << "ms\n";
}
return lastError;
}
