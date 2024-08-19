

#include "fuzzyalgorithm.hpp"

FuzzyAlgorithm::FuzzyAlgorithm(QObject* parent)
: QObject(parent)
{
inputMembersA.append(new SigmoidFunction(6.0, -0.6, false));
inputMembersA.append(new GaussFunction(-0.35, 0.075));
inputMembersA.append(new GaussFunction(0.35, 0.075));
inputMembersA.append(new SigmoidFunction(6.0, 0.6, true));

inputMembersB.append(new SigmoidFunction(6.0, -0.6, false));
inputMembersB.append(new GaussFunction(-0.35, 0.075));
inputMembersB.append(new GaussFunction(0.35, 0.075));
inputMembersB.append(new SigmoidFunction(6.0, 0.6, true));

outputMembers.append([] (const auto& V) -> auto
{
return (V[0] && V[4]) || (V[2] && V[7]) || (V[3] && V[6]);
});

outputMembers.append([] (const auto& V) -> auto
{
return (V[0] && V[7]) || (V[1] && V[5]) || (V[3] && V[5]);
});

outputMembers.append([] (const auto& V) -> auto
{
return (V[0] && V[5]) || (V[1] && V[6]) || (V[2] && V[4]);
});

outputMembers.append([] (const auto& V) -> auto
{
return (V[0] && V[6]) || (V[1] && V[4]) || (V[2] && V[5]);
});

outputMembers.append([] (const auto& V) -> auto
{
return (V[2] && V[6]) || (V[3] && V[7]);
});

outputMembers.append([] (const auto& V) -> auto
{
return (V[1] && V[7]) || (V[3] && V[4]);
});

singletonValues = {-0.95, -0.35, -0.10, 0.20, 0.45, 0.90 };
}

FuzzyAlgorithm::~FuzzyAlgorithm(void)
{
for (auto& f : inputMembersA) delete f;
for (auto& f : inputMembersB) delete f;
}

double FuzzyAlgorithm::runAsyncOpenmp(double A, double B) const
{
const int sizeA = inputMembersA.size();
const int sizeB = inputMembersB.size();

const int sizeO = outputMembers.size();
const int sizeI = sizeA + sizeB;

double members = 0.0;
double value = 0.0;

FuzzyVariables vIn;

vIn.resize(sizeI);

#pragma omp parallel for
for (int i = 0; i < sizeA; ++i)
{
vIn[i] = inputMembersA[i]->value(A);
}

#pragma omp parallel for
for (int i = 0; i < sizeB; ++i)
{
vIn[i + sizeA] = inputMembersB[i]->value(B);
}

#pragma omp parallel for reduction(+: value, members)
for (int i = 0; i < sizeO; ++i)
{
members += outputMembers[i](vIn);
value += members * singletonValues[i];
}

return value / members;
}

double FuzzyAlgorithm::runAsyncMapped(double A, double B) const
{
using IN_WRAPPER = FuzzyVariable (*) (const MembershipFunction*, double);
using OUT_WRAPPER = FuzzyVariable (*) (const OutputFunction, const FuzzyVariables&);

static const IN_WRAPPER inWrapper =
[] (const MembershipFunction* f, double p) -> FuzzyVariable
{
return f->value(p);
};

static const OUT_WRAPPER outWrapper =
[] (const OutputFunction f, const FuzzyVariables& p) -> FuzzyVariable
{
return f(p);
};

const int sizeA = inputMembersA.size();
const int sizeB = inputMembersB.size();

const int sizeI = sizeA + sizeB;

double members = 0.0;
double value = 0.0;

FuzzyVariables vIn;
vIn.reserve(sizeI);

QFutureSynchronizer<FuzzyVariable> inSynchronizer;

inSynchronizer.addFuture(QtConcurrent::mapped(inputMembersA, boost::bind(inWrapper, _1, A)));
inSynchronizer.addFuture(QtConcurrent::mapped(inputMembersB, boost::bind(inWrapper, _1, B)));

for (const auto& v : inSynchronizer.futures()) vIn.append(v.results().toVector());

const auto outMembers = QtConcurrent::blockingMapped(outputMembers, boost::bind(outWrapper, _1, vIn));

int i(0); for (const auto& res : outMembers)
{
value += res * singletonValues[i++];
members += res;
}

return value / members;
}

double FuzzyAlgorithm::runAsyncRun(double A, double B) const
{
const int sizeA = inputMembersA.size();
const int sizeB = inputMembersB.size();

const int sizeI = sizeA + sizeB;

double members = 0.0;
double value = 0.0;

FuzzyVariables vIn;
vIn.reserve(sizeI);

QFutureSynchronizer<FuzzyVariable> inSynchronizer;
QFutureSynchronizer<FuzzyVariable> outSynchronizer;

for (const auto& f : inputMembersA) inSynchronizer.addFuture(QtConcurrent::run(f, &MembershipFunction::value, A));
for (const auto& f : inputMembersB) inSynchronizer.addFuture(QtConcurrent::run(f, &MembershipFunction::value, B));

for (const auto& v : inSynchronizer.futures()) vIn.append(v.result());

for (const auto& f : outputMembers) outSynchronizer.addFuture(QtConcurrent::run(f, vIn));

int i(0); for (const auto& res : outSynchronizer.futures())
{
value += res.result() * singletonValues[i++];
members += res.result();
}

return value / members;
}

double FuzzyAlgorithm::runSync(double A, double B) const
{
const int sizeA = inputMembersA.size();
const int sizeB = inputMembersB.size();

const int sizeI = sizeA + sizeB;

double members = 0.0;
double value = 0.0;

FuzzyVariables vIn;
vIn.reserve(sizeI);

for (const auto& f : inputMembersA) vIn.append(f->value(A));
for (const auto& f : inputMembersB) vIn.append(f->value(B));

for (int i = 0; i < outputMembers.size(); ++i)
{
const double res = outputMembers[i](vIn);

value += res * singletonValues[i];
members += res;
}

return value / members;
}

void FuzzyAlgorithm::computeRequest(double A, double B)
{
QMutexLocker Locker(&threadSynchronizer);

emit computeFinished(runSync(A, B));
}
