#pragma once

class InstrumentalComponentTest : public BaseComponentTest {
public:
InstrumentalComponentTest() : BaseComponentTest() {}

static void testExecutionTime() {
std::cout << "I) Execution time for parallel:\n";
{
LOG_DURATION("1000 N, 4 threadNum")

ParallelInstrumental pi(1000, 4);
matr a = pi.createThirdDiagMatrI();
}

{
LOG_DURATION("10000 N, 1 threadNum")

ParallelInstrumental pi(1000, 1);
matr a = pi.createThirdDiagMatrI();
}

}

static void testEnteredData() {
std::cout << "II) Checking the correctness of the entered data:\n";

{
ParallelInstrumental pi(16, 4);
ASSERT(pi.checkData())
}

{
ParallelInstrumental pi(1600, 10);
ASSERT(pi.checkData())
}

{
ParallelInstrumental pi(161, 4);
ASSERT(!pi.checkData())
}

{
ParallelInstrumental pi(4, 4);
ASSERT(!pi.checkData())
}

{
ParallelInstrumental pi(16, 3);
ASSERT(!pi.checkData())
}
}

static void execute() {
std::vector<std::function<void()>> tests = {
[]() { InstrumentalComponentTest::testExecutionTime(); },
[]() { InstrumentalComponentTest::testEnteredData(); }
};

BaseComponentTest::execute(tests, "Instrumental Component Test");
}
};