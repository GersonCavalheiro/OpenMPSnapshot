#include "dem/tests/CollisionTest.h"


#include "tarch/compiler/CompilerSpecificSettings.h"
#include "tarch/tests/TestCaseFactory.h"



registerTest(dem::tests::CollisionTest)


#ifdef UseTestSpecificCompilerSettings
#pragma optimize("",off)
#endif


dem::tests::CollisionTest::CollisionTest():
tarch::tests::TestCase( "dem::tests::CollisionTest" ) {
}


dem::tests::CollisionTest::~CollisionTest() {
}


void dem::tests::CollisionTest::run() {
testMethod( testTwoParallelTriangles0 );
testMethod( testTwoParallelTriangles1 );
}


void dem::tests::CollisionTest::testTwoParallelTriangles0() {

}



void dem::tests::CollisionTest::testTwoParallelTriangles1() {

}


#ifdef UseTestSpecificCompilerSettings
#pragma optimize("",on)
#endif
