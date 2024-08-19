


#include <catch2/catch_tag_alias_autoregistrar.hpp>
#include <catch2/reporters/catch_reporter_event_listener.hpp>
#include <catch2/internal/catch_enforce.hpp>
#include <catch2/catch_test_case_info.hpp>
#include <catch2/reporters/catch_reporter_registrars.hpp>


CATCH_REGISTER_TAG_ALIAS("[@nhf]", "[failing]~[.]")
CATCH_REGISTER_TAG_ALIAS("[@tricky]", "[tricky]~[.]")

#ifdef __clang__
#   pragma clang diagnostic ignored "-Wpadded"
#   pragma clang diagnostic ignored "-Wweak-vtables"
#   pragma clang diagnostic ignored "-Wc++98-compat"
#endif


class ValidatingTestListener : public Catch::EventListenerBase {
struct EventCounter {
int starting = 0;
int ended = 0;

bool hasActiveEvent() const {
return starting > ended;
}
bool hasSingleActiveEvent() const {
return starting - 1 == ended;
}
bool allEventsEnded() const {
return starting == ended;
}
};

public:
static std::string getDescription() {
return "Validates ordering of Catch2's listener events";
}

ValidatingTestListener(Catch::IConfig const* config) :
EventListenerBase(config) {
m_preferences.shouldReportAllAssertions = true;
}

void testRunStarting( Catch::TestRunInfo const& ) override {
CATCH_ENFORCE( m_testRunCounter.starting == 0,
"Test run can only start once" );
++m_testRunCounter.starting;
}
void testCaseStarting(Catch::TestCaseInfo const&) override {
CATCH_ENFORCE( m_testRunCounter.hasActiveEvent(),
"Test case can only be started if the test run has already started" );
CATCH_ENFORCE( m_testCaseCounter.allEventsEnded(),
"Test case cannot start if there is an unfinished one" );

++m_testCaseCounter.starting;

m_lastSeenPartNumber = uint64_t(-1);
}

void testCasePartialStarting(Catch::TestCaseInfo const&,
uint64_t partNumber) override {
CATCH_ENFORCE( m_testCaseCounter.hasSingleActiveEvent(),
"Test case can only be partially started if the test case has fully started already" );
CATCH_ENFORCE( m_lastSeenPartNumber + 1 == partNumber,
"Partial test case started out of order" );

++m_testCasePartialCounter.starting;
m_lastSeenPartNumber = partNumber;
}

void sectionStarting(Catch::SectionInfo const&) override {
CATCH_ENFORCE( m_testCaseCounter.hasSingleActiveEvent(),
"Section can only start in a test case" );
CATCH_ENFORCE( m_testCasePartialCounter.hasSingleActiveEvent(),
"Section can only start in a test case" );

++m_sectionCounter.starting;
}

void assertionStarting(Catch::AssertionInfo const&) override {
CATCH_ENFORCE( m_testCaseCounter.hasSingleActiveEvent(),
"Assertion can only start if test case is started" );

++m_assertionCounter.starting;
}
void assertionEnded(Catch::AssertionStats const&) override {
++m_assertionCounter.ended;
}

void sectionEnded(Catch::SectionStats const&) override {
CATCH_ENFORCE( m_sectionCounter.hasActiveEvent(),
"Section ended without corresponding start" );

++m_sectionCounter.ended;
}


void testCasePartialEnded(Catch::TestCaseStats const&,
uint64_t partNumber) override {
CATCH_ENFORCE( m_lastSeenPartNumber == partNumber,
"Partial test case ended out of order" );
CATCH_ENFORCE( m_testCasePartialCounter.hasSingleActiveEvent(),
"Partial test case ended without corresponding start" );
CATCH_ENFORCE( m_sectionCounter.allEventsEnded(),
"Partial test case ended with unbalanced sections" );

++m_testCasePartialCounter.ended;
}


void testCaseEnded(Catch::TestCaseStats const&) override {
CATCH_ENFORCE( m_testCaseCounter.hasSingleActiveEvent(),
"Test case end is not matched with test case start" );
CATCH_ENFORCE( m_testCasePartialCounter.allEventsEnded(),
"A partial test case has not ended" );
CATCH_ENFORCE( m_sectionCounter.allEventsEnded(),
"Test case ended with unbalanced sections" );


++m_testCaseCounter.ended;
}
void testRunEnded( Catch::TestRunStats const& ) override {
CATCH_ENFORCE( m_testRunCounter.hasSingleActiveEvent(),
"Test run end is not matched with test run start" );
CATCH_ENFORCE( m_testRunCounter.ended == 0,
"Test run can only end once" );

++m_testRunCounter.ended;
}

~ValidatingTestListener() override;

private:
EventCounter m_testRunCounter;
EventCounter m_testCaseCounter;
EventCounter m_testCasePartialCounter;
uint64_t m_lastSeenPartNumber = 0;
EventCounter m_sectionCounter;
EventCounter m_assertionCounter;
};


ValidatingTestListener::~ValidatingTestListener() {

CATCH_ENFORCE( m_testRunCounter.ended < 2,
"Test run should be started at most once" );
CATCH_ENFORCE( m_testRunCounter.allEventsEnded(),
"The test run has not finished" );
CATCH_ENFORCE( m_testCaseCounter.allEventsEnded(),
"A test case did not finish" );

}

CATCH_REGISTER_LISTENER( ValidatingTestListener )
