int main (int argc, char *argv[])
{
#pragma nanos instrument declare (myEvent, "This is my event")
sleep(1);
#pragma nanos instrument emit (myEvent, "1")
sleep(1);
#pragma nanos instrument emit (myEvent, "2")
sleep(1);
#pragma nanos instrument emit (myEvent, "3")
sleep(1);
#pragma nanos instrument emit (myEvent, "4")
sleep(1);
#pragma nanos instrument emit (myEvent, "5")
return 0;
}
