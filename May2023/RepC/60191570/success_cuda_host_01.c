#pragma oss task device(cuda)
void outlineDeviceTask() {}
int main() {
#pragma oss task device(cuda)
{}
outlineDeviceTask();
#pragma oss taskwait
return 0;
}
