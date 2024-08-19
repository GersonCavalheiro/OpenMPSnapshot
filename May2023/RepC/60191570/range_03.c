#include <stdio.h>
int HasNextBurst(FILE *fp){
int status = 0; 
char line[256];
int i;
int HasEqual=0; 
int EndOfFile =0;
while ((!HasEqual)&&(!EndOfFile)) {
HasEqual= 0;
i=0;
while (i<255) {
line[i]=fgetc(fp);
if ((int)(line[i])) {
EndOfFile=1;
break;
}
if (line[i]=='=')
HasEqual=1;
}
line[i+1]='\0';
}
#pragma analysis_check assert range(HasEqual:0:1:0; EndOfFile:0:1:0)
if (HasEqual) {
status=1;
}
if (EndOfFile)
;
#pragma analysis_check assert range(status:0:1:0)
return status;
}
