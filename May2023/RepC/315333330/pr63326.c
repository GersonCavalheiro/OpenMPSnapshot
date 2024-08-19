void
f1 (int x)
{
int i;
if (x)
#pragma omp barrier				
;
if (x)
#pragma omp flush				
;
if (x)
#pragma omp taskwait			
;
if (x)
#pragma omp taskyield			
;
#pragma omp parallel
{
if (x)
#pragma omp cancel parallel		
;
}
#pragma omp parallel
{
if (x)
#pragma omp cancellation point parallel	
;
}
#pragma omp for ordered(1)
for (i = 0; i < 16; i++)
{
if (x)
#pragma omp ordered depend(source)	
;
if (x)
#pragma omp ordered depend(sink: i-1)	
;
}
if (x)
#pragma omp target enter data map(to:i)	
;
if (x)
#pragma omp target update to(i)		
;
if (x)
#pragma omp target exit data map(from:i)	
;
}
void
f2 (int x)
{
int i;
while (x)
#pragma omp barrier				
;
while (x)
#pragma omp flush				
;
while (x)
#pragma omp taskwait			
;
while (x)
#pragma omp taskyield			
;
#pragma omp parallel
{
while (x)
#pragma omp cancel parallel		
;
}
#pragma omp parallel
{
while (x)
#pragma omp cancellation point parallel	
;
}
#pragma omp for ordered(1)
for (i = 0; i < 16; i++)
{
while (x)
#pragma omp ordered depend(source)	
;
while (x)
#pragma omp ordered depend(sink: i-1)	
;
}
while (x)
#pragma omp target enter data map(to:i)	
;
while (x)
#pragma omp target update to(i)		
;
while (x)
#pragma omp target exit data map(from:i)	
;
}
void
f3 (int x)
{
int i;
for (x = 0; x < 10; x++)
#pragma omp barrier				
;
for (x = 0; x < 10; x++)
#pragma omp flush				
;
for (x = 0; x < 10; x++)
#pragma omp taskwait			
;
for (x = 0; x < 10; x++)
#pragma omp taskyield			
;
#pragma omp parallel
{
for (x = 0; x < 10; x++)
#pragma omp cancel parallel		
;
}
#pragma omp parallel
{
for (x = 0; x < 10; x++)
#pragma omp cancellation point parallel	
;
}
#pragma omp for ordered(1)
for (i = 0; i < 16; i++)
{
for (x = 0; x < 10; x++)
#pragma omp ordered depend(source)	
;
for (x = 0; x < 10; x++)
#pragma omp ordered depend(sink: i-1)	
;
}
for (x = 0; x < 10; x++)
#pragma omp target enter data map(to:i)	
;
for (x = 0; x < 10; x++)
#pragma omp target update to(i)		
;
for (x = 0; x < 10; x++)
#pragma omp target exit data map(from:i)	
;
}
void
f4 (int x)
{
int i;
{
do
#pragma omp barrier			
while (0); 
} 
{
do
#pragma omp flush				
while (0); 
} 
{
do
#pragma omp taskwait			
while (0); 
} 
{
do
#pragma omp taskyield			
while (0); 
} 
#pragma omp parallel
{
do
#pragma omp cancel parallel		
while (0); 
} 
#pragma omp parallel
{
do
#pragma omp cancellation point parallel	
while (0); 
} 
#pragma omp for ordered(1)
for (i = 0; i < 16; i++)
{
{
do
#pragma omp ordered depend(source)	
while (0); 
} 
{
do
#pragma omp ordered depend(sink: i-1)	
while (0); 
} 
}
{
do
#pragma omp target enter data map(to:i)	
while (0); 
} 
{
do
#pragma omp target update to(i)		
while (0); 
} 
{
do
#pragma omp target exit data map(from:i)	
while (0); 
} 
}
void
f5 (int x)
{
int i;
switch (x)
#pragma omp barrier				
;
switch (x)
#pragma omp flush				
;
switch (x)
#pragma omp taskwait			
;
switch (x)
#pragma omp taskyield			
;
#pragma omp parallel
{
switch (x)
#pragma omp cancel parallel		
;
}
#pragma omp parallel
{
switch (x)
#pragma omp cancellation point parallel	
;
}
#pragma omp for ordered(1)
for (i = 0; i < 16; i++)
{
switch (x)
#pragma omp ordered depend(source)	
;
switch (x)
#pragma omp ordered depend(sink: i-1)	
;
}
switch (x)
#pragma omp target enter data map(to:i)	
;
switch (x)
#pragma omp target update to(i)		
;
switch (x)
#pragma omp target exit data map(from:i)	
;
}
void
f6 (int x)
{
int i;
switch (x)
{
case 1:
#pragma omp barrier			
;
}
switch (x)
{
case 1:
#pragma omp flush				
;
}
switch (x)
{
case 1:
#pragma omp taskwait			
;
}
switch (x)
{
case 1:
#pragma omp taskyield			
;
}
#pragma omp parallel
{
switch (x)
{
case 1:
#pragma omp cancel parallel		
;
}
}
#pragma omp parallel
{
switch (x)
{
case 1:
#pragma omp cancellation point parallel	
;
}
}
#pragma omp for ordered(1)
for (i = 0; i < 16; i++)
{
switch (x)
{
case 1:
#pragma omp ordered depend(source)	
;
}
switch (x)
{
case 1:
#pragma omp ordered depend(sink: i-1)	
;
}
}
switch (x)
{
case 1:
#pragma omp target enter data map(to:i)	
;
}
switch (x)
{
case 1:
#pragma omp target update to(i)		
;
}
switch (x)
{
case 1:
#pragma omp target exit data map(from:i)	
;
}
}
void
f7 (int x)
{
int i;
switch (x)
{
default:
#pragma omp barrier			
;
}
switch (x)
{
default:
#pragma omp flush				
;
}
switch (x)
{
default:
#pragma omp taskwait			
;
}
switch (x)
{
default:
#pragma omp taskyield			
;
}
#pragma omp parallel
{
switch (x)
{
default:
#pragma omp cancel parallel		
;
}
}
#pragma omp parallel
{
switch (x)
{
default:
#pragma omp cancellation point parallel	
;
}
}
#pragma omp for ordered(1)
for (i = 0; i < 16; i++)
{
switch (x)
{
default:
#pragma omp ordered depend(source)	
;
}
switch (x)
{
default:
#pragma omp ordered depend(sink: i-1)	
;
}
}
switch (x)
{
default:
#pragma omp target enter data map(to:i)	
;
}
switch (x)
{
default:
#pragma omp target update to(i)		
;
}
switch (x)
{
default:
#pragma omp target exit data map(from:i)	
;
}
}
void
f8 (int x)
{
int i;
lab1:
#pragma omp barrier				
;
lab2:
#pragma omp flush				
;
lab3:
#pragma omp taskwait			
;
lab4:
#pragma omp taskyield			
;
#pragma omp parallel
{
lab5:
#pragma omp cancel parallel		
;
}
#pragma omp parallel
{
lab6:
#pragma omp cancellation point parallel	
;
}
#pragma omp for ordered(1)
for (i = 0; i < 16; i++)
{
lab7:
#pragma omp ordered depend(source)	
;
lab8:
#pragma omp ordered depend(sink: i-1)	
;
}
lab9:
#pragma omp target enter data map(to:i)	
;
lab10:
#pragma omp target update to(i)		
;
lab11:
#pragma omp target exit data map(from:i)	
;
}
