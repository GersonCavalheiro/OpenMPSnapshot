

#include<iostream>
#include<omp.h>
using namespace std;

int k=0;

class array
{
public:

int partition(int arr[], int low_index, int high_index)
{
int i, j, temp, key;
key = arr[low_index];
i= low_index + 1;
j= high_index;
while(1)
{
while(i < high_index && key >= arr[i])
i++;
while(key < arr[j])
j--;
if(i < j)
{
temp = arr[i];
arr[i] = arr[j];
arr[j] = temp;
}
else
{
temp= arr[low_index];
arr[low_index] = arr[j];
arr[j]= temp;
return(j);
}
}
}

void quicksort(int arr[], int low_index, int high_index)
{
int j;

if(low_index < high_index)
{
j = partition(arr, low_index, high_index);
cout<<"Pivot element with index "<<j<<" has been found out by thread "<<k<<"\n";

#pragma omp parallel sections
{
#pragma omp section
{
k=k+1;
quicksort(arr, low_index, j - 1);
}

#pragma omp section
{
k=k+1;
quicksort(arr, j + 1, high_index);
}

}
}
}

};

int main()
{
array a;

int arr[100];
int n,i;

cout<<"Enter the value of n\n";
cin>>n;
cout<<"Enter the "<<n<<" number of elements \n";

for(i=0;i<n;i++)
{
cin>>arr[i];
}

a.quicksort(arr, 0, n - 1);

cout<<"Elements of array after sorting \n";
for(i=0;i<n;i++)
{
cout<<arr[i]<<"\t";
}

cout<<"\n";
} 



