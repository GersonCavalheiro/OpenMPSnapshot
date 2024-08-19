using System;
using System.Collections.Generic;
#pragma warning disable CS0168
public class Class1
{
static void Main(string[] args)
{
const int ALLOC_SIZE = 1048576;
List<byte[]> allocations = new List<byte[]>();
int i = 0;
try
{
for (i = 1; ; i++)
allocations.Add(new byte[ALLOC_SIZE]);
}
catch (OutOfMemoryException e)
{
Console.Write("Exception: {2}\n{3}\nManaged to perform {0} allocations of {1} bytes each!\r", i, ALLOC_SIZE, e.Message, e.StackTrace);
}
catch (OverflowException e)
{
Console.WriteLine("Exception: {0}\n{1}", e.Message, e.StackTrace);
}
finally
{
Console.Write("\nPress Enter: ");
Console.ReadLine();
}
}
}
