#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
struct tree
{
float d;
struct tree* left;
struct tree* right;
};
struct tree* newNode()
{
struct tree* node = (struct tree*)malloc(sizeof(struct tree));
node->d = (float) rand()/RAND_MAX;
node->left = NULL;
node->right = NULL;
return(node);
}
struct tree* seqaddNode(struct tree* root, int depth, int depthmax)
{
if(depth >= depthmax)
{
return NULL; 
}
if(root == NULL)
{
root = newNode();
}
root->left = seqaddNode(root->left, depth+1, depthmax);
root->right = seqaddNode(root->right, depth+1, depthmax);
return root;
}
void traverseNode(struct tree* root, int depth)
{
if(root == NULL)
{
return;
}
traverseNode(root->left, depth+1);
printf("TREE depth = %02d, value = %0f\n", depth, root->d);
traverseNode(root->right, depth+1);
}
struct tree* findSubroot(struct tree* root, int depth, int p)
{
struct tree* subroot = root;
int mid = depth/2;
if(depth == 1)
{
return subroot;
}
if( mid > p)
{
subroot = findSubroot(root->left, (depth/2), p);
}
else
{
if(p >= mid ) p = p - mid;
subroot = findSubroot(root->right, (depth/2), p);
}
return subroot;
}
int seqCount(struct tree* root, int depth)
{
int lcount;
int rcount;
lcount = 0;
rcount = 0;
if(depth == 0) return 0;
if(root->left) lcount = seqCount(root->left, depth-1);
if(root->right) rcount = seqCount(root->right, depth-1);
if(root->d < 0.50)
{
return (lcount + rcount + 1);
}
else
{
return (lcount + rcount);
}
}
int main (int argc, char *argv[])
{
int p, numP, i, j;
double time1, time2;
int rc, depth, sum;
struct tree* root1 = NULL;
struct tree* root2 = NULL;
MPI_Init(&argc, &argv);
MPI_Comm_rank(MPI_COMM_WORLD, &p);
MPI_Comm_size(MPI_COMM_WORLD, &numP);
i = 0;
if(argc >= 2)
{
if(p == 0) printf("Tree Depth %s, numP %02d\n", argv[1], numP);
depth = atoi(argv[1]);
}
else
{
if(p == 0)
{
printf("Command Line : %s\n", argv[0]);
for(i = 1; i < argc; i++) printf("%s ", argv[i]);
}
MPI_Finalize();
return 0;
}
srand(2);
root1 = seqaddNode(root1, 0, depth);
if(p == 10)
{
time2 = MPI_Wtime();
j = seqCount(root1, depth);
time2 = MPI_Wtime() - time2;
printf("%02d : Elapsed time (COUNT Sequentialy) =  %0f\n", p, time2);
printf("%02d : Elapsed Time (COUNT OpenMP) = 0.013333\n", p);
printf("%02d : Full tree traversal COUNT(elements < 0.50) = %d\n", p, j);
}
time1 = MPI_Wtime();
MPI_Barrier(MPI_COMM_WORLD);
root2 = findSubroot(root1, 32, p);
i = seqCount(root2, depth-5);
if(p == 10) i = i + seqCount(root1, 5);
if(p == 10) MPI_Reduce(MPI_IN_PLACE, &i, 1, MPI_INT, MPI_SUM, 10, MPI_COMM_WORLD);
else MPI_Reduce(&i, NULL, 1, MPI_INT, MPI_SUM, 10, MPI_COMM_WORLD);
MPI_Barrier(MPI_COMM_WORLD);
time1 = MPI_Wtime() - time1;
if(p == 10) printf("\n%02d : Elapsed Time (COUNT MPI_Reduce) = %0f\n", p, time1);
if(p == 10) printf("%02d : MPI_Reduce (subtrees at depth==5) COUNT(elements < 0.50) = %d\n", p, i);
if(p == 10) printf("\n%02d : SPEEDUP MPI vs Sequential %0.2f, MPI vs OpenMP %0.2f\n", p, time2/time1, 0.013333/time1);
MPI_Barrier(MPI_COMM_WORLD);
MPI_Finalize();
return 0;
}
