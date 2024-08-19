#ifdef __cplusplus
extern "C" {
#endif
struct List_Header
{
Node_Id first;
Node_Id last;
Node_Id parent;
};
extern struct List_Header *List_Headers_Ptr;
extern Node_Id *Next_Node_Ptr;
extern Node_Id *Prev_Node_Ptr;
static Node_Id First (List_Id);
INLINE Node_Id
First (List_Id List)
{
return List_Headers_Ptr[List - First_List_Id].first;
}
#define First_Non_Pragma nlists__first_non_pragma
extern Node_Id First_Non_Pragma (Node_Id);
static Node_Id Last (List_Id);
INLINE Node_Id
Last (List_Id List)
{
return List_Headers_Ptr[List - First_List_Id].last;
}
#define First_Non_Pragma nlists__first_non_pragma
extern Node_Id First_Non_Pragma (List_Id);
static Node_Id Next (Node_Id);
INLINE Node_Id
Next (Node_Id Node)
{
return Next_Node_Ptr[Node - First_Node_Id];
}
#define Next_Non_Pragma nlists__next_non_pragma
extern Node_Id Next_Non_Pragma (List_Id);
static Node_Id Prev (Node_Id);
INLINE Node_Id
Prev (Node_Id Node)
{
return Prev_Node_Ptr[Node - First_Node_Id];
}
#define Prev_Non_Pragma nlists__prev_non_pragma
extern Node_Id Prev_Non_Pragma		(Node_Id);
static Boolean Is_Empty_List		(List_Id);
static Boolean Is_Non_Empty_List	(List_Id);
static Boolean Is_List_Member		(Node_Id);
static List_Id List_Containing		(Node_Id);
INLINE Boolean
Is_Empty_List (List_Id Id)
{
return (First (Id) == Empty);
}
INLINE Boolean
Is_Non_Empty_List (List_Id Id)
{
return (Present (Id) && First (Id) != Empty);
}
INLINE Boolean
Is_List_Member (Node_Id Node)
{
return Nodes_Ptr[Node - First_Node_Id].U.K.in_list;
}
INLINE List_Id
List_Containing (Node_Id Node)
{
return Nodes_Ptr[Node - First_Node_Id].V.NX.link;
}
#ifdef __cplusplus
}
#endif
