#include "define.c"
void kernel(public_struct public, private_struct private)
{
int ei_new;
float * d_in;
int rot_row;
int rot_col;
int in2_rowlow;
int in2_collow;
int ic;
int jc;
int jp1;
int ja1, ja2;
int ip1;
int ia1, ia2;
int ja, jb;
int ia, ib;
float s;
int i;
int j;
int row;
int col;
int ori_row;
int ori_col;
int position;
float sum;
int pos_ori;
float temp;
float temp2;
int location;
int cent;
int tMask_row;
int tMask_col;
float largest_value_current = 0;
float largest_value = 0;
int largest_coordinate_current = 0;
int largest_coordinate = 0;
float fin_max_val = 0;
int fin_max_coo = 0;
int largest_row;
int largest_col;
int offset_row;
int offset_col;
float in_final_sum;
float in_sqr_final_sum;
float mean;
float mean_sqr;
float variance;
float deviation;
float denomT;
int pointer;
int ori_pointer;
int loc_pointer;
int ei_mod;
if (public.frame_no==0)
{
pointer=((private.point_no*public.frames)+public.frame_no);
private.d_tRowLoc[pointer]=private.d_Row[private.point_no];
private.d_tColLoc[pointer]=private.d_Col[private.point_no];
d_in=( & private.d_T[private.in_pointer]);
#pragma loop name kernel#0 
for (col=0; col<public.in_mod_cols; col ++ )
{
#pragma loop name kernel#0#0 
for (row=0; row<public.in_mod_rows; row ++ )
{
ori_row=(((private.d_Row[private.point_no]-25)+row)-1);
ori_col=(((private.d_Col[private.point_no]-25)+col)-1);
ori_pointer=((ori_col*public.frame_rows)+ori_row);
d_in[(col*public.in_mod_rows)+row]=public.d_frame[ori_pointer];
}
}
}
if (public.frame_no!=0)
{
in2_rowlow=(private.d_Row[private.point_no]-public.sSize);
in2_collow=(private.d_Col[private.point_no]-public.sSize);
#pragma loop name kernel#1 
for (col=0; col<public.in2_cols; col ++ )
{
#pragma loop name kernel#1#0 
for (row=0; row<public.in2_rows; row ++ )
{
ori_row=((row+in2_rowlow)-1);
ori_col=((col+in2_collow)-1);
temp=public.d_frame[(ori_col*public.frame_rows)+ori_row];
private.d_in2[(col*public.in2_rows)+row]=temp;
private.d_in2_sqr[(col*public.in2_rows)+row]=(temp*temp);
}
}
d_in=( & private.d_T[private.in_pointer]);
#pragma loop name kernel#2 
for (col=0; col<public.in_mod_cols; col ++ )
{
#pragma loop name kernel#2#0 
for (row=0; row<public.in_mod_rows; row ++ )
{
rot_row=((public.in_mod_rows-1)-row);
rot_col=((public.in_mod_rows-1)-col);
pointer=((rot_col*public.in_mod_rows)+rot_row);
temp=d_in[pointer];
private.d_in_mod[(col*public.in_mod_rows)+row]=temp;
private.d_in_sqr[pointer]=(temp*temp);
}
}
in_final_sum=0;
#pragma loop name kernel#3 
#pragma cetus reduction(+: in_final_sum) 
#pragma cetus parallel 
#pragma omp parallel for reduction(+: in_final_sum)
for (i=0; i<public.in_mod_elem; i ++ )
{
in_final_sum=(in_final_sum+d_in[i]);
}
in_sqr_final_sum=0;
#pragma loop name kernel#4 
#pragma cetus reduction(+: in_sqr_final_sum) 
#pragma cetus parallel 
#pragma omp parallel for reduction(+: in_sqr_final_sum)
for (i=0; i<public.in_mod_elem; i ++ )
{
in_sqr_final_sum=(in_sqr_final_sum+private.d_in_sqr[i]);
}
mean=(in_final_sum/public.in_mod_elem);
mean_sqr=(mean*mean);
variance=((in_sqr_final_sum/public.in_mod_elem)-mean_sqr);
deviation=sqrt(variance);
denomT=(sqrt((float)(public.in_mod_elem-1))*deviation);
#pragma loop name kernel#5 
for (col=1; col<=public.conv_cols; col ++ )
{
j=(col+public.joffset);
jp1=(j+1);
if (public.in2_cols<jp1)
{
ja1=(jp1-public.in2_cols);
}
else
{
ja1=1;
}
if (public.in_mod_cols<j)
{
ja2=public.in_mod_cols;
}
else
{
ja2=j;
}
#pragma loop name kernel#5#0 
for (row=1; row<=public.conv_rows; row ++ )
{
i=(row+public.ioffset);
ip1=(i+1);
if (public.in2_rows<ip1)
{
ia1=(ip1-public.in2_rows);
}
else
{
ia1=1;
}
if (public.in_mod_rows<i)
{
ia2=public.in_mod_rows;
}
else
{
ia2=i;
}
s=0;
#pragma loop name kernel#5#0#0 
#pragma cetus reduction(+: s) 
for (ja=ja1; ja<=ja2; ja ++ )
{
jb=(jp1-ja);
#pragma loop name kernel#5#0#0#0 
#pragma cetus reduction(+: s) 
for (ia=ia1; ia<=ia2; ia ++ )
{
ib=(ip1-ia);
s=(s+(private.d_in_mod[((public.in_mod_rows*(ja-1))+ia)-1]*private.d_in2[((public.in2_rows*(jb-1))+ib)-1]));
}
}
private.d_conv[((col-1)*public.conv_rows)+(row-1)]=s;
}
}
#pragma loop name kernel#6 
for (col=0; col<public.in2_pad_cols; col ++ )
{
#pragma loop name kernel#6#0 
for (row=0; row<public.in2_pad_rows; row ++ )
{
if ((((row>(public.in2_pad_add_rows-1))&&(row<(public.in2_pad_add_rows+public.in2_rows)))&&(col>(public.in2_pad_add_cols-1)))&&(col<(public.in2_pad_add_cols+public.in2_cols)))
{
ori_row=(row-public.in2_pad_add_rows);
ori_col=(col-public.in2_pad_add_cols);
private.d_in2_pad[(col*public.in2_pad_rows)+row]=private.d_in2[(ori_col*public.in2_rows)+ori_row];
}
else
{
private.d_in2_pad[(col*public.in2_pad_rows)+row]=0;
}
}
}
#pragma loop name kernel#7 
for (ei_new=0; ei_new<public.in2_pad_cols; ei_new ++ )
{
pos_ori=(ei_new*public.in2_pad_rows);
sum=0;
#pragma loop name kernel#7#0 
for (position=pos_ori; position<(pos_ori+public.in2_pad_rows); position=(position+1))
{
private.d_in2_pad[position]=(private.d_in2_pad[position]+sum);
sum=private.d_in2_pad[position];
}
}
#pragma loop name kernel#8 
for (col=0; col<public.in2_sub_cols; col ++ )
{
#pragma loop name kernel#8#0 
for (row=0; row<public.in2_sub_rows; row ++ )
{
ori_row=((row+public.in2_pad_cumv_sel_rowlow)-1);
ori_col=((col+public.in2_pad_cumv_sel_collow)-1);
temp=private.d_in2_pad[(ori_col*public.in2_pad_rows)+ori_row];
ori_row=((row+public.in2_pad_cumv_sel2_rowlow)-1);
ori_col=((col+public.in2_pad_cumv_sel2_collow)-1);
temp2=private.d_in2_pad[(ori_col*public.in2_pad_rows)+ori_row];
private.d_in2_sub[(col*public.in2_sub_rows)+row]=(temp-temp2);
}
}
#pragma loop name kernel#9 
for (ei_new=0; ei_new<public.in2_sub_rows; ei_new ++ )
{
pos_ori=ei_new;
sum=0;
#pragma loop name kernel#9#0 
for (position=pos_ori; position<(pos_ori+public.in2_sub_elem); position=(position+public.in2_sub_rows))
{
private.d_in2_sub[position]=(private.d_in2_sub[position]+sum);
sum=private.d_in2_sub[position];
}
}
#pragma loop name kernel#10 
for (col=0; col<public.in2_sub2_sqr_cols; col ++ )
{
#pragma loop name kernel#10#0 
for (row=0; row<public.in2_sub2_sqr_rows; row ++ )
{
ori_row=((row+public.in2_sub_cumh_sel_rowlow)-1);
ori_col=((col+public.in2_sub_cumh_sel_collow)-1);
temp=private.d_in2_sub[(ori_col*public.in2_sub_rows)+ori_row];
ori_row=((row+public.in2_sub_cumh_sel2_rowlow)-1);
ori_col=((col+public.in2_sub_cumh_sel2_collow)-1);
temp2=private.d_in2_sub[(ori_col*public.in2_sub_rows)+ori_row];
temp2=(temp-temp2);
private.d_in2_sub2_sqr[(col*public.in2_sub2_sqr_rows)+row]=(temp2*temp2);
private.d_conv[(col*public.in2_sub2_sqr_rows)+row]=(private.d_conv[(col*public.in2_sub2_sqr_rows)+row]-((temp2*in_final_sum)/public.in_mod_elem));
}
}
#pragma loop name kernel#11 
for (col=0; col<public.in2_pad_cols; col ++ )
{
#pragma loop name kernel#11#0 
for (row=0; row<public.in2_pad_rows; row ++ )
{
if ((((row>(public.in2_pad_add_rows-1))&&(row<(public.in2_pad_add_rows+public.in2_rows)))&&(col>(public.in2_pad_add_cols-1)))&&(col<(public.in2_pad_add_cols+public.in2_cols)))
{
ori_row=(row-public.in2_pad_add_rows);
ori_col=(col-public.in2_pad_add_cols);
private.d_in2_pad[(col*public.in2_pad_rows)+row]=private.d_in2_sqr[(ori_col*public.in2_rows)+ori_row];
}
else
{
private.d_in2_pad[(col*public.in2_pad_rows)+row]=0;
}
}
}
#pragma loop name kernel#12 
for (ei_new=0; ei_new<public.in2_pad_cols; ei_new ++ )
{
pos_ori=(ei_new*public.in2_pad_rows);
sum=0;
#pragma loop name kernel#12#0 
for (position=pos_ori; position<(pos_ori+public.in2_pad_rows); position=(position+1))
{
private.d_in2_pad[position]=(private.d_in2_pad[position]+sum);
sum=private.d_in2_pad[position];
}
}
#pragma loop name kernel#13 
for (col=0; col<public.in2_sub_cols; col ++ )
{
#pragma loop name kernel#13#0 
for (row=0; row<public.in2_sub_rows; row ++ )
{
ori_row=((row+public.in2_pad_cumv_sel_rowlow)-1);
ori_col=((col+public.in2_pad_cumv_sel_collow)-1);
temp=private.d_in2_pad[(ori_col*public.in2_pad_rows)+ori_row];
ori_row=((row+public.in2_pad_cumv_sel2_rowlow)-1);
ori_col=((col+public.in2_pad_cumv_sel2_collow)-1);
temp2=private.d_in2_pad[(ori_col*public.in2_pad_rows)+ori_row];
private.d_in2_sub[(col*public.in2_sub_rows)+row]=(temp-temp2);
}
}
#pragma loop name kernel#14 
for (ei_new=0; ei_new<public.in2_sub_rows; ei_new ++ )
{
pos_ori=ei_new;
sum=0;
#pragma loop name kernel#14#0 
for (position=pos_ori; position<(pos_ori+public.in2_sub_elem); position=(position+public.in2_sub_rows))
{
private.d_in2_sub[position]=(private.d_in2_sub[position]+sum);
sum=private.d_in2_sub[position];
}
}
#pragma loop name kernel#15 
for (col=0; col<public.conv_cols; col ++ )
{
#pragma loop name kernel#15#0 
for (row=0; row<public.conv_rows; row ++ )
{
ori_row=((row+public.in2_sub_cumh_sel_rowlow)-1);
ori_col=((col+public.in2_sub_cumh_sel_collow)-1);
temp=private.d_in2_sub[(ori_col*public.in2_sub_rows)+ori_row];
ori_row=((row+public.in2_sub_cumh_sel2_rowlow)-1);
ori_col=((col+public.in2_sub_cumh_sel2_collow)-1);
temp2=private.d_in2_sub[(ori_col*public.in2_sub_rows)+ori_row];
temp2=(temp-temp2);
temp2=(temp2-(private.d_in2_sub2_sqr[(col*public.conv_rows)+row]/public.in_mod_elem));
if (temp2<0)
{
temp2=0;
}
temp2=sqrt(temp2);
temp2=(denomT*temp2);
private.d_conv[(col*public.conv_rows)+row]=(private.d_conv[(col*public.conv_rows)+row]/temp2);
}
}
cent=((public.sSize+public.tSize)+1);
pointer=((public.frame_no-1)+(private.point_no*public.frames));
tMask_row=(((cent+private.d_tRowLoc[pointer])-private.d_Row[private.point_no])-1);
tMask_col=(((cent+private.d_tColLoc[pointer])-private.d_Col[private.point_no])-1);
#pragma loop name kernel#16 
#pragma cetus parallel 
#pragma omp parallel for
for (ei_new=0; ei_new<public.tMask_elem; ei_new ++ )
{
private.d_tMask[ei_new]=0;
}
private.d_tMask[(tMask_col*public.tMask_rows)+tMask_row]=1;
#pragma loop name kernel#17 
for (col=1; col<=public.mask_conv_cols; col ++ )
{
j=(col+public.mask_conv_joffset);
jp1=(j+1);
if (public.mask_cols<jp1)
{
ja1=(jp1-public.mask_cols);
}
else
{
ja1=1;
}
if (public.tMask_cols<j)
{
ja2=public.tMask_cols;
}
else
{
ja2=j;
}
#pragma loop name kernel#17#0 
for (row=1; row<=public.mask_conv_rows; row ++ )
{
i=(row+public.mask_conv_ioffset);
ip1=(i+1);
if (public.mask_rows<ip1)
{
ia1=(ip1-public.mask_rows);
}
else
{
ia1=1;
}
if (public.tMask_rows<i)
{
ia2=public.tMask_rows;
}
else
{
ia2=i;
}
s=0;
#pragma loop name kernel#17#0#0 
#pragma cetus reduction(+: s) 
for (ja=ja1; ja<=ja2; ja ++ )
{
jb=(jp1-ja);
#pragma loop name kernel#17#0#0#0 
#pragma cetus reduction(+: s) 
for (ia=ia1; ia<=ia2; ia ++ )
{
ib=(ip1-ia);
s=(s+(private.d_tMask[((public.tMask_rows*(ja-1))+ia)-1]*1));
}
}
private.d_mask_conv[((col-1)*public.conv_rows)+(row-1)]=(private.d_conv[((col-1)*public.conv_rows)+(row-1)]*s);
}
}
fin_max_val=0;
fin_max_coo=0;
#pragma loop name kernel#18 
for (i=0; i<public.mask_conv_elem; i ++ )
{
if (private.d_mask_conv[i]>fin_max_val)
{
fin_max_val=private.d_mask_conv[i];
fin_max_coo=i;
}
}
largest_row=(((fin_max_coo+1)%public.mask_conv_rows)-1);
largest_col=((fin_max_coo+1)/public.mask_conv_rows);
if (((fin_max_coo+1)%public.mask_conv_rows)==0)
{
largest_row=(public.mask_conv_rows-1);
largest_col=(largest_col-1);
}
largest_row=(largest_row+1);
largest_col=(largest_col+1);
offset_row=((largest_row-public.in_mod_rows)-(public.sSize-public.tSize));
offset_col=((largest_col-public.in_mod_cols)-(public.sSize-public.tSize));
pointer=((private.point_no*public.frames)+public.frame_no);
private.d_tRowLoc[pointer]=(private.d_Row[private.point_no]+offset_row);
private.d_tColLoc[pointer]=(private.d_Col[private.point_no]+offset_col);
}
if ((public.frame_no!=0)&&((public.frame_no%10)==0))
{
loc_pointer=((private.point_no*public.frames)+public.frame_no);
private.d_Row[private.point_no]=private.d_tRowLoc[loc_pointer];
private.d_Col[private.point_no]=private.d_tColLoc[loc_pointer];
#pragma loop name kernel#19 
for (col=0; col<public.in_mod_cols; col ++ )
{
#pragma loop name kernel#19#0 
for (row=0; row<public.in_mod_rows; row ++ )
{
ori_row=(((private.d_Row[private.point_no]-25)+row)-1);
ori_col=(((private.d_Col[private.point_no]-25)+col)-1);
ori_pointer=((ori_col*public.frame_rows)+ori_row);
d_in[(col*public.in_mod_rows)+row]=((public.alpha*d_in[(col*public.in_mod_rows)+row])+((1.0-public.alpha)*public.d_frame[ori_pointer]));
}
}
}
return ;
}
