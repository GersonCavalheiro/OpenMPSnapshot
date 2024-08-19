
#pragma once


void mult_seq(long *result, long *a, long *b, int height, int width, int widthA);
void mult_seq_speed(long *result, long *a, long *b, int height, int width, int widthA);
void mult_seq_speed_cache(long *result, long *a, long *b, int height, int width, int widthA);


void mult_basic(long *result, long *a, long *b, int height, int width, int widthA);
void mult_basic_threads_size(long *result, long *a, long *b, int height, int width, int widthA, int threads);
void mult_3for(long *result, long *a, long *b, int height, int width, int widthA);
void mult_basic_private(long *result, long *a, long *b, int height, int width, int widthA);
void mult_basic_private_shared(long *result, long *a, long *b, int height, int width, int widthA);
void mult_basic_private_private(long *result, long *a, long *b, int height, int width, int widthA);
void mult_basic_private_switchedloops(long *result, long *a, long *b, int height, int width, int widthA);
void mult_collapse3(long *result, long *a, long *b, int height, int width, int widthA);
void mult_collapse2(long *result, long *a, long *b, int height, int width, int widthA);
void mult_collapse2_private(long *result, long *a, long *b, int height, int width, int widthA);
void mult_collapse2_private_cache(long *result, long *a, long *b, int height, int width, int widthA);
void mult_collapse2_private_cache_threads_size(long *result, long *a, long *b, int height, int width, int widthA, int threads);
void mult_reduction(long *result, long *a, long *b, int height, int width, int widthA);
void mult_reduction_threads_size(long *result, long *a, long *b, int height, int width, int widthA, int threads);
void mult_reduction_inner(long *result, long *a, long *b, int height, int width, int widthA);
void mult_reduction_inner_collapse(long *result, long *a, long *b, int height, int width, int widthA);
void mult_schedule_static(long *result, long *a, long *b, int height, int width, int widthA);
void mult_schedule_static_chunk(long *result, long *a, long *b, int height, int width, int widthA);
void mult_schedule_static_chunk_dynamic(long *result, long *a, long *b, int height, int width, int widthA);
void mult_schedule_dynamic(long *result, long *a, long *b, int height, int width, int widthA);
void mult_schedule_dynamic_chunk(long *result, long *a, long *b, int height, int width, int widthA);
void mult_schedule_guided(long *result, long *a, long *b, int height, int width, int widthA);
void mult_schedule_guided_chunk(long *result, long *a, long *b, int height, int width, int widthA);
void mult_schedule_auto(long *result, long *a, long *b, int height, int width, int widthA);
