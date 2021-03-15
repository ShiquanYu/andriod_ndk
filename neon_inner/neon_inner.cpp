#include <iostream>
#include <arm_neon.h>
#include <arm_fp16.h>
#include <sys/time.h>

using namespace std;

float nn_inner(const float *buf1, const float *buf2, const size_t len)
{
    float inner_product = 0.0f;

    if (!buf1 || !buf2 || len == 0)
        return 0.f;

    for (size_t i = 0; i < len; ++i)
    {
        inner_product += buf1[i] * buf2[i];
    }

    return inner_product;
}

float nn_inner_neon(float *buf1, float *buf2, const size_t len)
{
    float inner_product = 0.0f;
    float tmp[4];
    float32x4_t fsum = vdupq_n_f32(0);
    size_t count = len / 4;
    size_t rest = len % 4;

    if (!buf1 || !buf2 || len == 0)
        return 0.f;

    while (count--)
    {
        float32x4_t fbuf1 = vld1q_f32(buf1);
        float32x4_t fbuf2 = vld1q_f32(buf2);
        float32x4_t fmul = vmulq_f32(fbuf1, fbuf2);
        fsum = vaddq_f32(fsum, fmul);
        buf1 += 4;
        buf2 += 4;
    }
    vst1q_f32(tmp, fsum);
    for (size_t i = 0; i < 4; ++i)
    {
        inner_product += tmp[i];
    }
    if (rest)
    {
        float32x4_t fbuf1 = vld1q_f32(buf1);
        float32x4_t fbuf2 = vld1q_f32(buf2);
        float32x4_t fmul = vmulq_f32(fbuf1, fbuf2);
        vst1q_f32(tmp, fmul);
        for (size_t i = 0; i < rest; ++i)
        {
            inner_product += tmp[i];
        }
    }

    return inner_product;
}

int main(int argc, char const *argv[])
{
#define FEAT_LEN 512
    float feat1[FEAT_LEN] = {0};
    float feat2[FEAT_LEN] = {0};
    for (size_t i = 0; i < FEAT_LEN; i++)
    {
        feat1[i] = rand() / double(RAND_MAX);
        feat2[i] = rand() / double(RAND_MAX);
    }

    size_t loop_time = 100000;
    struct timeval tv_start;
    struct timeval tv_end;
    printf("loop %zu times.\n", loop_time);
    gettimeofday(&tv_start, NULL);
    for (size_t i = 0; i < loop_time; ++i)
    {
        nn_inner(feat1, feat2, FEAT_LEN);
    }
    gettimeofday(&tv_end, NULL);
    int elasped_time = (tv_end.tv_sec - tv_start.tv_sec) * 1000 + (tv_end.tv_usec - tv_start.tv_usec) / 1000;
    printf("time normal: %dms\n", elasped_time);

    gettimeofday(&tv_start, NULL);
    for (size_t i = 0; i < loop_time; ++i)
    {
        nn_inner_neon(feat1, feat2, FEAT_LEN);
    }
    gettimeofday(&tv_end, NULL);
    elasped_time = (tv_end.tv_sec - tv_start.tv_sec) * 1000 + (tv_end.tv_usec - tv_start.tv_usec) / 1000;
    printf("time neon: %dms\n", elasped_time);

    printf("inner normal: %f\n", nn_inner(feat1, feat2, FEAT_LEN));
    printf("inner neon: %f\n", nn_inner_neon(feat1, feat2, FEAT_LEN));


    return 0;
}
