#pragma once

#include <arm_neon.h>
#include <nnpack/hwinfo.h>

static inline void neon_transpose4x4_inplace_f32(
    float32x4_t row0[static restrict 1],
    float32x4_t row1[static restrict 1],
    float32x4_t row2[static restrict 1],
    float32x4_t row3[static restrict 1])
{
    /*
     * row0 = ( x00 x01 x02 x03 )
     * row1 = ( x10 x11 x12 x13 )
     * row2 = ( x20 x21 x22 x23 )
     * row3 = ( x30 x31 x32 x33 )
     */

    /*
     * row01 = ( x00 x10 x02 x12 ), ( x01 x11 x03, x13 )
     * row23 = ( x20 x30 x22 x32 ), ( x21 x31 x23, x33 )
     */
    float32x4x2_t row01 = vtrnq_f32(*row0, *row1);
    float32x4x2_t row23 = vtrnq_f32(*row2, *row3);

    /*
     * row0 = ( x00 x10 x20 x30 )
     * row1 = ( x01 x11 x21 x31 )
     * row2 = ( x02 x12 x22 x32 )
     * row3 = ( x03 x13 x23 x33 )
     */
    *row0 = vcombine_f32(vget_low_f32(row01.val[0]), vget_low_f32(row23.val[0]));
    *row1 = vcombine_f32(vget_low_f32(row01.val[1]), vget_low_f32(row23.val[1]));
    *row2 = vcombine_f32(vget_high_f32(row01.val[0]), vget_high_f32(row23.val[0]));
    *row3 = vcombine_f32(vget_high_f32(row01.val[1]), vget_high_f32(row23.val[1]));
}

static inline void neon_transpose4x4_inplace_f321_intertile(
    svfloat32_t *row0,
    svfloat32_t *row1,
    svfloat32_t *row2,
    svfloat32_t *row3)
{
int simd_width = svcntw();//nnp_hwinfo.sve_simd_width;//nnp_hwinfo.simd_width;
        int interchannels  = svcntw()/4;//nnp_hwinfo.globalinterchannels;
// int simd_width=16;//nnp_hwinfo.simd_width;
//const int interchannels  = simd_width/4;
 for(int i1=0;i1<simd_width;i1+=svcntw())
    {
                 svbool_t pg = svwhilelt_b32(i1,simd_width);
    svfloat32x4_t row0123 = svcreate4_f32(*row0, *row1, *row2, *row3);

     float buff_tmp[4*simd_width];
     svst4(pg, &buff_tmp[0], row0123);

    int index1_host[simd_width];
    for(int i=0;i<interchannels;i++)
    {
         for(int j=0;j<4;j++)
          {
                  index1_host[j+i*4] = j+(16*i);
          }
   }
    svint32_t index1, index2, index3, index4;
    index1 = svld1_s32(pg, &index1_host[0]); 
    index2 = svadd_s32_m(pg, index1, svdup_s32(4));
    index3 = svadd_s32_m(pg, index2, svdup_s32(4));
    index4 = svadd_s32_m(pg, index3, svdup_s32(4));
   
      *row0 = svld1_gather_s32index_f32(pg, &buff_tmp[0],index1);
     *row1 = svld1_gather_s32index_f32(pg, &buff_tmp[0],index2);
     *row2 = svld1_gather_s32index_f32(pg, &buff_tmp[0],index3);
     *row3 = svld1_gather_s32index_f32(pg, &buff_tmp[0],index4);

 }
}
static inline void neon_transpose4x4_inplace_f321(
    svfloat32_t *row0,
    svfloat32_t *row1,
    svfloat32_t *row2,
    svfloat32_t *row3)
{
 int simd_width=nnp_hwinfo.simd_width;
 for(int i1=0;i1<simd_width;i1+=svcntw())
                {
                 svbool_t pg = svwhilelt_b32(i1,simd_width);
    svfloat32x4_t row0123 = svcreate4_f32(*row0, *row1, *row2, *row3);

     float buff_tmp[4*simd_width];
     svst4(pg, &buff_tmp[0], row0123);
     *row0 = svld1(pg, &buff_tmp[0]);
     *row1 = svld1(pg, &buff_tmp[simd_width]);
     *row2 = svld1(pg, &buff_tmp[simd_width*2]);
     *row3 = svld1(pg, &buff_tmp[3*simd_width]);  }
}
