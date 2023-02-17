#pragma once
#include<arm_sve.h>
#include <stdbool.h>
#include <nnpack/hwinfo.h>
#include <nnpack/arm_neon.h>
#include <nnpack/macros.h>

static  void winograd_f6k3_input_transform_intertile(
        const svfloat32_t d0,
        const svfloat32_t d1,
        const svfloat32_t d2,
        const svfloat32_t d3,
        const svfloat32_t d4,
        const svfloat32_t d5,
        const svfloat32_t d6,
        const svfloat32_t d7,
        svfloat32_t *transform0,
        svfloat32_t *transform1,
        svfloat32_t *transform2,
        svfloat32_t *transform3,
        svfloat32_t *transform4,
        svfloat32_t *transform5,
        svfloat32_t *transform6,
        svfloat32_t *transform7)
{
 int simd_width = nnp_hwinfo.sve_simd_width; //svcntw();//nnp_hwinfo.sve_simd_width;//nnp_hwinfo.simd_width;
  //      const int interchannels  = nnp_hwinfo.globalinterchannels;

         for(int i1=0;i1<simd_width;i1+=svcntw())
                {
                //printf("I am going in sve");
                svbool_t pg = svwhilelt_b32(i1,simd_width);
        // svfloat32_t const_0_25__5_00 = svld1rq(pg, datatmp);
         svfloat32_t const_0_25 = svdup_f32(0.25f);
         svfloat32_t const_0_5 = svdup_f32(5.00f);

        // const float32x4_t const_0_25 = vmovq_n_f32(0.25f);

        // Compute wd0 := d0 - d6
        svfloat32_t wd0 = svsub_m(pg, d0, d6);
        const svfloat32_t d4_sub_d2 = svsub_m(pg, d4, d2);
    


	    // Compute wd7 := d7 - d1
        svfloat32_t wd7 = svsub_m(pg, d7, d1);
        const svfloat32_t d3_sub_d5 = svsub_m(pg, d3, d5);
        // float32x4_t wd1 := d2 + d6
        svfloat32_t wd1 = svadd_m(pg, d2, d6);
        // Compute wd2 := d1 + d5
        svfloat32_t wd2 = svadd_m(pg, d1, d5);
          // Compute wd4 := d5 + 0.25 * d1
        svfloat32_t wd4 = svmla_m( pg, d5, d1, const_0_25);
        // Compute wd5 := d6 - 5.0 * d4
        svfloat32_t wd5 = svmls_m(pg, d6, d4, const_0_5);
        // Compute wd3 := d6 + 0.25 * d2
        svfloat32_t wd3 = svmla_m( pg, d6, d2, const_0_25);
	// Compute wd6 := d1 + 0.25 * d5
        svfloat32_t wd6 = svmla_m( pg, d1, d5, const_0_25);
       // const svfloat32_t const_5_25__4_25 = svld1rq(pg, datatmp1);
	const svfloat32_t const_5_25 = svdup_f32(5.25f);
	const svfloat32_t const_4_25 = svdup_f32(4.25f);
// Compute wd0 := (d0 - d6) + 5.25 * (d4 - d2)
        wd0 = svmla_m( pg, wd0, d4_sub_d2, const_5_25);
        // Compute wd7 := (d7 - d1) + 5.25 * (d3 - d5)
        wd7 = svmla_m( pg,wd7, d3_sub_d5, const_5_25);

        // Compute
        //   wd1 := (d6 + d2) - 4.25 * d4
        //   wd2 := (d1 + d5) - 4.25 * d3
        wd1 = svmls_m(pg,wd1, d4, const_4_25);
        wd2 = svmls_m( pg,wd2, d3, const_4_25);
        //const svfloat32_t const_1_25__4_00 = svld1(pg, datatmp2);
	const svfloat32_t const_1_25 = svdup_f32(1.25f);
	const svfloat32_t const_4_00 = svdup_f32(4.00f);
        // Compute
        //   wd3 := (d6 + 0.25 * d2) - 1.25 * d4
        //   wd4 := (d5 + 0.25 * d1) - 1.25 * d3
        //   wd6 := (d1 + 0.25 * d5) - 1.25 * d3
        //   wd5 := (d6 - 5.0 * d4) + 4.0 * d2
        wd3 = svmls_m( pg,wd3, d4, const_1_25);
        wd5 = svmla_m( pg,wd5, d2, const_4_00);
        wd4 = svmls_m( pg,wd4, d3, const_1_25);
        wd6 = svmls_m( pg,wd6, d3, const_1_25);

        const svfloat32_t const_2 = svdup_f32(2.0f);

        *transform0 = wd0;
        *transform1 = svadd_m(pg, wd1, wd2);
        *transform2 = svsub_m(pg, wd1, wd2);
        *transform3 = svmla_m(pg, wd3, wd4, const_2);
        *transform4 = svmls_m(pg, wd3, wd4, const_2);
        *transform5 = svmla_m(pg, wd5, wd6, const_2);
        *transform6 = svmls_m(pg, wd5, wd6, const_2);
        *transform7 = wd7;
}
}
static  void winograd_f6k3_input_transform1(
        const svfloat32_t d0,
        const svfloat32_t d1,
        const svfloat32_t d2,
        const svfloat32_t d3,
        const svfloat32_t d4,
        const svfloat32_t d5,
        const svfloat32_t d6,
        const svfloat32_t d7,
        svfloat32_t *transform0,
        svfloat32_t *transform1,
        svfloat32_t *transform2,
        svfloat32_t *transform3,
        svfloat32_t *transform4,
        svfloat32_t *transform5,
        svfloat32_t *transform6,
        svfloat32_t *transform7)
{
int simd_width=4;// nnp_hwinfo.simd_width;
         for(int i1=0;i1<simd_width;i1+=svcntw())
                {
                //printf("I am going in sve");
                svbool_t pg = svwhilelt_b32(i1,simd_width);
                float datatmp[] = {0.25f};
                float datatmp1[] = {5.00f};
        // svfloat32_t const_0_25__5_00 = svld1rq(pg, datatmp);
         svfloat32_t const_0_25 = svdup_f32(0.25f);
         svfloat32_t const_0_5 = svdup_f32(5.00f);

        // const float32x4_t const_0_25 = vmovq_n_f32(0.25f);

        // Compute wd0 := d0 - d6
        svfloat32_t wd0 = svsub_m(pg, d0, d6);
        const svfloat32_t d4_sub_d2 = svsub_m(pg, d4, d2);
    

	    // Compute wd7 := d7 - d1
        svfloat32_t wd7 = svsub_m(pg, d7, d1);
        const svfloat32_t d3_sub_d5 = svsub_m(pg, d3, d5);
        // float32x4_t wd1 := d2 + d6
        svfloat32_t wd1 = svadd_m(pg, d2, d6);
        // Compute wd2 := d1 + d5
        svfloat32_t wd2 = svadd_m(pg, d1, d5);
          // Compute wd4 := d5 + 0.25 * d1
        svfloat32_t wd4 = svmla_m( pg, d5, d1, const_0_25);
        // Compute wd5 := d6 - 5.0 * d4
        svfloat32_t wd5 = svmls_m(pg, d6, d4, const_0_5);
        // Compute wd3 := d6 + 0.25 * d2
        svfloat32_t wd3 = svmla_m( pg, d6, d2, const_0_25);
	// Compute wd6 := d1 + 0.25 * d5
        svfloat32_t wd6 = svmla_m( pg, d1, d5, const_0_25);
        float datatmp2[] = {5.25f};
        float datatmp3[] = {4.25f };
       // const svfloat32_t const_5_25__4_25 = svld1rq(pg, datatmp1);
	const svfloat32_t const_5_25 = svdup_f32(5.25f);
	const svfloat32_t const_4_25 = svdup_f32(4.25f);
// Compute wd0 := (d0 - d6) + 5.25 * (d4 - d2)
        wd0 = svmla_m( pg, wd0, d4_sub_d2, const_5_25);
        // Compute wd7 := (d7 - d1) + 5.25 * (d3 - d5)
        wd7 = svmla_m( pg,wd7, d3_sub_d5, const_5_25);

        // Compute
        //   wd1 := (d6 + d2) - 4.25 * d4
        //   wd2 := (d1 + d5) - 4.25 * d3
        wd1 = svmls_m(pg,wd1, d4, const_4_25);
        wd2 = svmls_m( pg,wd2, d3, const_4_25);
        float datatmp4[]={ 1.25f };
        float datatmp5[]={4.00f };
        //const svfloat32_t const_1_25__4_00 = svld1(pg, datatmp2);
	const svfloat32_t const_1_25 = svdup_f32(1.25f);
	const svfloat32_t const_4_00 = svdup_f32(4.00f);
        // Compute
        //   wd3 := (d6 + 0.25 * d2) - 1.25 * d4
        //   wd4 := (d5 + 0.25 * d1) - 1.25 * d3
        //   wd6 := (d1 + 0.25 * d5) - 1.25 * d3
        //   wd5 := (d6 - 5.0 * d4) + 4.0 * d2
        wd3 = svmls_m( pg,wd3, d4, const_1_25);
        wd5 = svmla_m( pg,wd5, d2, const_4_00);
        wd4 = svmls_m( pg,wd4, d3, const_1_25);
        wd6 = svmls_m( pg,wd6, d3, const_1_25);

        const svfloat32_t const_2 = svdup_f32(2.0f);

        *transform0 = wd0;
        *transform1 = svadd_m(pg, wd1, wd2);
        *transform2 = svsub_m(pg, wd1, wd2);
        *transform3 = svmla_m(pg, wd3, wd4, const_2);
        *transform4 = svmls_m(pg, wd3, wd4, const_2);
        *transform5 = svmla_m(pg, wd5, wd6, const_2);
        *transform6 = svmls_m(pg, wd5, wd6, const_2);
        *transform7 = wd7;}
}


static NNP_INLINE void winograd_f6k3_input_transform(
	const float32x4_t d0,
	const float32x4_t d1,
	const float32x4_t d2,
	const float32x4_t d3,
	const float32x4_t d4,
	const float32x4_t d5,
	const float32x4_t d6,
	const float32x4_t d7,
	float32x4_t transform0[restrict static 1],
	float32x4_t transform1[restrict static 1],
	float32x4_t transform2[restrict static 1],
	float32x4_t transform3[restrict static 1],
	float32x4_t transform4[restrict static 1],
	float32x4_t transform5[restrict static 1],
	float32x4_t transform6[restrict static 1],
	float32x4_t transform7[restrict static 1])
{
	static const float32x2_t const_0_25__5_00 = { 0.25f, 5.00f };

	// const float32x4_t const_0_25 = vmovq_n_f32(0.25f);

	// Compute wd0 := d0 - d6
	float32x4_t wd0 = vsubq_f32(d0, d6);
	const float32x4_t d4_sub_d2 = vsubq_f32(d4, d2);
	
	 float check[4];
        	 vst1q_f32(&check[0], d4_sub_d2);        
		printf("\n\n");
	// Compute wd7 := d7 - d1
	float32x4_t wd7 = vsubq_f32(d7, d1);
	const float32x4_t d3_sub_d5 = vsubq_f32(d3, d5);
	// float32x4_t wd1 := d2 + d6
	float32x4_t wd1 = vaddq_f32(d2, d6);
	// Compute wd2 := d1 + d5
	float32x4_t wd2 = vaddq_f32(d1, d5);
	// Compute wd4 := d5 + 0.25 * d1
	float32x4_t wd4 = vmuladdq_lane0_f32(d5, d1, const_0_25__5_00);
	// Compute wd5 := d6 - 5.0 * d4
	float32x4_t wd5 = vmulsubq_lane1_f32(d6, d4, const_0_25__5_00);
	// Compute wd3 := d6 + 0.25 * d2
	float32x4_t wd3 = vmuladdq_lane0_f32(d6, d2, const_0_25__5_00);
	// Compute wd6 := d1 + 0.25 * d5
	float32x4_t wd6 = vmuladdq_lane0_f32(d1, d5, const_0_25__5_00);

	const float32x2_t const_5_25__4_25 = { 5.25f, 4.25f };
	// Compute wd0 := (d0 - d6) + 5.25 * (d4 - d2)
	wd0 = vmuladdq_lane0_f32(wd0, d4_sub_d2, const_5_25__4_25);
	// Compute wd7 := (d7 - d1) + 5.25 * (d3 - d5)
	wd7 = vmuladdq_lane0_f32(wd7, d3_sub_d5, const_5_25__4_25);

	// Compute
	//   wd1 := (d6 + d2) - 4.25 * d4
	//   wd2 := (d1 + d5) - 4.25 * d3
	wd1 = vmulsubq_lane1_f32(wd1, d4, const_5_25__4_25);
	wd2 = vmulsubq_lane1_f32(wd2, d3, const_5_25__4_25);

	const float32x2_t const_1_25__4_00 = { 1.25f, 4.00f };
	// Compute
	//   wd3 := (d6 + 0.25 * d2) - 1.25 * d4
	//   wd4 := (d5 + 0.25 * d1) - 1.25 * d3
	//   wd6 := (d1 + 0.25 * d5) - 1.25 * d3
	//   wd5 := (d6 - 5.0 * d4) + 4.0 * d2
	wd3 = vmulsubq_lane0_f32(wd3, d4, const_1_25__4_00);
	wd5 = vmuladdq_lane1_f32(wd5, d2, const_1_25__4_00);
	wd4 = vmulsubq_lane0_f32(wd4, d3, const_1_25__4_00);
	wd6 = vmulsubq_lane0_f32(wd6, d3, const_1_25__4_00);

	const float32x4_t const_2 = vmovq_n_f32(2.0f);

	*transform0 = wd0;
	*transform1 = vaddq_f32(wd1, wd2);
	*transform2 = vsubq_f32(wd1, wd2);
	*transform3 = vmuladdq_f32(wd3, wd4, const_2);
	*transform4 = vmulsubq_f32(wd3, wd4, const_2);
	*transform5 = vmuladdq_f32(wd5, wd6, const_2);
	*transform6 = vmulsubq_f32(wd5, wd6, const_2);
	*transform7 = wd7;
}


static  void winograd_f6k3_kernel_transform_intertile(
        const svfloat32_t g0, const svfloat32_t g1, const svfloat32_t g2,
        svfloat32_t *transform0,
        svfloat32_t *transform1,
        svfloat32_t *transform2,
        svfloat32_t *transform3,
        svfloat32_t *transform4,
        svfloat32_t *transform5,
        svfloat32_t *transform6,
        svfloat32_t *transform7,
        bool rescale_coefficients)
{
int simd_width = nnp_hwinfo.sve_simd_width;
  for(int i1=0;i1<simd_width;i1+=svcntw())
        {
                svbool_t pg = svwhilelt_b32(i1,simd_width);

        /*
         * w0 = g0
         * w1 = ((g0 + g2) + g1) * (-2.0 / 9)
         * w2 = ((g0 + g2) - g1) * (-2.0 / 9)
         * w3 = ((g0 + 4 * g2) + 2 * g1) * (1.0 / 90)
         * w4 = ((g0 + 4 * g2) - 2 * g1) * (1.0 / 90)
         * w5 = ((g2 + 4 * g0) + 2 * g1) * (1.0 / 180)
         * w6 = ((g2 + 4 * g0) - 2 * g1) * (1.0 / 180)
         * w7 = g2
         */

        /*
         * Compute
         *   w2 := g0 + g2
         *   w4 := g0 + 4 * g2
         *   w6 := g2 + 4 * g0
         */
        const svfloat32_t const_4 = svdup_f32(4.0f);
        svfloat32_t w2 = svadd_m(pg, g0, g2);
        svfloat32_t w4 = svmla_m(pg, g0, const_4, g2);
        svfloat32_t w6 = svmla_m(pg, g2, const_4, g0);


        /*
         * Compute
         *   w1 = (g0 + g2) + g1
         *   w2 = (g0 + g2) - g1
         *   w3 = (g0 + 4 * g2) + 2 * g1
         *   w4 = (g0 + 4 * g2) - 2 * g1
         *   w5 = (g2 + 4 * g0) + 2 * g1
         *   w6 = (g2 + 4 * g0) - 2 * g1
         */
        const svfloat32_t two_g1 = svmul_m(pg, g1 , svdup_f32(2.0f));
        svfloat32_t w1 = svadd_m(pg,w2 , g1);
        w2 = svsub_m(pg, w2, g1);
        //w2 = w2 - g1;
        svfloat32_t w3 = svadd_m(pg, w4 , two_g1);
        w4 = svsub_m(pg, w4 , two_g1);
        svfloat32_t w5 = svadd_m(pg, w6 , two_g1);
        w6 = svsub_m(pg, w6 , two_g1);

 

        if (rescale_coefficients) {
                const svfloat32_t minus_2_over_9 = svdup_f32(-0x1.C71C72p-3f);

                w1 = svmul_m(pg, w1,  minus_2_over_9);
                w2 = svmul_m(pg, w2,  minus_2_over_9);

                const svfloat32_t rcp_90 = svdup_f32(0x1.6C16C2p-7f);
                w3 = svmul_m(pg, w3,  rcp_90);
                w4 = svmul_m(pg, w4,  rcp_90);

                const svfloat32_t rcp_180 = svdup_f32(0x1.6C16C2p-8f);
                w5= svmul_m(pg, w5,  rcp_180);
                w6 = svmul_m(pg, w6, rcp_180);
        }

        *transform0 = g0;
        *transform1 = w1;
        *transform2 = w2;
        *transform3 = w3;
        *transform4 = w4;
        *transform5 = w5;
        *transform6 = w6;
        *transform7 = g2;
}}
static  void winograd_f6k3_kernel_transform1(
        const svfloat32_t g0, const svfloat32_t g1, const svfloat32_t g2,
        svfloat32_t *transform0,
        svfloat32_t *transform1,
        svfloat32_t *transform2,
        svfloat32_t *transform3,
        svfloat32_t *transform4,
        svfloat32_t *transform5,
        svfloat32_t *transform6,
        svfloat32_t *transform7,
        bool rescale_coefficients)
{
int simd_width=nnp_hwinfo.simd_width;
  for(int i1=0;i1<simd_width;i1+=svcntw())
                {
                 svbool_t pg = svwhilelt_b32(i1,simd_width);

        /*
         * w0 = g0
         * w1 = ((g0 + g2) + g1) * (-2.0 / 9)
         * w2 = ((g0 + g2) - g1) * (-2.0 / 9)
         * w3 = ((g0 + 4 * g2) + 2 * g1) * (1.0 / 90)
         * w4 = ((g0 + 4 * g2) - 2 * g1) * (1.0 / 90)
         * w5 = ((g2 + 4 * g0) + 2 * g1) * (1.0 / 180)
         * w6 = ((g2 + 4 * g0) - 2 * g1) * (1.0 / 180)
         * w7 = g2
         */

        /*
         * Compute
         *   w2 := g0 + g2
         *   w4 := g0 + 4 * g2
         *   w6 := g2 + 4 * g0
         */
        const svfloat32_t const_4 = svdup_f32(4.0f);
        svfloat32_t w2 = svadd_m(pg, g0, g2);
        svfloat32_t w4 = svmla_m(pg, g0, const_4, g2);
        svfloat32_t w6 = svmla_m(pg, g2, const_4, g0);


        /*
         * Compute
         *   w1 = (g0 + g2) + g1
         *   w2 = (g0 + g2) - g1
         *   w3 = (g0 + 4 * g2) + 2 * g1
         *   w4 = (g0 + 4 * g2) - 2 * g1
         *   w5 = (g2 + 4 * g0) + 2 * g1
         *   w6 = (g2 + 4 * g0) - 2 * g1
         */
        const svfloat32_t two_g1 = svmul_m(pg, g1 , svdup_f32(2.0f));
        svfloat32_t w1 = svadd_m(pg,w2 , g1);
        w2 = svsub_m(pg, w2, g1);
        //w2 = w2 - g1;
        svfloat32_t w3 = svadd_m(pg, w4 , two_g1);
        w4 = svsub_m(pg, w4 , two_g1);
        svfloat32_t w5 = svadd_m(pg, w6 , two_g1);
        w6 = svsub_m(pg, w6 , two_g1);

 

        if (rescale_coefficients) {
                const svfloat32_t minus_2_over_9 = svdup_f32(-0x1.C71C72p-3f);

                w1 = svmul_m(pg, w1,  minus_2_over_9);
                w2 = svmul_m(pg, w2,  minus_2_over_9);

                const svfloat32_t rcp_90 = svdup_f32(0x1.6C16C2p-7f);
                w3 = svmul_m(pg, w3,  rcp_90);
                w4 = svmul_m(pg, w4,  rcp_90);

                const svfloat32_t rcp_180 = svdup_f32(0x1.6C16C2p-8f);
                w5= svmul_m(pg, w5,  rcp_180);
                w6 = svmul_m(pg, w6, rcp_180);
        }

        *transform0 = g0;
        *transform1 = w1;
        *transform2 = w2;
        *transform3 = w3;
        *transform4 = w4;
        *transform5 = w5;
        *transform6 = w6;
        *transform7 = g2;
}}

static NNP_INLINE void winograd_f6k3_kernel_transform(
	const float32x4_t g0, const float32x4_t g1, const float32x4_t g2,
	float32x4_t transform0[restrict static 1],
	float32x4_t transform1[restrict static 1],
	float32x4_t transform2[restrict static 1],
	float32x4_t transform3[restrict static 1],
	float32x4_t transform4[restrict static 1],
	float32x4_t transform5[restrict static 1],
	float32x4_t transform6[restrict static 1],
	float32x4_t transform7[restrict static 1],
	bool rescale_coefficients)
{
	/*
	 * w0 = g0
	 * w1 = ((g0 + g2) + g1) * (-2.0 / 9)
	 * w2 = ((g0 + g2) - g1) * (-2.0 / 9)
	 * w3 = ((g0 + 4 * g2) + 2 * g1) * (1.0 / 90)
	 * w4 = ((g0 + 4 * g2) - 2 * g1) * (1.0 / 90)
	 * w5 = ((g2 + 4 * g0) + 2 * g1) * (1.0 / 180)
	 * w6 = ((g2 + 4 * g0) - 2 * g1) * (1.0 / 180)
	 * w7 = g2
	 */

	/*
	 * Compute
	 *   w2 := g0 + g2
	 *   w4 := g0 + 4 * g2
	 *   w6 := g2 + 4 * g0
	 */
	const float32x4_t const_4 = vdupq_n_f32(4.0f);
	float32x4_t w2 = g0 + g2;
	float32x4_t w4 = vmuladdq_f32(g0, const_4, g2);
	float32x4_t w6 = vmuladdq_f32(g2, const_4, g0);

	/*
	 * Compute
	 *   w1 = (g0 + g2) + g1
	 *   w2 = (g0 + g2) - g1
	 *   w3 = (g0 + 4 * g2) + 2 * g1
	 *   w4 = (g0 + 4 * g2) - 2 * g1
	 *   w5 = (g2 + 4 * g0) + 2 * g1
	 *   w6 = (g2 + 4 * g0) - 2 * g1
	 */
	const float32x4_t two_g1 = g1 * vdupq_n_f32(2.0f);
	float32x4_t w1 = w2 + g1;
	w2 = w2 - g1;
	float32x4_t w3 = w4 + two_g1;
	w4 = w4 - two_g1;
	float32x4_t w5 = w6 + two_g1;
	w6 = w6 - two_g1;

	if (rescale_coefficients) {
		const float32x4_t minus_2_over_9 = vdupq_n_f32(-0x1.C71C72p-3f);
		w1 *= minus_2_over_9;
		w2 *= minus_2_over_9;

		const float32x4_t rcp_90 = vdupq_n_f32(0x1.6C16C2p-7f);
		w3 *= rcp_90;
		w4 *= rcp_90;

		const float32x4_t rcp_180 = vdupq_n_f32(0x1.6C16C2p-8f);
		w5 *= rcp_180;
		w6 *= rcp_180;
	}

	*transform0 = g0;
	*transform1 = w1;
	*transform2 = w2;
	*transform3 = w3;
	*transform4 = w4;
	*transform5 = w5;
	*transform6 = w6;
	*transform7 = g2;
}

static NNP_INLINE void winograd_f6k3_output_transform(
	const float32x2_t m0,
	const float32x2_t m1,
	const float32x2_t m2,
	const float32x2_t m3,
	const float32x2_t m4,
	const float32x2_t m5,
	const float32x2_t m6,
	const float32x2_t m7,
	float32x2_t output0[restrict static 1],
	float32x2_t output1[restrict static 1],
	float32x2_t output2[restrict static 1],
	float32x2_t output3[restrict static 1],
	float32x2_t output4[restrict static 1],
	float32x2_t output5[restrict static 1])
{
	/*
	 * s0 = m0 + (m1 + m2) +      (m3 + m4) + 32 * (m5 + m6)
	 * s1 =      (m1 - m2) +  2 * (m3 - m4) + 16 * (m5 - m6)
	 * s2 =      (m1 + m2) +  4 * (m3 + m4) +  8 * (m5 + m6)
	 * s3 =      (m1 - m2) +  8 * (m3 - m4) +  4 * (m5 - m6)
	 * s4 =      (m1 + m2) + 16 * (m3 + m4) +  2 * (m5 + m6)
	 * s5 =      (m1 - m2) + 32 * (m3 - m4) +      (m5 - m6) + m7
	 */

	const float32x2_t m1_add_m2 = vadd_f32(m1, m2);
	const float32x2_t m1_sub_m2 = vsub_f32(m1, m2);
	const float32x2_t m3_add_m4 = vadd_f32(m3, m4);
	const float32x2_t m3_sub_m4 = vsub_f32(m3, m4);
	const float32x2_t m5_add_m6 = vadd_f32(m5, m6);
	const float32x2_t m5_sub_m6 = vsub_f32(m5, m6);

	float32x2_t s0 = vadd_f32(m0, m1_add_m2);
	float32x2_t s5 = vadd_f32(m7, m1_sub_m2);

	const float32x2_t const_16__8 = { 16.0f, 8.0f };
	float32x2_t s1 = vmuladd_lane0_f32(m1_sub_m2, m5_sub_m6, const_16__8);
	float32x2_t s4 = vmuladd_lane0_f32(m1_add_m2, m3_add_m4, const_16__8);
	float32x2_t s2 = vmuladd_lane1_f32(m1_add_m2, m5_add_m6, const_16__8);
	float32x2_t s3 = vmuladd_lane1_f32(m1_sub_m2, m3_sub_m4, const_16__8);

	const float32x2_t const_32__2 = { 32.0f, 2.0f };
	s0 = vmuladd_lane0_f32(s0, m5_add_m6, const_32__2);
	s5 = vmuladd_lane0_f32(s5, m3_sub_m4, const_32__2);
	s1 = vmuladd_lane1_f32(s1, m3_sub_m4, const_32__2);
	s4 = vmuladd_lane1_f32(s4, m5_add_m6, const_32__2);

	s0 = vadd_f32(s0, m3_add_m4);
	s5 = vadd_f32(s5, m5_sub_m6);

	const float32x2_t const_4 = vmov_n_f32(4.0f);
	s2 = vmuladd_f32(s2, m3_add_m4, const_4);
	s3 = vmuladd_f32(s3, m5_sub_m6, const_4);

	*output0 = s0;
	*output1 = s1;
	*output2 = s2;
	*output3 = s3;
	*output4 = s4;
	*output5 = s5;
}


static NNP_INLINE void winograd_f6k3_output_transform1_intertile(
	const svfloat32_t m0,
	const svfloat32_t m1,
	const svfloat32_t m2,
	const svfloat32_t m3,
	const svfloat32_t m4,
	const svfloat32_t m5,
	const svfloat32_t m6,
	const svfloat32_t m7,
	svfloat32_t *output0,
	svfloat32_t *output1,
	svfloat32_t *output2,
	svfloat32_t *output3,
	svfloat32_t *output4,
	svfloat32_t *output5)
{
	/*
	 * s0 = m0 + (m1 + m2) +      (m3 + m4) + 32 * (m5 + m6)
	 * s1 =      (m1 - m2) +  2 * (m3 - m4) + 16 * (m5 - m6)
	 * s2 =      (m1 + m2) +  4 * (m3 + m4) +  8 * (m5 + m6)
	 * s3 =      (m1 - m2) +  8 * (m3 - m4) +  4 * (m5 - m6)
	 * s4 =      (m1 + m2) + 16 * (m3 + m4) +  2 * (m5 + m6)
	 * s5 =      (m1 - m2) + 32 * (m3 - m4) +      (m5 - m6) + m7
	 */
//int simd_width = svcntw()/2;//nnp_hwinfo.sve_simd_width/2;
int simd_width = nnp_hwinfo.sve_simd_width/2;
	for(int i1=0;i1<simd_width;i1+=svcntw())
        {
         svbool_t pg = svwhilelt_b32(i1,simd_width);

        const svfloat32_t m1_add_m2 = svadd_m(pg, m1, m2);
        const svfloat32_t m1_sub_m2 = svsub_m(pg, m1, m2);
        const svfloat32_t m3_add_m4 = svadd_m(pg , m3, m4);
        const svfloat32_t m3_sub_m4 = svsub_m(pg, m3, m4);
        const svfloat32_t m5_add_m6 = svadd_m(pg, m5, m6);
        const svfloat32_t m5_sub_m6 = svsub_m(pg, m5, m6);

        svfloat32_t s0 = svadd_m(pg, m0, m1_add_m2);
        svfloat32_t s5 = svadd_m(pg, m7, m1_sub_m2);

         svfloat32_t const_16 = svdup_f32(16.0f);
         svfloat32_t const_8 = svdup_f32(8.0f);

        svfloat32_t s1 = svmla_m(pg, m1_sub_m2, m5_sub_m6, const_16);
        svfloat32_t s4 = svmla_m(pg, m1_add_m2, m3_add_m4, const_16);
        svfloat32_t s2 = svmla_m(pg, m1_add_m2, m5_add_m6, const_8);
        svfloat32_t s3 = svmla_m(pg, m1_sub_m2, m3_sub_m4, const_8);

           svfloat32_t const_32 = svdup_f32(32.0f);
         svfloat32_t const_2 = svdup_f32(2.0f);

        s0 = svmla_m(pg, s0, m5_add_m6, const_32);
        s5 = svmla_m(pg, s5, m3_sub_m4, const_32);
        s1 = svmla_m(pg, s1, m3_sub_m4, const_2);
        s4 = svmla_m(pg, s4, m5_add_m6, const_2);

        s0 = svadd_m(pg, s0, m3_add_m4);
        s5 = svadd_m(pg, s5, m5_sub_m6);

        const svfloat32_t const_4 = svdup_f32(4.0f);
        s2 = svmla_m(pg, s2, m3_add_m4, const_4);
        s3 = svmla_m(pg, s3, m5_sub_m6, const_4);
	*output0 = s0;
	*output1 = s1;
	*output2 = s2;
	*output3 = s3;
	*output4 = s4;
	*output5 = s5;
	}	
}
static NNP_INLINE void winograd_f6k3_output_transform1(
	const svfloat32_t m0,
	const svfloat32_t m1,
	const svfloat32_t m2,
	const svfloat32_t m3,
	const svfloat32_t m4,
	const svfloat32_t m5,
	const svfloat32_t m6,
	const svfloat32_t m7,
	svfloat32_t *output0,
	svfloat32_t *output1,
	svfloat32_t *output2,
	svfloat32_t *output3,
	svfloat32_t *output4,
	svfloat32_t *output5)
{
	/*
	 * s0 = m0 + (m1 + m2) +      (m3 + m4) + 32 * (m5 + m6)
	 * s1 =      (m1 - m2) +  2 * (m3 - m4) + 16 * (m5 - m6)
	 * s2 =      (m1 + m2) +  4 * (m3 + m4) +  8 * (m5 + m6)
	 * s3 =      (m1 - m2) +  8 * (m3 - m4) +  4 * (m5 - m6)
	 * s4 =      (m1 + m2) + 16 * (m3 + m4) +  2 * (m5 + m6)
	 * s5 =      (m1 - m2) + 32 * (m3 - m4) +      (m5 - m6) + m7
	 */

	int simd_width=nnp_hwinfo.simd_width/2;
	for(int i1=0;i1<simd_width;i1+=svcntw())
        {
         svbool_t pg = svwhilelt_b32(i1,simd_width);

        const svfloat32_t m1_add_m2 = svadd_m(pg, m1, m2);
        const svfloat32_t m1_sub_m2 = svsub_m(pg, m1, m2);
        const svfloat32_t m3_add_m4 = svadd_m(pg , m3, m4);
        const svfloat32_t m3_sub_m4 = svsub_m(pg, m3, m4);
        const svfloat32_t m5_add_m6 = svadd_m(pg, m5, m6);
        const svfloat32_t m5_sub_m6 = svsub_m(pg, m5, m6);

        svfloat32_t s0 = svadd_m(pg, m0, m1_add_m2);
        svfloat32_t s5 = svadd_m(pg, m7, m1_sub_m2);

         svfloat32_t const_16 = svdup_f32(16.0f);
         svfloat32_t const_8 = svdup_f32(8.0f);

        svfloat32_t s1 = svmla_m(pg, m1_sub_m2, m5_sub_m6, const_16);
        svfloat32_t s4 = svmla_m(pg, m1_add_m2, m3_add_m4, const_16);
        svfloat32_t s2 = svmla_m(pg, m1_add_m2, m5_add_m6, const_8);
        svfloat32_t s3 = svmla_m(pg, m1_sub_m2, m3_sub_m4, const_8);

           svfloat32_t const_32 = svdup_f32(32.0f);
         svfloat32_t const_2 = svdup_f32(2.0f);

        s0 = svmla_m(pg, s0, m5_add_m6, const_32);
        s5 = svmla_m(pg, s5, m3_sub_m4, const_32);
        s1 = svmla_m(pg, s1, m3_sub_m4, const_2);
        s4 = svmla_m(pg, s4, m5_add_m6, const_2);

        s0 = svadd_m(pg, s0, m3_add_m4);
        s5 = svadd_m(pg, s5, m5_sub_m6);

        const svfloat32_t const_4 = svdup_f32(4.0f);
        s2 = svmla_m(pg, s2, m3_add_m4, const_4);
        s3 = svmla_m(pg, s3, m5_sub_m6, const_4);
	*output0 = s0;
	*output1 = s1;
	*output2 = s2;
	*output3 = s3;
	*output4 = s4;
	*output5 = s5;
	}	
}
static NNP_INLINE void winograd_f6k3_output_transformq(
	const float32x4_t m0,
	const float32x4_t m1,
	const float32x4_t m2,
	const float32x4_t m3,
	const float32x4_t m4,
	const float32x4_t m5,
	const float32x4_t m6,
	const float32x4_t m7,
	float32x4_t output0[restrict static 1],
	float32x4_t output1[restrict static 1],
	float32x4_t output2[restrict static 1],
	float32x4_t output3[restrict static 1],
	float32x4_t output4[restrict static 1],
	float32x4_t output5[restrict static 1])
{
	/*
	 * s0 = m0 + (m1 + m2) +      (m3 + m4) + 32 * (m5 + m6)
	 * s1 =      (m1 - m2) +  2 * (m3 - m4) + 16 * (m5 - m6)
	 * s2 =      (m1 + m2) +  4 * (m3 + m4) +  8 * (m5 + m6)
	 * s3 =      (m1 - m2) +  8 * (m3 - m4) +  4 * (m5 - m6)
	 * s4 =      (m1 + m2) + 16 * (m3 + m4) +  2 * (m5 + m6)
	 * s5 =      (m1 - m2) + 32 * (m3 - m4) +      (m5 - m6) + m7
	 */

	const float32x4_t m1_add_m2 = vaddq_f32(m1, m2);
	const float32x4_t m1_sub_m2 = vsubq_f32(m1, m2);
	const float32x4_t m3_add_m4 = vaddq_f32(m3, m4);
	const float32x4_t m3_sub_m4 = vsubq_f32(m3, m4);
	const float32x4_t m5_add_m6 = vaddq_f32(m5, m6);
	const float32x4_t m5_sub_m6 = vsubq_f32(m5, m6);

	float32x4_t s0 = vaddq_f32(m0, m1_add_m2);
	float32x4_t s5 = vaddq_f32(m7, m1_sub_m2);

	const float32x2_t const_16__8 = { 16.0f, 8.0f };
	float32x4_t s1 = vmuladdq_lane0_f32(m1_sub_m2, m5_sub_m6, const_16__8);
	float32x4_t s4 = vmuladdq_lane0_f32(m1_add_m2, m3_add_m4, const_16__8);
	float32x4_t s2 = vmuladdq_lane1_f32(m1_add_m2, m5_add_m6, const_16__8);
	float32x4_t s3 = vmuladdq_lane1_f32(m1_sub_m2, m3_sub_m4, const_16__8);

	const float32x2_t const_32__2 = { 32.0f, 2.0f };
	s0 = vmuladdq_lane0_f32(s0, m5_add_m6, const_32__2);
	s5 = vmuladdq_lane0_f32(s5, m3_sub_m4, const_32__2);
	s1 = vmuladdq_lane1_f32(s1, m3_sub_m4, const_32__2);
	s4 = vmuladdq_lane1_f32(s4, m5_add_m6, const_32__2);

	s0 = vaddq_f32(s0, m3_add_m4);
	s5 = vaddq_f32(s5, m5_sub_m6);

	const float32x4_t const_4 = vmovq_n_f32(4.0f);
	s2 = vmuladdq_f32(s2, m3_add_m4, const_4);
	s3 = vmuladdq_f32(s3, m5_sub_m6, const_4);

	*output0 = s0;
	*output1 = s1;
	*output2 = s2;
	*output3 = s3;
	*output4 = s4;
	*output5 = s5;
}



static NNP_INLINE void winograd_f6k3_output_transformq1_intertile(
	const svfloat32_t m0,
	const svfloat32_t m1,
	const svfloat32_t m2,
	const svfloat32_t m3,
	const svfloat32_t m4,
	const svfloat32_t m5,
	const svfloat32_t m6,
	const svfloat32_t m7,
	svfloat32_t *output0,
	svfloat32_t *output1,
	svfloat32_t *output2,
	svfloat32_t *output3,
	svfloat32_t *output4,
	svfloat32_t *output5)
{
int simd_width =svcntw();// nnp_hwinfo.sve_simd_width;
	//int simd_width=16;//nnp_hwinfo.simd_width;
	/*
	 * s0 = m0 + (m1 + m2) +      (m3 + m4) + 32 * (m5 + m6)
	 * s1 =      (m1 - m2) +  2 * (m3 - m4) + 16 * (m5 - m6)
	 * s2 =      (m1 + m2) +  4 * (m3 + m4) +  8 * (m5 + m6)
	 * s3 =      (m1 - m2) +  8 * (m3 - m4) +  4 * (m5 - m6)
	 * s4 =      (m1 + m2) + 16 * (m3 + m4) +  2 * (m5 + m6)
	 * s5 =      (m1 - m2) + 32 * (m3 - m4) +      (m5 - m6) + m7
	 */

	for(int i1=0;i1<simd_width;i1+=svcntw())
        {
	 svbool_t pg = svwhilelt_b32(i1,simd_width);

        const svfloat32_t m1_add_m2 = svadd_m(pg, m1, m2);
        const svfloat32_t m1_sub_m2 = svsub_m(pg, m1, m2);
        const svfloat32_t m3_add_m4 = svadd_m(pg, m3, m4);
        const svfloat32_t m3_sub_m4 = svsub_m(pg, m3, m4);
        const svfloat32_t m5_add_m6 = svadd_m(pg, m5, m6);
        const svfloat32_t m5_sub_m6 = svsub_m(pg, m5, m6);

        svfloat32_t s0 = svadd_m(pg, m0, m1_add_m2);
        svfloat32_t s5 = svadd_m(pg, m7, m1_sub_m2);

         svfloat32_t const_16 = svdup_f32(16.0f);
         svfloat32_t const_8 = svdup_f32(8.0f);	
	
	svfloat32_t s1 = svmla_m(pg, m1_sub_m2, m5_sub_m6, const_16);
	svfloat32_t s4 = svmla_m(pg, m1_add_m2, m3_add_m4, const_16);
	svfloat32_t s2 = svmla_m(pg, m1_add_m2, m5_add_m6, const_8);
	svfloat32_t s3 = svmla_m(pg, m1_sub_m2, m3_sub_m4, const_8);
	
	   svfloat32_t const_32 = svdup_f32(32.0f);
         svfloat32_t const_2 = svdup_f32(2.0f);

	s0 = svmla_m(pg, s0, m5_add_m6, const_32);
	s5 = svmla_m(pg, s5, m3_sub_m4, const_32);
	s1 = svmla_m(pg, s1, m3_sub_m4, const_2);
	s4 = svmla_m(pg, s4, m5_add_m6, const_2);

	s0 = svadd_m(pg, s0, m3_add_m4);
	s5 = svadd_m(pg, s5, m5_sub_m6);

	const svfloat32_t const_4 = svdup_f32(4.0f);
	s2 = svmla_m(pg, s2, m3_add_m4, const_4);
	s3 = svmla_m(pg, s3, m5_sub_m6, const_4);
	*output0 = s0;
	*output1 = s1;
	*output2 = s2;
	*output3 = s3;
	*output4 = s4;
	*output5 = s5;
	}
}
static NNP_INLINE void winograd_f6k3_output_transformq1(
	const svfloat32_t m0,
	const svfloat32_t m1,
	const svfloat32_t m2,
	const svfloat32_t m3,
	const svfloat32_t m4,
	const svfloat32_t m5,
	const svfloat32_t m6,
	const svfloat32_t m7,
	svfloat32_t *output0,
	svfloat32_t *output1,
	svfloat32_t *output2,
	svfloat32_t *output3,
	svfloat32_t *output4,
	svfloat32_t *output5)
{
	int simd_width=nnp_hwinfo.simd_width;
	/*
	 * s0 = m0 + (m1 + m2) +      (m3 + m4) + 32 * (m5 + m6)
	 * s1 =      (m1 - m2) +  2 * (m3 - m4) + 16 * (m5 - m6)
	 * s2 =      (m1 + m2) +  4 * (m3 + m4) +  8 * (m5 + m6)
	 * s3 =      (m1 - m2) +  8 * (m3 - m4) +  4 * (m5 - m6)
	 * s4 =      (m1 + m2) + 16 * (m3 + m4) +  2 * (m5 + m6)
	 * s5 =      (m1 - m2) + 32 * (m3 - m4) +      (m5 - m6) + m7
	 */

	for(int i1=0;i1<simd_width;i1+=svcntw())
        {
	 svbool_t pg = svwhilelt_b32(i1,simd_width);

        const svfloat32_t m1_add_m2 = svadd_m(pg, m1, m2);
        const svfloat32_t m1_sub_m2 = svsub_m(pg, m1, m2);
        const svfloat32_t m3_add_m4 = svadd_m(pg, m3, m4);
        const svfloat32_t m3_sub_m4 = svsub_m(pg, m3, m4);
        const svfloat32_t m5_add_m6 = svadd_m(pg, m5, m6);
        const svfloat32_t m5_sub_m6 = svsub_m(pg, m5, m6);

        svfloat32_t s0 = svadd_m(pg, m0, m1_add_m2);
        svfloat32_t s5 = svadd_m(pg, m7, m1_sub_m2);

         svfloat32_t const_16 = svdup_f32(16.0f);
         svfloat32_t const_8 = svdup_f32(8.0f);	
	
	svfloat32_t s1 = svmla_m(pg, m1_sub_m2, m5_sub_m6, const_16);
	svfloat32_t s4 = svmla_m(pg, m1_add_m2, m3_add_m4, const_16);
	svfloat32_t s2 = svmla_m(pg, m1_add_m2, m5_add_m6, const_8);
	svfloat32_t s3 = svmla_m(pg, m1_sub_m2, m3_sub_m4, const_8);
	
	   svfloat32_t const_32 = svdup_f32(32.0f);
         svfloat32_t const_2 = svdup_f32(2.0f);

	s0 = svmla_m(pg, s0, m5_add_m6, const_32);
	s5 = svmla_m(pg, s5, m3_sub_m4, const_32);
	s1 = svmla_m(pg, s1, m3_sub_m4, const_2);
	s4 = svmla_m(pg, s4, m5_add_m6, const_2);

	s0 = svadd_m(pg, s0, m3_add_m4);
	s5 = svadd_m(pg, s5, m5_sub_m6);

	const svfloat32_t const_4 = svdup_f32(4.0f);
	s2 = svmla_m(pg, s2, m3_add_m4, const_4);
	s3 = svmla_m(pg, s3, m5_sub_m6, const_4);
	*output0 = s0;
	*output1 = s1;
	*output2 = s2;
	*output3 = s3;
	*output4 = s4;
	*output5 = s5;
	}
}
