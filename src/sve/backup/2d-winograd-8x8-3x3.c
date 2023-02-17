#include <stdint.h>
#include <stddef.h>
#include<stdio.h>
#include <nnpack/arm_neon.h>
#include <nnpack/activations.h>
#include <nnpack/macros.h>
#include <nnpack/utils.h>
#include<nnpack.h>
#include <sve/winograd/f6x6k3x3.h>
#include <sve/transpose.h>
#include <nnpack/hwinfo.h>
#include <stdlib.h>
void nnp_iwt8x8_3x3_with_offset__neon(
        const float data[restrict static 1],
        void* transform,
        size_t data_stride,
        size_t transform_stride,
        uint32_t row_count,
        uint32_t column_count,
        uint32_t row_offset,
        uint32_t column_offset)
{

//	printf("row_count=%d column_count=%d, transform_stride=%d, data_stride=%d\n", row_count, column_count, transform_stride, data_stride);
	 int simd_width = nnp_hwinfo.simd_width;
	  NNP_SIMD_ALIGN svfloat32_t wd0, wd1, wd2, wd3, wd4, wd5, wd6, wd7, wd8, wd9, wd10, wd11, wd12, wd13, wd14, wd15;
		 
                for(int i1=0;i1<simd_width;i1+=svcntw())
                {
         //       printf("I am going in sve");
                svbool_t pg = svwhilelt_b32(i1,simd_width);

        	if NNP_LIKELY(row_count == 8 && column_count == 8 && row_offset == 0 && column_offset == 0) {
                // Fast path where we can directly load `data` into `wd`.


                        winograd_f6k3_input_transform1(
                                svld1_f32(pg, &data[0 * data_stride + 0 * simd_width]),
                                svld1_f32(pg, &data[1 * data_stride + 0 * simd_width]),
                                svld1_f32(pg, &data[2 * data_stride + 0 * simd_width]),
                                svld1_f32(pg, &data[3 * data_stride + 0 * simd_width]),
                                svld1_f32(pg, &data[4 * data_stride + 0 * simd_width]),
                                svld1_f32(pg, &data[5 * data_stride + 0 * simd_width]),
                                svld1_f32(pg, &data[6 * data_stride + 0 * simd_width]),
                                svld1_f32(pg, &data[7 * data_stride + 0 * simd_width]),
                                &wd0, &wd1, &wd2, &wd3,&wd8, &wd9, &wd10, &wd11);

                         winograd_f6k3_input_transform1(
                                svld1_f32(pg, &data[0 * data_stride + 1 * simd_width]),
                                svld1_f32(pg, &data[1 * data_stride + 1 * simd_width]),
				svld1_f32(pg, &data[2 * data_stride + 1 * simd_width]),
                                svld1_f32(pg, &data[3 * data_stride + 1 * simd_width]),
                                svld1_f32(pg, &data[4 * data_stride + 1 * simd_width]),
                                svld1_f32(pg, &data[5 * data_stride + 1 * simd_width]),
                                svld1_f32(pg, &data[6 * data_stride + 1 * simd_width]),
                                svld1_f32(pg, &data[7 * data_stride + 1 * simd_width]),
                                &wd4, &wd5, &wd6, &wd7,&wd12, &wd13, &wd14, &wd15);

        } else {
                NNP_SIMD_ALIGN float block[8][simd_width*2];
                {
                        const svfloat32_t vzero = svdup_f32(0.0f);
                        for (float *block_ptr = &block[0][0], *block_end = &block[8][0]; block_ptr != block_end; block_ptr += simd_width) {
                                svst1_f32(pg, block_ptr, vzero);
                        }
                }
                for (size_t i = 0; i < row_count; i++) {
                        for (size_t j = 0; j < column_count; j++) {
                                block[row_offset + i][column_offset + j] = data[i * data_stride + j];
                        }
                }
                for (size_t col = 0; col < 1; col++) {
                        winograd_f6k3_input_transform1(
                                svld1_f32(pg, &block[0][0 * simd_width]),
                                svld1_f32(pg, &block[1][0 * simd_width]),
                                svld1_f32(pg, &block[2][0 * simd_width]),
                                svld1_f32(pg, &block[3][0 * simd_width]),
                                svld1_f32(pg, &block[4][0 * simd_width]),
                                svld1_f32(pg, &block[5][0 * simd_width]),
                                svld1_f32(pg, &block[6][0 * simd_width]),
                                svld1_f32(pg, &block[7][0 * simd_width]),
                                &wd0, &wd1, &wd2, &wd3,&wd8, &wd9, &wd10, &wd11);

                 winograd_f6k3_input_transform1(
                                svld1_f32(pg, &block[0][1 * simd_width]),
                                svld1_f32(pg, &block[1][1 * simd_width]),
                                svld1_f32(pg, &block[2][1 * simd_width]),
                                svld1_f32(pg, &block[3][1 * simd_width]),
                                svld1_f32(pg, &block[4][1 * simd_width]),
                                svld1_f32(pg, &block[5][1 * simd_width]),
                                svld1_f32(pg, &block[6][1 * simd_width]),
                                svld1_f32(pg, &block[7][1 * simd_width]),
				&wd4, &wd5, &wd6, &wd7,&wd12, &wd13, &wd14, &wd15);
                }

        }


                  //svfloat32_t const_0_25__5_00 = svld1_f32rq(pg, datatmp);
         for (size_t col = 0; col < 1; col++) {
                svfloat32_t vout0, vout1, vout2, vout3, vout4, vout5, vout6, vout7;
                svfloat32x4_t vin0123create = svcreate4_f32(wd0, wd1, wd2, wd3);
                svfloat32x4_t vin4567create = svcreate4_f32(wd4, wd5, wd6, wd7);
                svfloat32x4_t vin0123create1 = svcreate4_f32(wd8, wd9, wd10, wd11);
                svfloat32x4_t vin4567create1 = svcreate4_f32(wd12, wd13, wd14, wd15);
                float vin0123[4*simd_width],vin4567[4*simd_width], vin01231[4*simd_width],vin45671[4*simd_width];
                 svst4(pg, &vin0123[0], vin0123create);
                 svst4(pg, &vin4567[0], vin4567create);
                 svst4(pg, &vin01231[0], vin0123create1);
                 svst4(pg, &vin45671[0], vin4567create1);
                int icol = 0;
                 winograd_f6k3_input_transform1(
                        svld1_f32( pg, &vin0123[0]), svld1_f32( pg, &vin0123[simd_width]), svld1_f32(pg, &vin0123[2*simd_width]), svld1_f32( pg, &vin0123[3*simd_width]),svld1_f32(pg, &vin4567[0]), svld1_f32( pg, &vin4567[simd_width]), svld1_f32(pg, &vin4567[2*simd_width]), svld1_f32(pg, &vin4567[3*simd_width]),
                        &vout0, &vout1, &vout2, &vout3, &vout4, &vout5, &vout6, &vout7);

		
                svst1_f32(pg, &transform[0], vout0);
                transform += transform_stride;
                svst1_f32(pg, &transform[0], vout1);
                transform += transform_stride;
                svst1_f32(pg, transform, vout2);
                transform += transform_stride;
                svst1_f32(pg, transform, vout3);
                transform += transform_stride;
                svst1_f32(pg, transform, vout4);
                transform += transform_stride;
                svst1_f32(pg, transform, vout5);
                transform += transform_stride;
                svst1_f32(pg, transform, vout6);
                transform += transform_stride;
		 svst1_f32(pg, transform, vout7);
                transform += transform_stride;
		
		svfloat32_t vout10, vout11, vout12, vout13, vout14, vout15, vout16, vout17;
		winograd_f6k3_input_transform1(
                        svld1_f32( pg, &vin01231[0]), svld1_f32( pg, &vin01231[simd_width]), svld1_f32(pg, &vin01231[2*simd_width]), svld1_f32( pg, &vin01231[3*simd_width]),svld1_f32(pg, &vin45671[0]), svld1_f32( pg, &vin45671[simd_width]), svld1_f32(pg, &vin45671[2*simd_width]), svld1_f32(pg, &vin45671[3*simd_width]),
                        &vout10, &vout11, &vout12, &vout13, &vout14, &vout15, &vout16, &vout17);
                svst1_f32(pg, transform, vout10);
		transform += transform_stride;
                svst1_f32(pg, transform, vout11);
                transform += transform_stride;
                svst1_f32(pg, transform, vout12);
                transform += transform_stride;
                svst1_f32(pg, transform, vout13);
                transform += transform_stride;
                svst1_f32(pg, transform, vout14);
                transform += transform_stride;
                svst1_f32(pg, transform, vout15);
                transform += transform_stride;
                svst1_f32(pg, transform, vout16);
                transform += transform_stride;
                svst1_f32(pg, transform, vout17);
                transform += transform_stride;
        }}//}
}



void nnp_iwt8x8_3x3_with_offset__neon_intertile1(
        const float *data[restrict static 1],
        void** transform,
        size_t data_stride,
        size_t transform_stride,
        uint32_t row_count,
        uint32_t column_count,
        uint32_t row_offset,
        uint32_t column_offset)
{
        //printf("I am in intertile neo function\n");
         int simd_width = nnp_hwinfo.sve_simd_width;//nnp_hwinfo.simd_width;
	//printf("sve simd_width = %d", nnp_hwinfo.sve_simd_width);
	const int interchannels	 = nnp_hwinfo.globalinterchannels;
        float *new_data, *new_data1;
        new_data = (float *)malloc(sizeof(float) * 8*8*interchannels);
        new_data1 = (float *)malloc(sizeof(float) * 8*8*interchannels);
	 if NNP_LIKELY(row_count == 8 && column_count == 8 && row_offset == 0 && column_offset == 0) {
        int tmp_width = 4;
	#pragma loop unroll_count(interchannels)
	for(int k=0;k<interchannels;k++){
        for(int i=0;i<8;i++)
        {
                for(int j=0;j<4;j++)
                {
                        	new_data[(i*simd_width)+((0*tmp_width)+(j+k*4))] =  data[k][(i * data_stride) + ((0 * tmp_width)+j)];
                        	new_data1[(i*simd_width)+((0*tmp_width)+(j+k*4))] = data[k][(i * data_stride) + ((1 * tmp_width)+j)];
                }
        	}
	}
	}
	 int tmp_stride = simd_width; //for tmp new_data and new_data1
          NNP_SIMD_ALIGN svfloat32_t wd0, wd1, wd2, wd3, wd4, wd5, wd6, wd7, wd8, wd9, wd10, wd11, wd12, wd13, wd14, wd15;
          for(int i1=0;i1<simd_width;i1+=svcntw())
          {
                svbool_t pg = svwhilelt_b32(i1,simd_width);
                if NNP_LIKELY(row_count == 8 && column_count == 8 && row_offset == 0 && column_offset == 0) {
                // Fast path where we can directly load `data` into `wd`.


                        winograd_f6k3_input_transform_intertile(
                                svld1_f32(pg, &new_data[0 * tmp_stride + 0 * simd_width]),
                                svld1_f32(pg, &new_data[1 * tmp_stride + 0 * simd_width]),
                                svld1_f32(pg, &new_data[2 * tmp_stride + 0 * simd_width]),
                                svld1_f32(pg, &new_data[3 * tmp_stride + 0 * simd_width]),
                                svld1_f32(pg, &new_data[4 * tmp_stride + 0 * simd_width]),
                                svld1_f32(pg, &new_data[5 * tmp_stride + 0 * simd_width]),
                                svld1_f32(pg, &new_data[6 * tmp_stride + 0 * simd_width]),
                                svld1_f32(pg, &new_data[7 * tmp_stride + 0 * simd_width]),
                                &wd0, &wd1, &wd2, &wd3,&wd8, &wd9, &wd10, &wd11);

                         winograd_f6k3_input_transform_intertile(
                                svld1_f32(pg, &new_data1[0 * tmp_stride + 0 * simd_width]),
                                svld1_f32(pg, &new_data1[1 * tmp_stride + 0 * simd_width]),
                                svld1_f32(pg, &new_data1[2 * tmp_stride + 0 * simd_width]),
                                svld1_f32(pg, &new_data1[3 * tmp_stride + 0 * simd_width]),
                                svld1_f32(pg, &new_data1[4 * tmp_stride + 0 * simd_width]),
                                svld1_f32(pg, &new_data1[5 * tmp_stride + 0 * simd_width]),
                                svld1_f32(pg, &new_data1[6 * tmp_stride + 0 * simd_width]),
                                svld1_f32(pg, &new_data1[7 * tmp_stride + 0 * simd_width]),
                                &wd4, &wd5, &wd6, &wd7,&wd12, &wd13, &wd14, &wd15);

        } 
	else {
	                NNP_SIMD_ALIGN float block[16][simd_width*2];
                {
                        const svfloat32_t vzero = svdup_f32(0.0f);
                        //for (float *block_ptr = &block[0][0], *block_end = &block[8][0]; block_ptr != block_end; block_ptr += simd_width) {
                        for (float *block_ptr = &block[0][0], *block_end = &block[16][0]; block_ptr != block_end; block_ptr += simd_width) {
                                svst1_f32(pg, block_ptr, vzero);
                        }
                       // for (float *block_ptr = &block1[0][0], *block_end = &block1[8][0]; block_ptr != block_end; block_ptr += simd_width) {
                         //       svst1_f32(pg, block_ptr, vzero);
                       // }
                }
         #pragma loop unroll_count(interchannels)
        for(int k=0;k<interchannels;k++){ 
	       for (size_t i = 0; i < row_count; i++) {
                        for (size_t j = 0; j < (column_count); j++) {
                		if(j <4){ 
		               block[row_offset + i][column_offset + (j+k*4)] = data[k][i * data_stride + j];
				}
				else{
                                block[row_offset + (i+8)][column_offset + ((j+k*4)-4)] = data[k][i * data_stride + (j)];
                        	}
			}
                }
	}
                for (size_t col = 0; col < 1; col++) {
                        winograd_f6k3_input_transform_intertile(
                                svld1_f32(pg, &block[0][0 * simd_width]),
                                svld1_f32(pg, &block[1][0 * simd_width]),
                                svld1_f32(pg, &block[2][0 * simd_width]),
                                svld1_f32(pg, &block[3][0 * simd_width]),
                                svld1_f32(pg, &block[4][0 * simd_width]),
                                svld1_f32(pg, &block[5][0 * simd_width]),
                                svld1_f32(pg, &block[6][0 * simd_width]),
                                svld1_f32(pg, &block[7][0 * simd_width]),
                                &wd0, &wd1, &wd2, &wd3,&wd8, &wd9, &wd10, &wd11);

                 winograd_f6k3_input_transform_intertile(
                                svld1_f32(pg, &block[0+8][0 * simd_width]),
                                svld1_f32(pg, &block[1+8][0 * simd_width]),
                                svld1_f32(pg, &block[2+8][0 * simd_width]),
                                svld1_f32(pg, &block[3+8][0 * simd_width]),
                                svld1_f32(pg, &block[4+8][0 * simd_width]),
                                svld1_f32(pg, &block[5+8][0 * simd_width]),
                                svld1_f32(pg, &block[6+8][0 * simd_width]),
                                svld1_f32(pg, &block[7+8][0 * simd_width]),
                                &wd4, &wd5, &wd6, &wd7,&wd12, &wd13, &wd14, &wd15);
                }
	        }


                  //svfloat32_t const_0_25__5_00 = svld1_f32rq(pg, datatmp);
         for (size_t col = 0; col < 1; col++) {
                svfloat32_t vout0, vout1, vout2, vout3, vout4, vout5, vout6, vout7;
                svfloat32x4_t vin0123create = svcreate4_f32(wd0, wd1, wd2, wd3);
                svfloat32x4_t vin4567create = svcreate4_f32(wd4, wd5, wd6, wd7);
                svfloat32x4_t vin0123create1 = svcreate4_f32(wd8, wd9, wd10, wd11);
                svfloat32x4_t vin4567create1 = svcreate4_f32(wd12, wd13, wd14, wd15);
                float vin0123[4*simd_width],vin4567[4*simd_width], vin01231[4*simd_width],vin45671[4*simd_width];
                 svst4(pg, &vin0123[0], vin0123create);
                 svst4(pg, &vin4567[0], vin4567create);
                 svst4(pg, &vin01231[0], vin0123create1);
                 svst4(pg, &vin45671[0], vin4567create1);
                int icol = 0;
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
	      winograd_f6k3_input_transform_intertile(
                        svld1_gather_s32index_f32( pg, &vin0123[0],index1), svld1_gather_s32index_f32( pg, &vin0123[0], index2), svld1_gather_s32index_f32(pg, &vin0123[0], index3), svld1_gather_s32index_f32( pg, &vin0123[0], index4),svld1_gather_s32index_f32(pg, &vin4567[0], index1), svld1_gather_s32index_f32( pg, &vin4567[0], index2), svld1_gather_s32index_f32(pg, &vin4567[0], index3), svld1_gather_s32index_f32(pg, &vin4567[0], index4),
                        &vout0, &vout1, &vout2, &vout3, &vout4, &vout5, &vout6, &vout7);


                svfloat32_t vout10, vout11, vout12, vout13, vout14, vout15, vout16, vout17;
                winograd_f6k3_input_transform_intertile(
                        svld1_gather_s32index_f32( pg, &vin01231[0],index1), svld1_gather_s32index_f32( pg, &vin01231[0], index2), svld1_gather_s32index_f32(pg, &vin01231[0], index3), svld1_gather_s32index_f32( pg, &vin01231[0], index4),svld1_gather_s32index_f32(pg, &vin45671[0], index1), svld1_gather_s32index_f32( pg, &vin45671[0], index2), svld1_gather_s32index_f32(pg, &vin45671[0], index3), svld1_gather_s32index_f32(pg, &vin45671[0], index4),&vout10, &vout11, &vout12, &vout13, &vout14, &vout15, &vout16, &vout17);
                 

                //float tmp_transform[16][simd_width];
                float tmp_transform[16*simd_width];
                svst1_f32(pg, &tmp_transform[0], vout0);
                svst1_f32(pg, &tmp_transform[1*simd_width], vout1);
                svst1_f32(pg, &tmp_transform[2*simd_width], vout2);
                svst1_f32(pg, &tmp_transform[3*simd_width], vout3);
                svst1_f32(pg, &tmp_transform[4*simd_width], vout4);
                svst1_f32(pg, &tmp_transform[5*simd_width], vout5);
                svst1_f32(pg, &tmp_transform[6*simd_width], vout6);
                svst1_f32(pg, &tmp_transform[7*simd_width], vout7);
                svst1_f32(pg, &tmp_transform[8*simd_width], vout10);
                svst1_f32(pg, &tmp_transform[9*simd_width], vout11);
                svst1_f32(pg, &tmp_transform[10*simd_width], vout12);
                svst1_f32(pg, &tmp_transform[11*simd_width], vout13);
                svst1_f32(pg, &tmp_transform[12*simd_width], vout14);
                svst1_f32(pg, &tmp_transform[13*simd_width], vout15);
                svst1_f32(pg, &tmp_transform[14*simd_width], vout16);
                svst1_f32(pg, &tmp_transform[15*simd_width], vout17);
        

		/*  512-bit*/
         
		/*for (size_t i = 0; i < 16; i++) {
                        for (size_t j = 0; j < 4; j++) {
                                *(((float *)transform[0])+j)   = tmp_transform[i*simd_width+j];
                                *(((float *)transform[1])+j)   = tmp_transform[i*simd_width + (j+4)];
                                *(((float *)transform[2])+j)   = tmp_transform[i*simd_width + (j+8)];
                                *(((float *)transform[3])+j)   = tmp_transform[i*simd_width + (j+12)];
                        }
                         transform[0] += transform_stride;
                        transform[1] += transform_stride;
                        transform[2] += transform_stride;
                        transform[3] += transform_stride;
                }*/
		 for (size_t i = 0; i < 16; i++) {
                        for (size_t j = 0; j < 4; j++) {
                                *(((float *)transform[0])+j)   = tmp_transform[i*simd_width+j];
                                *(((float *)transform[1])+j)   = tmp_transform[i*simd_width + (j+4)];
                                if(interchannels > 2) {*(((float *)transform[2])+j)   = tmp_transform[i*simd_width + (j+8)];}
                                if(interchannels > 3) {*(((float *)transform[3])+j)   = tmp_transform[i*simd_width + (j+12)];}
                                if(interchannels > 4) {*(((float *)transform[4])+j)   = tmp_transform[i*simd_width + (j+16)];}
                                if(interchannels > 5){*(((float *)transform[5])+j)   = tmp_transform[i*simd_width + (j+20)];}
                                if(interchannels > 6) {*(((float *)transform[6])+j)   = tmp_transform[i*simd_width + (j+24)];}
                                if(interchannels > 7) { *(((float *)transform[7])+j)   = tmp_transform[i*simd_width + (j+28)];}
                        	if(interchannels > 8) {*(((float *)transform[8])+j)   = tmp_transform[i*simd_width + (j+32)];}
                                if(interchannels > 9) {*(((float *)transform[9])+j)   = tmp_transform[i*simd_width + (j+36)];}
                                if(interchannels > 10) {*(((float *)transform[10])+j)   = tmp_transform[i*simd_width + (j+40)];}
                                if(interchannels > 11){*(((float *)transform[11])+j)   = tmp_transform[i*simd_width + (j+44)];}
                                if(interchannels > 12) {*(((float *)transform[12])+j)   = tmp_transform[i*simd_width + (j+48)];}
                                if(interchannels > 13) { *(((float *)transform[13])+j)   = tmp_transform[i*simd_width + (j+52)];}
                                if(interchannels > 14) { *(((float *)transform[14])+j)   = tmp_transform[i*simd_width + (j+56)];}
                                if(interchannels > 15) { *(((float *)transform[15])+j)   = tmp_transform[i*simd_width + (j+60)];}

			}
                         transform[0] += transform_stride;
                        transform[1] += transform_stride;
                        if(interchannels > 2) {transform[2] += transform_stride;}
                        if(interchannels > 3) {transform[3] += transform_stride;}
                         if(interchannels > 4) {transform[4] += transform_stride;}
                        if(interchannels > 5) {transform[5] += transform_stride;}
                        if(interchannels > 6) {transform[6] += transform_stride;}
                        if(interchannels > 7) {transform[7] += transform_stride;}
			 if(interchannels > 8) {transform[8] += transform_stride;}
                        if(interchannels > 9) {transform[9] += transform_stride;}
                         if(interchannels > 10) {transform[10] += transform_stride;}
                        if(interchannels > 11) {transform[11] += transform_stride;}
                        if(interchannels > 12) {transform[12] += transform_stride;}
                        if(interchannels > 13) {transform[13] += transform_stride;}
                        if(interchannels > 14) {transform[14] += transform_stride;}
                        if(interchannels > 15) {transform[15] += transform_stride;}

                }


		/*origbal	*/
	/*	for(int k=0;k<interchannels;k++)
		{
		   for (size_t i = 0; i < 16; i++) {
                        for (size_t j = 0; j < 4; j++) {
                                *(((float *)transform[k])+j)   = tmp_transform[i*simd_width+(j+(k*4))];
                        }
                         transform[k] += transform_stride;
                }	
		}	*/

	
        }}//}
	free(new_data);
	free(new_data1);
}
		
void nnp_kwt8x8_3x3__neon_intertile(
        const float *g[restrict static 9],
        float *transform[restrict static 1],
        size_t stride_g,
        size_t transform_stride,
        uint32_t row_count,
        uint32_t column_count,
        uint32_t row_offset,
        uint32_t column_offset)
{
	 int simd_width = nnp_hwinfo.sve_simd_width;//nnp_hwinfo.simd_width;
         int interchannels  = nnp_hwinfo.globalinterchannels;

	transform_stride /= sizeof(float);
	float *g0;
        g0 = (float *) malloc(sizeof(float) * 8*8*interchannels);        
	for(int k=0;k<interchannels;k++){  
	for(int j=0;j<4;j++)
        {       
                g0[(0*simd_width)+(j+(k*4))] =  g[k][j];
                g0[(1*simd_width)+(j+(k*4))] =  g[k][3+j];

      	  if(j<3){       
                g0[(2*simd_width)+(j+(k*4))] =  g[k][6+j];
		}
	}
        g0[(2*simd_width)+(3+(k*4))] =  g[k][5];
	}

        for(int i1=0;i1<simd_width;i1+=svcntw())
        {
        svbool_t pg = svwhilelt_b32(i1,simd_width);
         svfloat32_t g0_vec = svld1_f32(pg, &g0[0]);
         svfloat32_t g1_vec = svld1_f32(pg, &g0[1*simd_width+0]);
        // g2[3] is junk
         svfloat32_t g2_vec = svld1_f32(pg, &g0[2*simd_width+0]);
        svfloat32_t w0, w1, w2, w3, w4, w5, w6, w7;
        winograd_f6k3_kernel_transform_intertile(g0_vec, g1_vec, g2_vec,
                &w0, &w1, &w2, &w3, &w4, &w5, &w6, &w7,
                true /* rescale coefficients */);
        
	neon_transpose4x4_inplace_f321_intertile(&w0, &w1, &w2, &w3);
        neon_transpose4x4_inplace_f321_intertile(&w4, &w5, &w6, &w7);
        svfloat32_t wg00, wg10, wg20, wg30, wg40, wg50, wg60, wg70, wg01, wg11, wg21, wg31, wg41, wg51, wg61, wg71;
        winograd_f6k3_kernel_transform_intertile(w0, w1, w2,
                &wg00, &wg10, &wg20, &wg30,
                &wg40, &wg50, &wg60, &wg70,
                true /* rescale coefficients */);
       
	 winograd_f6k3_kernel_transform_intertile(w4, w5, w6,
                &wg01, &wg11, &wg21, &wg31,
                &wg41, &wg51, &wg61, &wg71,
                true /* rescale coefficients */);
        float tmp_transform[16*simd_width];
         svst1_f32(pg, &tmp_transform[0], wg00);
        svst1_f32(pg, &tmp_transform[1*simd_width], wg10);
        svst1_f32(pg, &tmp_transform[2*simd_width], wg20);
        svst1_f32(pg, &tmp_transform[3*simd_width], wg30);
        svst1_f32(pg, &tmp_transform[4*simd_width], wg40);
        svst1_f32(pg, &tmp_transform[5*simd_width], wg50);
        svst1_f32(pg, &tmp_transform[6*simd_width], wg60);
        svst1_f32(pg, &tmp_transform[7*simd_width], wg70);
        svst1_f32(pg, &tmp_transform[8*simd_width], wg01);
        svst1_f32(pg, &tmp_transform[9*simd_width], wg11);
        svst1_f32(pg, &tmp_transform[10*simd_width], wg21);
        svst1_f32(pg, &tmp_transform[11*simd_width], wg31);
        svst1_f32(pg, &tmp_transform[12*simd_width], wg41);
        svst1_f32(pg, &tmp_transform[13*simd_width], wg51);
        svst1_f32(pg, &tmp_transform[14*simd_width], wg61);
        svst1_f32(pg, &tmp_transform[15*simd_width], wg71);
        
        for (size_t i = 0; i < 16; i++) {
            #pragma loop unroll_count(interchannels)
            for(int k=0;k<interchannels;k++){
		for (size_t j = 0; j < 4; j++) {
                        *(transform[k]+j)   = tmp_transform[i*simd_width+(j+k*4)];
                
		       }
                transform[k] += transform_stride;
        }
      }
   }
	free(g0);
}
       

void nnp_kwt8x8_3x3__neon(
	const float g[restrict static 9],
	float transform[restrict static 1],
	size_t stride_g,
	size_t transform_stride,
	uint32_t row_count,
	uint32_t column_count,
	uint32_t row_offset,
	uint32_t column_offset)
{
 int simd_width = nnp_hwinfo.simd_width;
//printf("stride_g=%d", stride_g); 
for(int i1=0;i1<simd_width;i1+=svcntw())
                {
                //printf("I am going in sve");
                svbool_t pg = svwhilelt_b32(i1,simd_width);

        transform_stride /= sizeof(float);

        const svfloat32_t g0 = svld1(pg, g);
        const svfloat32_t g1 = svld1(pg, g + 3);
        // g2[3] is junk
    const float32x4_t g2_neon = vextq_f32(vld1q_f32(g + 5), vld1q_f32(g + 5), 1);
        float check[simd_width];
        vst1q_f32(&check[0],g2_neon);
        svfloat32_t g2 = svld1(pg, &check[0]);

         // const svfloat32_t g2 = svext_f32(svld1(pg, g + 5), svld1(pg, g + 5), 5);
         svfloat32_t w0, w1, w2, w3, w4, w5, w6, w7;
        winograd_f6k3_kernel_transform1(g0, g1, g2,
                &w0, &w1, &w2, &w3, &w4, &w5, &w6, &w7,
                true /* rescale coefficients */);
        neon_transpose4x4_inplace_f321(&w0, &w1, &w2, &w3);
        neon_transpose4x4_inplace_f321(&w4, &w5, &w6, &w7);

        //NNP_SIMD_ALIGN svfloat32x4_t wg[8][2];
         svfloat32_t wg00, wg10, wg20, wg30, wg40, wg50, wg60, wg70, wg01, wg11, wg21, wg31, wg41, wg51, wg61, wg71;
        winograd_f6k3_kernel_transform1(w0, w1, w2,
                &wg00, &wg10, &wg20, &wg30,
                &wg40, &wg50, &wg60, &wg70,
                true /* rescale coefficients */);
        winograd_f6k3_kernel_transform1(w4, w5, w6,
                &wg01, &wg11, &wg21, &wg31,
                &wg41, &wg51, &wg61, &wg71,
                true /* rescale coefficients */);

	svst1(pg, transform, wg00);
        transform += transform_stride;
        svst1(pg, transform, wg10);
        transform += transform_stride;
        svst1(pg, transform, wg20);
        transform += transform_stride;
        svst1(pg, transform, wg30);
        transform += transform_stride;
        svst1(pg, transform, wg40);
        transform += transform_stride;
        svst1(pg, transform, wg50);
        transform += transform_stride;
        svst1(pg, transform, wg60);
        transform += transform_stride;
        svst1(pg, transform, wg70);
        transform += transform_stride;
        svst1(pg, transform, wg01);
        transform += transform_stride;
        svst1(pg, transform, wg11);
        transform += transform_stride;
        svst1(pg, transform, wg21);
        transform += transform_stride;
        svst1(pg, transform, wg31);
        transform += transform_stride;
        svst1(pg, transform, wg41);
        transform += transform_stride;
        svst1(pg, transform, wg51);
        transform += transform_stride;
        svst1(pg, transform, wg61);
        transform += transform_stride;
        svst1(pg, transform, wg71);
        transform += transform_stride;


}}

#if !NNP_INFERENCE_ONLY
void nnp_kwt8x8_3Rx3R__neon(
	const float g[restrict static 9],
	float transform[restrict static 1],
	size_t stride_g,
	size_t transform_stride,
	uint32_t row_count,
	uint32_t column_count,
	uint32_t row_offset,
	uint32_t column_offset)
{
	transform_stride /= sizeof(float);

	const float32x4_t g5678 = vld1q_f32(g + 5);
	const float32x4_t g2345 = vld1q_f32(g + 2);
	const float32x4_t g0123 = vld1q_f32(g);

	/* g0 = { g[8], g[7], g[6], g[6] }; */
	const float32x4_t g0 = vcombine_f32(vrev64_f32(vld1_f32(&g[7])), vld1_dup_f32(&g[6]));
	/* g1 = { g[5], g[4], g[3], g[3] }; */
	const float32x4_t g1 = vcombine_f32(vrev64_f32(vld1_f32(&g[4])), vld1_dup_f32(&g[3]));
	/* g2 = { g[2], g[1], g[0], g[0] }; */
	const float32x4_t g2 = vcombine_f32(vrev64_f32(vld1_f32(&g[1])), vld1_dup_f32(&g[0]));

	NNP_SIMD_ALIGN float32x4_t w[8];
	winograd_f6k3_kernel_transform(g0, g1, g2,
		&w[0], &w[1], &w[2], &w[3], &w[4], &w[5], &w[6], &w[7],
		true /* rescale coefficients */);
	neon_transpose4x4_inplace_f32(&w[0], &w[1], &w[2], &w[3]);
	neon_transpose4x4_inplace_f32(&w[4], &w[5], &w[6], &w[7]);

	NNP_SIMD_ALIGN float32x4_t wg[8][2];
	winograd_f6k3_kernel_transform(w[0], w[1], w[2],
		&wg[0][0], &wg[1][0], &wg[2][0], &wg[3][0],
		&wg[4][0], &wg[5][0], &wg[6][0], &wg[7][0],
		true /* rescale coefficients */);
	winograd_f6k3_kernel_transform(w[4], w[5], w[6],
		&wg[0][1], &wg[1][1], &wg[2][1], &wg[3][1],
		&wg[4][1], &wg[5][1], &wg[6][1], &wg[7][1],
		true /* rescale coefficients */);

	for (size_t col = 0; col < 2; col++) {
		for (size_t row = 0; row < 8; row++) {
			vst1q_f32(transform, wg[row][col]);
			transform += transform_stride;
		}
	}
}

void nnp_owt8x8_3x3__neon(
	const void *restrict transform,
	float output[restrict static 1],
	size_t transform_stride,
	size_t output_stride,
	uint32_t row_count,
	uint32_t column_count,
	uint32_t row_offset,
	uint32_t column_offset)
{

	NNP_SIMD_ALIGN float buffer[8 * 6];
	float*restrict qbuffer = buffer;
	float*restrict dbuffer = buffer + 32;
                 for(int i1=0;i1<4;i1+=svcntw())
                {
         //       printf("I am going in sve");
                svbool_t pg = svwhilelt_b32(i1,4);
	svfloat32_t m0,m1,m2,m3,m4,m5,m6,m7;
		svbool_t  pg_high = svunpkhi_b(pg);
		svbool_t  pg_low = svunpklo_b(pg);
		 m0 = svld1_f32(pg, transform); transform += transform_stride;
		 m1 = svld1_f32(pg, transform); transform += transform_stride;
		 m2 = svld1_f32(pg, transform); transform += transform_stride;
		 m3 = svld1_f32(pg, transform); transform += transform_stride;
		 m4 = svld1_f32(pg, transform); transform += transform_stride;
		 m5 = svld1_f32(pg, transform); transform += transform_stride;
		 m6 = svld1_f32(pg, transform); transform += transform_stride;
		 m7 = svld1_f32(pg, transform); transform += transform_stride;
		svfloat32_t o0, o1, o2, o3, o4, o5;
		winograd_f6k3_output_transformq1(
			m0, m1, m2, m3, m4, m5, m6, m7,
			&o0, &o1, &o2, &o3, &o4, &o5);
		svst1_f32(pg, qbuffer, o0); qbuffer += 4;
		svst1_f32(pg, qbuffer, o1); qbuffer += 4;
		svst1_f32(pg, qbuffer, o2); qbuffer += 4;
		svst1_f32(pg, qbuffer, o3); qbuffer += 4;
		svst1_f32(pg_low, dbuffer, o4); dbuffer += 2;
		svst1_f32(pg_low, dbuffer, o5); dbuffer += 2;
		svst1_f32(pg_high, dbuffer, o4); dbuffer += 2;
		svst1_f32(pg_high, dbuffer, o5); dbuffer += 2;
		//col1
                 m0 = svld1_f32(pg, transform); transform += transform_stride;
                 m1 = svld1_f32(pg, transform); transform += transform_stride;
                 m2 = svld1_f32(pg, transform); transform += transform_stride;
                 m3 = svld1_f32(pg, transform); transform += transform_stride;
                 m4 = svld1_f32(pg, transform); transform += transform_stride;
                 m5 = svld1_f32(pg, transform); transform += transform_stride;
                 m6 = svld1_f32(pg, transform); transform += transform_stride;
                 m7 = svld1_f32(pg, transform); transform += transform_stride;
                winograd_f6k3_output_transformq1(
                        m0, m1, m2, m3, m4, m5, m6, m7,
                        &o0, &o1, &o2, &o3, &o4, &o5);
                svst1_f32(pg, qbuffer, o0); qbuffer += 4;
                svst1_f32(pg, qbuffer, o1); qbuffer += 4;
                svst1_f32(pg, qbuffer, o2); qbuffer += 4;
                svst1_f32(pg, qbuffer, o3); qbuffer += 4;
                svst1_f32(pg_low, dbuffer, o4); dbuffer += 2;
                svst1_f32(pg_low, dbuffer, o5); dbuffer += 2;
                svst1_f32(pg_high, dbuffer, o4); dbuffer += 2;
                svst1_f32(pg_high, dbuffer, o5); dbuffer += 2;


	const float*restrict read_ptr = buffer;
	if NNP_LIKELY(row_count == 6 && column_count == 6 && output_stride >= 6) {
		// Fast path to reuse `s` array and write directly into `output`.
		svfloat32_t a0,a1,a2,a3, a4, a5, a6, a7;
		a0 = svld1_f32(pg,  &read_ptr[0]);
		a1 = svld1_f32(pg,  &read_ptr[4]);
		a2 = svld1_f32(pg,  &read_ptr[8]);
		a3 = svld1_f32(pg,  &read_ptr[12]);
		a4 = svld1_f32(pg,  &read_ptr[16]);
		a5 = svld1_f32(pg,  &read_ptr[20]);
		a6 = svld1_f32(pg,  &read_ptr[24]);
		a7 = svld1_f32(pg,  &read_ptr[28]);
		svfloat32x4_t inp0123 =  svcreate4(a0,a1,a2,a3);
		svfloat32x4_t inp4567 =  svcreate4(a4,a5,a6,a7);
		  float  qin0123[16], qin4567[16];
		svst4(pg, &qin0123[0], inp0123);
		svst4(pg, &qin4567[0], inp4567);
		//float32x4x4_t qin0123 = vld4q_f32(read_ptr); read_ptr += 16;
		//float32x4x4_t qin4567 = vld4q_f32(read_ptr); read_ptr += 16;
		svfloat32_t qout0, qout1, qout2, qout3, qout4, qout5;
		winograd_f6k3_output_transformq1(
			svld1_f32( pg, &qin0123[0]),svld1_f32( pg, &qin0123[4]), svld1_f32( pg, &qin0123[8]), svld1_f32( pg, &qin0123[12]), svld1_f32( pg, &qin4567[0]),svld1_f32( pg, &qin4567[4]), svld1_f32( pg, &qin4567[8]), svld1_f32( pg, &qin4567[12]),  &qout0, &qout1, &qout2, &qout3, &qout4, &qout5); 
		float* output_col0123 = output;
		svst1_f32(pg, output_col0123, qout0); output_col0123 += output_stride;
		svst1_f32(pg, output_col0123, qout1); output_col0123 += output_stride;
		svst1_f32(pg, output_col0123, qout2); output_col0123 += output_stride;
		svst1_f32(pg, output_col0123, qout3); output_col0123 += output_stride;
		svst1_f32(pg, output_col0123, qout4); output_col0123 += output_stride;
		svst1_f32(pg, output_col0123, qout5);
		
                a0 = svld1(pg,  &read_ptr[32]);
                a1 = svld1(pg,  &read_ptr[34]);
                a2 = svld1(pg,  &read_ptr[36]);
                a3 = svld1(pg,  &read_ptr[38]);
                a4 = svld1(pg,  &read_ptr[40]);
                a5 = svld1(pg,  &read_ptr[42]);
                a6 = svld1(pg,  &read_ptr[44]);
                a7 = svld1(pg,  &read_ptr[46]);
		svfloat32x2_t din01 =  svcreate2(a0,a1);
                svfloat32x2_t din23 =  svcreate2(a2, a3);
                svfloat32x2_t din45 =  svcreate2(a4, a5);
                svfloat32x2_t din67 =  svcreate2(a6, a7);
		 float  din101[16], din123[16], din145[16], din167[16];
                svst2(pg, &din101[0], din01);
                svst2(pg, &din123[0], din23);	
                svst2(pg, &din145[0], din45);
                svst2(pg, &din167[0], din67);	

///		float32x2x2_t din01 = vld2_f32(read_ptr); read_ptr += 4;
//		float32x2x2_t din23 = vld2_f32(read_ptr); read_ptr += 4;
//		float32x2x2_t din45 = vld2_f32(read_ptr); read_ptr += 4;
//		float32x2x2_t din67 = vld2_f32(read_ptr);
		svfloat32_t dout0, dout1, dout2, dout3, dout4, dout5;
		winograd_f6k3_output_transform1(
			svld1_f32( pg, &din101[0]), svld1_f32( pg, &din101[2]), svld1_f32( pg, &din123[0]), svld1_f32( pg, &din123[2]), svld1_f32( pg, &din145[0]), svld1_f32( pg, &din145[2]), svld1_f32( pg, &din167[0]), svld1_f32( pg, &din167[2]), &dout0, &dout1, &dout2, &dout3, &dout4, &dout5);
		float* output_col45 = output + 4;
		svst1_f32(pg, output_col45, dout0); output_col45 += output_stride;
		svst1_f32(pg, output_col45, dout1); output_col45 += output_stride;
		svst1_f32(pg, output_col45, dout2); output_col45 += output_stride;
		svst1_f32(pg, output_col45, dout3); output_col45 += output_stride;
		svst1_f32(pg, output_col45, dout4); output_col45 += output_stride;
		svst1_f32(pg, output_col45, dout5);
	} else {
		NNP_SIMD_ALIGN float block[6][8];
		 svfloat32_t a0,a1,a2,a3, a4, a5, a6, a7;
                a0=svld1(pg,  &read_ptr[0]);
                a1=svld1(pg,  &read_ptr[4]);
                a2=svld1(pg,  &read_ptr[8]);
                a3=svld1(pg,  &read_ptr[12]);
                a4=svld1(pg,  &read_ptr[16]);
                a5=svld1(pg,  &read_ptr[20]);
                a6=svld1(pg,  &read_ptr[24]);
                a7=svld1(pg,  &read_ptr[28]);
                svfloat32x4_t inp0123 =  svcreate4(a0,a1,a2,a3);
                svfloat32x4_t inp4567 =  svcreate4(a4,a5,a6,a7);
                  float  qin0123[16], qin4567[16];
                svst4(pg, &qin0123[0], inp0123);
                svst4(pg, &qin4567[0], inp4567);
		svfloat32_t qout0, qout1, qout2, qout3, qout4, qout5;
		winograd_f6k3_output_transformq1(
                        svld1_f32( pg, &qin0123[0]),svld1_f32( pg, &qin0123[4]), svld1_f32( pg, &qin0123[8]), svld1_f32( pg, &qin0123[12]), svld1_f32( pg, &qin4567[0]),svld1_f32( pg, &qin4567[4]), svld1_f32( pg, &qin4567[8]), svld1_f32( pg, &qin4567[12]),  &qout0, &qout1, &qout2, &qout3, &qout4, &qout5);

	///	float32x4x4_t qin0123 = vld4q_f32(read_ptr); read_ptr += 16;
	//	float32x4x4_t qin4567 = vld4q_f32(read_ptr); read_ptr += 16;
		//winograd_f6k3_output_transformq(
		//	qin0123.val[0], qin0123.val[1], qin0123.val[2], qin0123.val[3],
		//	qin4567.val[0], qin4567.val[1], qin4567.val[2], qin4567.val[3],
		//	&qout0, &qout1, &qout2, &qout3, &qout4, &qout5);
		svst1_f32(pg, &block[0][0], qout0);
		svst1_f32(pg, &block[1][0], qout1);
		svst1_f32(pg, &block[2][0], qout2);
		svst1_f32(pg, &block[3][0], qout3);
		svst1_f32(pg, &block[4][0], qout4);
		svst1_f32(pg, &block[5][0], qout5);

		a0 =  svld1(pg,  &read_ptr[32]);
                a1 = svld1(pg,  &read_ptr[34]);
                a2 = svld1(pg,  &read_ptr[36]);
                a3 = svld1(pg,  &read_ptr[38]);
                a4 = svld1(pg,  &read_ptr[40]);
                a5 = svld1(pg,  &read_ptr[42]);
                a6 = svld1(pg,  &read_ptr[44]);
                a7 = svld1(pg,  &read_ptr[46]);
                svfloat32x2_t din01 =  svcreate2(a0,a1);
                svfloat32x2_t din23 =  svcreate2(a2, a3);
                svfloat32x2_t din45 =  svcreate2(a4, a5);
                svfloat32x2_t din67 =  svcreate2(a6, a7);
                 float  din101[16], din123[16], din145[16], din167[16];
                svst2(pg, &din101[0], din01);
                svst2(pg, &din123[0], din23);
                svst2(pg, &din145[0], din45);
                svst2(pg, &din167[0], din67);


//		float32x2x2_t din01 = vld2_f32(read_ptr); read_ptr += 4;
//		float32x2x2_t din23 = vld2_f32(read_ptr); read_ptr += 4;
//		float32x2x2_t din45 = vld2_f32(read_ptr); read_ptr += 4;
//		float32x2x2_t din67 = vld2_f32(read_ptr);
		svfloat32_t dout0, dout1, dout2, dout3, dout4, dout5;

		winograd_f6k3_output_transform1(
                        svld1_f32( pg, &din101[0]), svld1_f32( pg, &din101[2]), svld1_f32( pg, &din123[0]), svld1_f32( pg, &din123[2]), svld1_f32( pg, &din145[0]), svld1_f32( pg, &din145[2]), svld1_f32( pg, &din167[0]), svld1_f32( pg, &din167[2]), &dout0, &dout1, &dout2, &dout3, &dout4, &dout5);
//		winograd_f6k3_output_transform(
///			din01.val[0], din01.val[1], din23.val[0], din23.val[1],
//			din45.val[0], din45.val[1], din67.val[0], din67.val[1],
//			&dout0, &dout1, &dout2, &dout3, &dout4, &dout5);
		svst1_f32(pg, &block[0][4], dout0);
		svst1_f32(pg, &block[1][4], dout1);
		svst1_f32(pg, &block[2][4], dout2);
		svst1_f32(pg, &block[3][4], dout3);
		svst1_f32(pg, &block[4][4], dout4);
		svst1_f32(pg, &block[5][4], dout5);

		for (size_t i = 0; i < row_count; i++) {
			for (size_t j = 0; j < column_count; j++) {
				output[i * output_stride + j] = block[i][j];
			}
		}
	}
	}
}
#endif /* !NNP_INFERENCE_ONLY */


void nnp_owt8x8_3x3_with_bias__neon1(
        const void *restrict transform,
        float output[restrict static 1],
        const float bias[restrict static 1],
        size_t transform_stride,
        size_t output_stride,
        uint32_t row_count,
        uint32_t column_count)
{
        NNP_SIMD_ALIGN float buffer[8 * 6];
        float*restrict qbuffer = buffer;
        float*restrict dbuffer = buffer + 32;
       // float32x2_t vbias = vreinterpret_f32_u64(vshl_n_u64(vreinterpret_u64_f32(vld1_dup_f32(bias)), 32));  //by sonia
        for (uint32_t col = 0; col < 2; col++) {
                const float32x4_t m0 = vld1q_f32(transform); transform += transform_stride;
                float32x4_t m1 = vld1q_f32(transform); transform += transform_stride;
                /* The only difference in the with_bias vs non with_bias case. */
               // m1 = vcombine_f32(vadd_f32(vget_low_f32(m1), vbias), vget_high_f32(m1)); //by sonia
               // vbias = vmov_n_f32(0.0f);  //by sonia
                const float32x4_t m2 = vld1q_f32(transform); transform += transform_stride;
                const float32x4_t m3 = vld1q_f32(transform); transform += transform_stride;
                const float32x4_t m4 = vld1q_f32(transform); transform += transform_stride;
                const float32x4_t m5 = vld1q_f32(transform); transform += transform_stride;
                const float32x4_t m6 = vld1q_f32(transform); transform += transform_stride;
                const float32x4_t m7 = vld1q_f32(transform); transform += transform_stride;
                float32x4_t o0, o1, o2, o3, o4, o5;
                winograd_f6k3_output_transformq(
                        m0, m1, m2, m3, m4, m5, m6, m7,
                        &o0, &o1, &o2, &o3, &o4, &o5);
                vst1q_f32(qbuffer, o0); qbuffer += 4;
                vst1q_f32(qbuffer, o1); qbuffer += 4;
                vst1q_f32(qbuffer, o2); qbuffer += 4;
                vst1q_f32(qbuffer, o3); qbuffer += 4;
                vst1_f32(dbuffer, vget_low_f32(o4)); dbuffer += 2;
                vst1_f32(dbuffer, vget_low_f32(o5)); dbuffer += 2;
                vst1_f32(dbuffer, vget_high_f32(o4)); dbuffer += 2;
                vst1_f32(dbuffer, vget_high_f32(o5)); dbuffer += 2;
        }

	 const float*restrict read_ptr = buffer;
        if NNP_LIKELY(row_count == 6 && column_count == 6 && output_stride >= 6) {
                // Fast path to reuse `s` array and write directly into `output`.
                float32x4x4_t qin0123 = vld4q_f32(read_ptr); read_ptr += 16;
                float32x4x4_t qin4567 = vld4q_f32(read_ptr); read_ptr += 16;
                float32x4_t qout0, qout1, qout2, qout3, qout4, qout5;
                winograd_f6k3_output_transformq(
                        qin0123.val[0], qin0123.val[1], qin0123.val[2], qin0123.val[3],
                        qin4567.val[0], qin4567.val[1], qin4567.val[2], qin4567.val[3],
                        &qout0, &qout1, &qout2, &qout3, &qout4, &qout5);
                float* output_col0123 = output;
                vst1q_f32(output_col0123, qout0); output_col0123 += output_stride;
                vst1q_f32(output_col0123, qout1); output_col0123 += output_stride;
                vst1q_f32(output_col0123, qout2); output_col0123 += output_stride;
                vst1q_f32(output_col0123, qout3); output_col0123 += output_stride;
                vst1q_f32(output_col0123, qout4); output_col0123 += output_stride;
                vst1q_f32(output_col0123, qout5);

                float32x2x2_t din01 = vld2_f32(read_ptr); read_ptr += 4;
                float32x2x2_t din23 = vld2_f32(read_ptr); read_ptr += 4;
                float32x2x2_t din45 = vld2_f32(read_ptr); read_ptr += 4;
                float32x2x2_t din67 = vld2_f32(read_ptr);
                float32x2_t dout0, dout1, dout2, dout3, dout4, dout5;
                winograd_f6k3_output_transform(
                        din01.val[0], din01.val[1], din23.val[0], din23.val[1],
                        din45.val[0], din45.val[1], din67.val[0], din67.val[1],
                        &dout0, &dout1, &dout2, &dout3, &dout4, &dout5);
                float* output_col45 = output + 4;
                vst1_f32(output_col45, dout0); output_col45 += output_stride;
                vst1_f32(output_col45, dout1); output_col45 += output_stride;
                vst1_f32(output_col45, dout2); output_col45 += output_stride;
                vst1_f32(output_col45, dout3); output_col45 += output_stride;
                vst1_f32(output_col45, dout4); output_col45 += output_stride;
                vst1_f32(output_col45, dout5);
        } else {
	 NNP_SIMD_ALIGN float block[6][8];

                float32x4x4_t qin0123 = vld4q_f32(read_ptr); read_ptr += 16;
                float32x4x4_t qin4567 = vld4q_f32(read_ptr); read_ptr += 16;
                float32x4_t qout0, qout1, qout2, qout3, qout4, qout5;
                winograd_f6k3_output_transformq(
                        qin0123.val[0], qin0123.val[1], qin0123.val[2], qin0123.val[3],
                        qin4567.val[0], qin4567.val[1], qin4567.val[2], qin4567.val[3],
                        &qout0, &qout1, &qout2, &qout3, &qout4, &qout5);
                vst1q_f32(&block[0][0], qout0);
                vst1q_f32(&block[1][0], qout1);
                vst1q_f32(&block[2][0], qout2);
                vst1q_f32(&block[3][0], qout3);
                vst1q_f32(&block[4][0], qout4);
                vst1q_f32(&block[5][0], qout5);

                float32x2x2_t din01 = vld2_f32(read_ptr); read_ptr += 4;
                float32x2x2_t din23 = vld2_f32(read_ptr); read_ptr += 4;
                float32x2x2_t din45 = vld2_f32(read_ptr); read_ptr += 4;
                float32x2x2_t din67 = vld2_f32(read_ptr);
                float32x2_t dout0, dout1, dout2, dout3, dout4, dout5;
                winograd_f6k3_output_transform(
                        din01.val[0], din01.val[1], din23.val[0], din23.val[1],
                        din45.val[0], din45.val[1], din67.val[0], din67.val[1],
                        &dout0, &dout1, &dout2, &dout3, &dout4, &dout5);
                vst1_f32(&block[0][4], dout0);
                vst1_f32(&block[1][4], dout1);
                vst1_f32(&block[2][4], dout2);
                vst1_f32(&block[3][4], dout3);
                vst1_f32(&block[4][4], dout4);
                vst1_f32(&block[5][4], dout5);

                for (size_t i = 0; i < row_count; i++) {
                        for (size_t j = 0; j < column_count; j++) {
                                output[i * output_stride + j] = block[i][j];
                        }
                }
        }
}
/* original and working output transformations checked with nnpack test example layers*/
void nnp_owt8x8_3x3_with_bias__neon2(
	const void *restrict transform,
	float output[restrict static 1],
	const float bias[restrict static 1],
	size_t transform_stride,
	size_t output_stride,
	uint32_t row_count,
	uint32_t column_count)
{

	NNP_SIMD_ALIGN float buffer[8 * 6];
	float*restrict qbuffer = buffer;
	float*restrict dbuffer = buffer + 32;
	 


	 for(int i1=0;i1<4;i1+=svcntw())
                {
                svbool_t pg = svwhilelt_b32(i1,4);
                 float32x2_t vbias_neon = vreinterpret_f32_u64(vshl_n_u64(vreinterpret_u64_f32(vld1_dup_f32(bias)), 32));
                float block[4];
                vst1_f32(&block[0], vbias_neon);
                svfloat32_t vbias =  svld1(pg, &block[0]);
                float block1[4];
                svst1(pg, &block1[0], vbias);

                svfloat32_t m0, m1,m2, m3, m4, m5, m6, m7;

                //first iteration
                m0 = svld1_f32(pg, transform); transform += transform_stride;
                ////neon-test//

               // float32x4_t m1_neon = vld1q_f32(transform);transform += transform_stride;
               //  m1_neon = vcombine_f32(vadd_f32(vget_low_f32(m1_neon), vbias_neon), vget_high_f32(m1_neon));
               // float check_neon[4];
              //  vst1q_f32(&check_neon[0], m1_neon);

                //logic - for sve - m1 starts///        
                svbool_t pg_low1 = svdupq_b32(1,1,0,0);
                svbool_t pg_high1 = svdupq_b32(0,0,1,1);
		
	//	 m1 = svld1(pg, &check_neon[0]);         //temporary solution
                //vbias_neon = vmov_n_f32(0.0f);
                 m1 = svld1_f32(pg, transform); transform += transform_stride;
		 m2 = svld1_f32(pg, transform); transform += transform_stride;
                 m3 = svld1_f32(pg, transform); transform += transform_stride;
                 m4 = svld1_f32(pg, transform); transform += transform_stride;
                 m5 = svld1_f32(pg, transform); transform += transform_stride;
                 m6 = svld1_f32(pg, transform); transform += transform_stride;
                 m7 = svld1_f32(pg, transform); transform += transform_stride;
                float check[4];
                svfloat32_t o0, o1, o2, o3, o4, o5;
                winograd_f6k3_output_transformq1(
                        m0, m1, m2, m3, m4, m5, m6, m7,
                        &o0, &o1, &o2, &o3, &o4, &o5);

                  svst1_f32(pg, &qbuffer[0], o0); qbuffer += 4;
                  svst1_f32(pg, qbuffer, o1); qbuffer += 4;
               svst1_f32(pg, qbuffer, o2); qbuffer += 4;
                svst1_f32(pg, qbuffer, o3);qbuffer += 4;
                float o4_buff[4], o5_buff[4];
                //svst1(pg, o4_buff, o4);
                //svst1(pg, o5_buff, o5);
//               float check[4];

                svst1_f32(pg_low1, dbuffer, o4); dbuffer += 2;
               svst1_f32(pg_low1, dbuffer, o5);//dbuffer += 2;
               svst1_f32(pg_high1, dbuffer, o4); dbuffer += 2;
                svst1_f32(pg_high1, dbuffer, o5); dbuffer += 4;
		//second iteration
                 m0 = svld1_f32(pg, transform); transform += transform_stride;
                ////neon-test//

               //  m1_neon = vld1q_f32(transform); transform += transform_stride;
               //  m1_neon = vcombine_f32(vadd_f32(vget_low_f32(m1_neon), vbias_neon), vget_high_f32(m1_neon));
               // float check_neon1[4];
               // vst1q_f32(&check_neon1[0], m1_neon);
                //neon=test end// 

                //logic - for sve - m1 starts///        
                pg_low1 = svdupq_b32(1,1,0,0);
                pg_high1 = svdupq_b32(0,0,1,1);

                // m1 = svld1(pg, &check_neon1[0]);         //temporary solution
        	  m1 = svld1_f32(pg, transform); transform += transform_stride;  
	       m2 = svld1_f32(pg, transform); transform += transform_stride;
                 m3 = svld1_f32(pg, transform); transform += transform_stride;
                 m4 = svld1_f32(pg, transform); transform += transform_stride;
                 m5 = svld1_f32(pg, transform); transform += transform_stride;
                 m6 = svld1_f32(pg, transform); transform += transform_stride;
                 m7 = svld1_f32(pg, transform); transform += transform_stride;
               // float check1[4];
//                svfloat32_t o0, o1, o2, o3, o4, o5;
                winograd_f6k3_output_transformq1(
                        m0, m1, m2, m3, m4, m5, m6, m7,
                        &o0, &o1, &o2, &o3, &o4, &o5);

		 svst1_f32(pg, qbuffer, o0); qbuffer += 4;
                  svst1_f32(pg, qbuffer, o1); qbuffer += 4;
               svst1_f32(pg, qbuffer, o2); qbuffer += 4;
                svst1_f32(pg, qbuffer, o3);qbuffer += 4;
                //svst1(pg, o4_buff, o4);
                //svst1(pg, o5_buff, o5);
//               float check[4];

                svst1_f32(pg_low1, dbuffer, o4); dbuffer += 2;
               svst1_f32(pg_low1, dbuffer, o5);//dbuffer += 2;
               svst1_f32(pg_high1, dbuffer, o4); dbuffer += 2;
                svst1_f32(pg_high1, dbuffer, o5); dbuffer += 4;

		


	const float*restrict read_ptr = buffer;
	if NNP_LIKELY(row_count == 6 && column_count == 6 && output_stride >= 6) {
	svfloat32_t a0,a1,a2,a3, a4, a5, a6, a7;
                a0 = svld1_f32(pg,  &read_ptr[0]);
                a1 = svld1_f32(pg,  &read_ptr[4]);
                a2 = svld1_f32(pg,  &read_ptr[8]);
                a3 = svld1_f32(pg,  &read_ptr[12]);
                a4 = svld1_f32(pg,  &read_ptr[16]);
                a5 = svld1_f32(pg,  &read_ptr[20]);
                a6 = svld1_f32(pg,  &read_ptr[24]);
                a7 = svld1_f32(pg,  &read_ptr[28]);
                svfloat32x4_t inp0123 =  svcreate4(a0,a1,a2,a3);
                svfloat32x4_t inp4567 =  svcreate4(a4,a5,a6,a7);
                  float  qin0123[16], qin4567[16];
                svst4(pg, &qin0123[0], inp0123);
                svst4(pg, &qin4567[0], inp4567);
                //float32x4x4_t qin0123 = vld4q_f32(read_ptr); read_ptr += 16;
                //float32x4x4_t qin4567 = vld4q_f32(read_ptr); read_ptr += 16;
                svfloat32_t qout0, qout1, qout2, qout3, qout4, qout5;
                winograd_f6k3_output_transformq1(
                        svld1_f32( pg, &qin0123[0]),svld1_f32( pg, &qin0123[4]), svld1_f32( pg, &qin0123[8]), svld1_f32( pg, &qin0123[12]), svld1_f32( pg, &qin4567[0]),svld1_f32( pg, &qin4567[4]), svld1_f32( pg, &qin4567[8]), svld1_f32( pg, &qin4567[12]),  &qout0, &qout1, &qout2, &qout3, &qout4, &qout5);
                float* output_col0123 = output;
                svst1_f32(pg, output_col0123, qout0); output_col0123 += output_stride;
                svst1_f32(pg, output_col0123, qout1); output_col0123 += output_stride;
                svst1_f32(pg, output_col0123, qout2); output_col0123 += output_stride;
                svst1_f32(pg, output_col0123, qout3); output_col0123 += output_stride;
                svst1_f32(pg, output_col0123, qout4); output_col0123 += output_stride;
                svst1_f32(pg, output_col0123, qout5);
		a0 = svld1(pg,  &read_ptr[32]);
                a1 = svld1(pg,  &read_ptr[34]);
                a2 = svld1(pg,  &read_ptr[36]);
                a3 = svld1(pg,  &read_ptr[38]);
                a4 = svld1(pg,  &read_ptr[40]);
                a5 = svld1(pg,  &read_ptr[42]);
                a6 = svld1(pg,  &read_ptr[44]);
                a7 = svld1(pg,  &read_ptr[46]);

                svfloat32x2_t din01 =  svcreate2(a0,a1);
                svfloat32x2_t din23 =  svcreate2(a2, a3);
                svfloat32x2_t din45 =  svcreate2(a4, a5);
                svfloat32x2_t din67 =  svcreate2(a6, a7);
                 float  din101[16], din123[16], din145[16], din167[16];
                svst2(pg, &din101[0], din01);
                svst2(pg, &din123[0], din23);
                svst2(pg, &din145[0], din45);
                svst2(pg, &din167[0], din67);
		 svfloat32_t dout0, dout1, dout2, dout3, dout4, dout5;
                winograd_f6k3_output_transform1(
                        svld1_f32( pg, &din101[0]), svld1_f32( pg, &din101[2]), svld1_f32( pg, &din123[0]), svld1_f32( pg, &din123[2]), svld1_f32( pg, &din145[0]), svld1_f32( pg, &din145[2]), svld1_f32( pg, &din167[0]), svld1_f32( pg, &din167[2]), &dout0, &dout1, &dout2, &dout3, &dout4, &dout5);
                float* output_col45 = output + 4;
                
		 svbool_t pg_new = svwhilelt_b32(i1,2);
                svst1_f32(pg_new, output_col45, dout0); output_col45 += output_stride;
                svst1_f32(pg_new, output_col45, dout1); output_col45 += output_stride;
                svst1_f32(pg_new, output_col45, dout2); output_col45 += output_stride;
                svst1_f32(pg_new, output_col45, dout3); output_col45 += output_stride;
                svst1_f32(pg_new, output_col45, dout4); output_col45 += output_stride;
                svst1_f32(pg_new, output_col45, dout5);

	} else {
		NNP_SIMD_ALIGN float block[6][8];
		 svfloat32_t a0,a1,a2,a3, a4, a5, a6, a7;
                a0=svld1(pg,  &read_ptr[0]);
                a1=svld1(pg,  &read_ptr[4]);
                a2=svld1(pg,  &read_ptr[8]);
                a3=svld1(pg,  &read_ptr[12]);
                a4=svld1(pg,  &read_ptr[16]);
                a5=svld1(pg,  &read_ptr[20]);
                a6=svld1(pg,  &read_ptr[24]);
                a7=svld1(pg,  &read_ptr[28]);
                svfloat32x4_t inp0123 =  svcreate4(a0,a1,a2,a3);
                svfloat32x4_t inp4567 =  svcreate4(a4,a5,a6,a7);
                  float  qin0123[16], qin4567[16];
                svst4(pg, &qin0123[0], inp0123);
                svst4(pg, &qin4567[0], inp4567);
		svfloat32_t qout0, qout1, qout2, qout3, qout4, qout5;
		winograd_f6k3_output_transformq1(
                        svld1_f32( pg, &qin0123[0]),svld1_f32( pg, &qin0123[4]), svld1_f32( pg, &qin0123[8]), svld1_f32( pg, &qin0123[12]), svld1_f32( pg, &qin4567[0]),svld1_f32( pg, &qin4567[4]), svld1_f32( pg, &qin4567[8]), svld1_f32( pg, &qin4567[12]),  &qout0, &qout1, &qout2, &qout3, &qout4, &qout5);

		svst1_f32(pg, &block[0][0], qout0);
		svst1_f32(pg, &block[1][0], qout1);
		svst1_f32(pg, &block[2][0], qout2);
		svst1_f32(pg, &block[3][0], qout3);
		svst1_f32(pg, &block[4][0], qout4);
		svst1_f32(pg, &block[5][0], qout5);

		a0 =  svld1(pg,  &read_ptr[32]);
                a1 = svld1(pg,  &read_ptr[34]);
                a2 = svld1(pg,  &read_ptr[36]);
                a3 = svld1(pg,  &read_ptr[38]);
                a4 = svld1(pg,  &read_ptr[40]);
                a5 = svld1(pg,  &read_ptr[42]);
                a6 = svld1(pg,  &read_ptr[44]);
                a7 = svld1(pg,  &read_ptr[46]);
                svfloat32x2_t din01 =  svcreate2(a0,a1);
                svfloat32x2_t din23 =  svcreate2(a2, a3);
                svfloat32x2_t din45 =  svcreate2(a4, a5);
                svfloat32x2_t din67 =  svcreate2(a6, a7);
                 float  din101[16], din123[16], din145[16], din167[16];
                svst2(pg, &din101[0], din01);
                svst2(pg, &din123[0], din23);
                svst2(pg, &din145[0], din45);
                svst2(pg, &din167[0], din67);


		svfloat32_t dout0, dout1, dout2, dout3, dout4, dout5;

		winograd_f6k3_output_transform1(
                        svld1_f32( pg, &din101[0]), svld1_f32( pg, &din101[2]), svld1_f32( pg, &din123[0]), svld1_f32( pg, &din123[2]), svld1_f32( pg, &din145[0]), svld1_f32( pg, &din145[2]), svld1_f32( pg, &din167[0]), svld1_f32( pg, &din167[2]), &dout0, &dout1, &dout2, &dout3, &dout4, &dout5);

		 svbool_t pg_new = svwhilelt_b32(i1,2);

		svst1_f32(pg_new, &block[0][4], dout0);
		svst1_f32(pg_new, &block[1][4], dout1);
		svst1_f32(pg_new, &block[2][4], dout2);
		svst1_f32(pg_new, &block[3][4], dout3);
		svst1_f32(pg_new, &block[4][4], dout4);
		svst1_f32(pg_new, &block[5][4], dout5);

		for (size_t i = 0; i < row_count; i++) {
			for (size_t j = 0; j < column_count; j++) {
				output[i * output_stride + j] = block[i][j];
			}
		}
	}
/////
	}
}


/* output transformations trying to get darknet working - sonia disable bias */
void nnp_owt8x8_3x3_with_bias__neon_intertile(
	const void **restrict transform,
	float *output[restrict static 1],
	const float bias[restrict static 1],
	size_t transform_stride,
	size_t output_stride,
	uint32_t row_count,
	uint32_t column_count)
{
 int simd_width = nnp_hwinfo.sve_simd_width;//nnp_hwinfo.simd_width;
        int interchannels  = nnp_hwinfo.globalinterchannels;

float *new_data;
        new_data = (float *) malloc(sizeof(float) * 8*8*8*interchannels);

   for(int i=0;i<16;i++)
        {       
	   #pragma loop unroll_count(interchannels)
           for(int k=0;k<interchannels;k++){
                for(int j=0;j<4;j++)
                {       
                        new_data[(i*simd_width)+(j+k*4)] =  *(((float *)transform[k])+j);
                }
                transform[k] += transform_stride;
           }
        }

         for(int i1=0;i1<simd_width;i1+=svcntw())
         {
                svbool_t pg = svwhilelt_b32(i1,simd_width);
                svfloat32_t m0, m1,m2, m3, m4, m5, m6, m7;

                //first iteration
                m0 = svld1_f32(pg, &new_data[0*simd_width]);
                m1 = svld1_f32(pg, &new_data[1*simd_width]);
                m2 = svld1_f32(pg, &new_data[2*simd_width]);
                m3 = svld1_f32(pg, &new_data[3*simd_width]);
                m4 = svld1_f32(pg, &new_data[4*simd_width]);
                m5 = svld1_f32(pg, &new_data[5*simd_width]);
                m6 = svld1_f32(pg, &new_data[6*simd_width]);
                m7 = svld1_f32(pg, &new_data[7*simd_width]);
                svfloat32_t o0, o1, o2, o3, o4, o5;
                winograd_f6k3_output_transformq1_intertile(
                        m0, m1, m2, m3, m4, m5, m6, m7,
                        &o0, &o1, &o2, &o3, &o4, &o5);



        //second iteration
                svfloat32_t o6, o7, o8, o9, o10, o11;
                m0 = svld1_f32(pg, &new_data[8*simd_width]);
                m1 = svld1_f32(pg, &new_data[9*simd_width]);
                m2 = svld1_f32(pg, &new_data[10*simd_width]);
                m3 = svld1_f32(pg, &new_data[11*simd_width]);
                m4 = svld1_f32(pg, &new_data[12*simd_width]);
                m5 = svld1_f32(pg, &new_data[13*simd_width]);
                m6 = svld1_f32(pg, &new_data[14*simd_width]);
                m7 = svld1_f32(pg, &new_data[15*simd_width]);
                winograd_f6k3_output_transformq1_intertile(
                        m0, m1, m2, m3, m4, m5, m6, m7,
                        &o6, &o7, &o8, &o9, &o10, &o11);

                if NNP_LIKELY(row_count == 6 && column_count == 6 && output_stride >= 6) {
                svfloat32_t a0,a1,a2,a3, a4, a5, a6, a7;
                svfloat32x4_t inp0123 =  svcreate4(o0,o1,o2,o3);
                svfloat32x4_t inp4567 =  svcreate4(o6,o7,o8,o9);
                  float  qin0123[4*simd_width], qin4567[4*simd_width];
                svst4(pg, &qin0123[0], inp0123);
                svst4(pg, &qin4567[0], inp4567);

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

                 svfloat32_t qout0, qout1, qout2, qout3, qout4, qout5;
                winograd_f6k3_output_transformq1_intertile(
                        svld1_gather_s32index_f32( pg, &qin0123[0], index1),svld1_gather_s32index_f32( pg, &qin0123[0], index2), svld1_gather_s32index_f32( pg, &qin0123[0], index3), svld1_gather_s32index_f32( pg, &qin0123[0], index4), svld1_gather_s32index_f32( pg, &qin4567[0], index1),svld1_gather_s32index_f32( pg, &qin4567[0], index2), svld1_gather_s32index_f32( pg, &qin4567[0], index3), svld1_gather_s32index_f32( pg, &qin4567[0], index4),  &qout0, &qout1, &qout2, &qout3, &qout4, &qout5);
                float output_ptr[16*simd_width];
                svst1_f32(pg, &output_ptr[0], qout0);
                svst1_f32(pg, &output_ptr[simd_width], qout1);
                svst1_f32(pg, &output_ptr[2*simd_width], qout2);
                svst1_f32(pg, &output_ptr[3*simd_width], qout3);
                svst1_f32(pg, &output_ptr[4*simd_width], qout4);
                svst1_f32(pg, &output_ptr[5*simd_width], qout5);

                svbool_t pg_new = svwhilelt_b32(i1,simd_width/2);

                svfloat32x2_t din01 =  svcreate2(o4,o5);
                svfloat32x2_t din23 =  svcreate2(o10, o11);

                float  din101[4*simd_width], din123[4*simd_width], din145[4*simd_width], din167[4*simd_width];
                svst2(pg, &din101[0], din01);
                svst2(pg, &din123[0], din23);
		
                svint32_t index01, index02, index03, index04;
                
		 int index45_host1[simd_width];
                for(int i=0;i<interchannels;i++)
                {       
                        for(int j=0;j<2;j++)
                        {       
                                index45_host1[j+i*2] = j+(8*i);
                        }
                }

                index01 = svld1(pg_new, &index45_host1[0]);
                 index02 = svadd_s32_m(pg, index01, svdup_s32(2));
                index03 = svadd_s32_m(pg, index02, svdup_s32(2));
                index04 = svadd_s32_m(pg, index03, svdup_s32(2));
                 
		svfloat32_t dout0, dout1, dout2, dout3, dout4, dout5;
                winograd_f6k3_output_transform1_intertile(
                        svld1_gather_s32index_f32( pg_new, &din101[0], index01), svld1_gather_s32index_f32( pg_new, &din101[0], index02), svld1_gather_s32index_f32( pg_new, &din101[0], index03), svld1_gather_s32index_f32( pg_new, &din101[0], index04), svld1_gather_s32index_f32( pg_new, &din123[0], index01), svld1_gather_s32index_f32( pg_new, &din123[0], index02), svld1_gather_s32index_f32( pg_new, &din123[0], index03), svld1_gather_s32index_f32( pg_new, &din123[0], index04), &dout0, &dout1, &dout2, &dout3, &dout4, &dout5);
                float output_col45ptr[16*simd_width];
                svst1_f32(pg_new, &output_col45ptr[0], dout0);
                svst1_f32(pg_new, &output_col45ptr[simd_width/2], dout1);
                svst1_f32(pg_new, &output_col45ptr[2*(simd_width/2)], dout2);
                svst1_f32(pg_new, &output_col45ptr[3*(simd_width/2)], dout3);
                svst1_f32(pg_new, &output_col45ptr[4*(simd_width/2)], dout4);
                svst1_f32(pg_new, &output_col45ptr[5*(simd_width/2)], dout5);
                
	float* output_col0123[interchannels];
	float* output_col45[interchannels];
	for(int k=0;k<interchannels;k++)
	{
		output_col0123[k] = output[k];
		 output_col45[k] = output[k] + 4;
	}

             for (size_t i = 0; i < 6; i++) {
        	#pragma loop unroll_count(interchannels)
                for(int k=0;k<interchannels;k++){     
	           for (size_t j = 0; j < 4; j++) {
                                output_col0123[k][j]   = output_ptr[i*simd_width+(j+k*4)];
			}
                        if(i<5){
                        output_col0123[k] += output_stride;
			}
		}}
	           	
             for (size_t i = 0; i < 6; i++) {
        		#pragma loop unroll_count(interchannels)
                	for(int k=0;k<interchannels;k++){     
				for (size_t j = 0; j < 2; j++) {
				output_col45[k][j]    = output_col45ptr[i*(simd_width/2)+(j+k*2)];
			}
                        if(i<5){
			output_col45[k] += output_stride;
                        }
		     }
                }


		/* float* output_col0123 = output[0];
                float* output1_col0123 = output[1];
                float* output2_col0123 = output[2];
                float* output3_col0123 = output[3];
		for (size_t i = 0; i < 6; i++) {
                        for (size_t j = 0; j < 4; j++) {
                                output_col0123[j]   = output_ptr[i*simd_width+j];
                                output1_col0123[j]   = output_ptr[i*simd_width + (j+4)];
                                output2_col0123[j]   = output_ptr[i*simd_width + (j+8)];
                                output3_col0123[j]  = output_ptr[i*simd_width + (j+12)];
                        }
                        if(i<5){
                        output_col0123 += output_stride;
                        output1_col0123 += output_stride;
                        output2_col0123 += output_stride;
                        output3_col0123 += output_stride;
                        }
                        }
                float* output_col45 = output[0] + 4;
                float* output1_col45 = output[1] + 4;
                float* output2_col45 = output[2] + 4;
                float* output3_col45 = output[3] + 4;
                for (size_t i = 0; i < 6; i++) {
                        for (size_t j = 0; j < 2; j++) {
                                output_col45[j]    = output_col45ptr[i*(simd_width/2)+j];
                                output1_col45[j]    = output_col45ptr[i*(simd_width/2) + (j+2)];
                                output2_col45[j]   = output_col45ptr[i*(simd_width/2) + (j+4)];
                                output3_col45[j]   = output_col45ptr[i*(simd_width/2) + (j+6)];
                        }
                        if(i<5){
                        output_col45 += output_stride;
                        output1_col45 += output_stride;
                        output2_col45  += output_stride;
                        output3_col45 += output_stride;
                        }
                        }*/


        }
        else {
                float block[6*simd_width];
                float block1[6*simd_width];
                svfloat32x4_t inp0123 =  svcreate4(o0,o1,o2,o3);
                svfloat32x4_t inp4567 =  svcreate4(o6,o7,o8,o9);
                  float  qin0123[4*simd_width], qin4567[4*simd_width];
                svst4(pg, &qin0123[0], inp0123);
                svst4(pg, &qin4567[0], inp4567);
                svfloat32_t qout0, qout1, qout2, qout3, qout4, qout5;
                
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

                winograd_f6k3_output_transformq1_intertile(
                        svld1_gather_s32index_f32( pg, &qin0123[0], index1),svld1_gather_s32index_f32( pg, &qin0123[0], index2), svld1_gather_s32index_f32( pg, &qin0123[0], index3), svld1_gather_s32index_f32( pg, &qin0123[0], index4), svld1_gather_s32index_f32( pg, &qin4567[0], index1),svld1_gather_s32index_f32( pg, &qin4567[0], index2), svld1_gather_s32index_f32( pg, &qin4567[0], index3), svld1_gather_s32index_f32( pg, &qin4567[0], index4),  &qout0, &qout1, &qout2, &qout3, &qout4, &qout5);

		svst1_f32(pg, &block[0], qout0);
                svst1_f32(pg, &block[simd_width], qout1);
                svst1_f32(pg, &block[2*simd_width], qout2);
                svst1_f32(pg, &block[3*simd_width], qout3);
                svst1_f32(pg, &block[4*simd_width], qout4);
                svst1_f32(pg, &block[5*simd_width], qout5);
                 svbool_t pg_new = svwhilelt_b32(i1,simd_width/2);
                svfloat32x2_t din01 =  svcreate2(o4,o5);
                svfloat32x2_t din23 =  svcreate2(o10, o11);

                float  din101[4*simd_width], din123[4*simd_width], din145[4*simd_width], din167[4*simd_width];
                svst2(pg, &din101[0], din01);
                svst2(pg, &din123[0], din23);

		svint32_t index01, index02, index03, index04;
		int index45_host1[simd_width];

                for(int i=0;i<interchannels;i++)
                {
                        for(int j=0;j<2;j++)
                        {
                                index45_host1[j+i*2] = j+(8*i);
                        }
                }

                index01 = svld1(pg_new, &index45_host1[0]);
                 index02 = svadd_s32_m(pg, index01, svdup_s32(2));
                index03 = svadd_s32_m(pg, index02, svdup_s32(2));
                index04 = svadd_s32_m(pg, index03, svdup_s32(2));



                 svfloat32_t dout0, dout1, dout2, dout3, dout4, dout5;
                winograd_f6k3_output_transform1_intertile(
                        svld1_gather_s32index_f32( pg_new, &din101[0], index01), svld1_gather_s32index_f32( pg_new, &din101[0], index02), svld1_gather_s32index_f32( pg_new, &din101[0], index03), svld1_gather_s32index_f32( pg_new, &din101[0], index04), svld1_gather_s32index_f32( pg_new, &din123[0], index01), svld1_gather_s32index_f32( pg_new, &din123[0], index02), svld1_gather_s32index_f32( pg_new, &din123[0], index03), svld1_gather_s32index_f32( pg_new, &din123[0], index04), &dout0, &dout1, &dout2, &dout3, &dout4, &dout5);

                svst1_f32(pg_new, &block1[0], dout0);
                svst1_f32(pg_new, &block1[simd_width/2], dout1);
                svst1_f32(pg_new, &block1[2*(simd_width/2)], dout2);
                svst1_f32(pg_new, &block1[3*(simd_width/2)], dout3);
                svst1_f32(pg_new, &block1[4*(simd_width/2)], dout4);
                svst1_f32(pg_new, &block1[5*(simd_width/2)], dout5);
                
		//chpice of ths is we need to keep it
		/*for (size_t i = 0; i < row_count; i++) {
                        for (size_t j = 0; j < column_count; j++) {
                                if(j <4){
                                output[0][i * output_stride + j] = block[i*simd_width+ j];
                                output[1][i * output_stride + j] = block[ i *simd_width + (j +4)];
                                if(interchannels > 2) {output[2][i * output_stride + j] = block[ i *simd_width +( j +8)];}
                                if(interchannels > 3) {output[3][i * output_stride + j] =  block[ i *simd_width +(j +12)] ;}
                                if(interchannels > 4) {output[4][i * output_stride + j] = block[ i *simd_width +( j +16)];}
                                if(interchannels > 5) {output[5][i * output_stride + j] =  block[ i *simd_width +(j +20)] ;}
                                if(interchannels > 6) {output[6][i * output_stride + j] = block[ i *simd_width +( j +24)];}
                                if(interchannels > 7) {output[7][i * output_stride + j] =  block[ i *simd_width +(j +28)] ;}
                                }
                                else if(j >3 && j<6)
                                {
                                output[0][i * output_stride + j] = block1[i*(simd_width/2)+ (j-4)];
                                output[1][i * output_stride + j] = block1[ i *(simd_width/2) +(j-2 )];
                                if(interchannels > 2) {output[2][i * output_stride + j] = block1[ i*(simd_width/2)+(j )];}
                                if(interchannels > 3) {output[3][i * output_stride + j] =  block1[ i*(simd_width/2)+(j +2)] ;}
                                if(interchannels > 4) {output[4][i * output_stride + j] = block1[ i*(simd_width/2)+(j +4)];}
                                if(interchannels > 5) {output[5][i * output_stride + j] =  block1[ i*(simd_width/2)+(j +6)] ;}
                                if(interchannels > 6) {output[6][i * output_stride + j] = block1[ i*(simd_width/2)+(j+8 )];}
                                if(interchannels > 7) {output[7][i * output_stride + j] =  block1[ i*(simd_width/2)+(j +10)] ;}
                                }
                                }
                        }*/

		//original need to uncomment

		for (size_t i = 0; i < row_count; i++) {
		    #pragma loop unroll_count(interchannels)
            	    for(int k=0;k<interchannels;k++){

                        for (size_t j = 0; j < column_count; j++)   //original trying to change it 
			{
                        	if(j<4) {  //original trying to change it 
                                output[k][i * output_stride + j] = block[i*simd_width+ (j+k*4)];
			}
		    }
		}
		   #pragma loop unroll_count(interchannels)
                    for(int k=0;k<interchannels;k++)
		    {
			 for (size_t j = 0; j < column_count; j++)
                       	 {
				if(j>3&&j<6){
                                output[k][i * output_stride + j] = block1[(i*(simd_width/2))+ ((j+k*2)-4)];
                                }
			}
			}
		   }
		
        }
    }
free(new_data);
	
}
/* output transformations trying to get darknet working - sonia disable bias */
void nnp_owt8x8_3x3_with_bias__neon(
	const void *restrict transform,
	float output[restrict static 1],
	const float bias[restrict static 1],
	size_t transform_stride,
	size_t output_stride,
	uint32_t row_count,
	uint32_t column_count)
{


int simd_width=nnp_hwinfo.simd_width;
	NNP_SIMD_ALIGN float buffer[2*simd_width * 6];
	float*restrict qbuffer = buffer;
	//float*restrict dbuffer = buffer + 32;
	float*restrict dbuffer = buffer + (8*simd_width);
	 


	 for(int i1=0;i1<simd_width;i1+=svcntw())
                {
                svbool_t pg = svwhilelt_b32(i1,simd_width);
		float32x4_t m1_neon;
		float32x2_t vbias_neon;
		svfloat32_t m0, m1,m2, m3, m4, m5, m6, m7;

                //first iteration
                m0 = svld1_f32(pg, transform); transform += transform_stride;
                m1 = svld1_f32(pg, transform); transform += transform_stride;
      

                 m2 = svld1_f32(pg, transform); transform += transform_stride;
                 m3 = svld1_f32(pg, transform); transform += transform_stride;
                 m4 = svld1_f32(pg, transform); transform += transform_stride;
                 m5 = svld1_f32(pg, transform); transform += transform_stride;
                 m6 = svld1_f32(pg, transform); transform += transform_stride;
                 m7 = svld1_f32(pg, transform); transform += transform_stride;
                svfloat32_t o0, o1, o2, o3, o4, o5;
                svfloat32_t o6, o7, o8, o9, o10, o11;
                winograd_f6k3_output_transformq1(
                        m0, m1, m2, m3, m4, m5, m6, m7,
                        &o0, &o1, &o2, &o3, &o4, &o5);

		//second iteration
                 m0 = svld1_f32(pg, transform); transform += transform_stride;
                 m1 = svld1_f32(pg, transform); transform += transform_stride;

                 m2 = svld1_f32(pg, transform); transform += transform_stride;
                 m3 = svld1_f32(pg, transform); transform += transform_stride;
                 m4 = svld1_f32(pg, transform); transform += transform_stride;
                 m5 = svld1_f32(pg, transform); transform += transform_stride;
                 m6 = svld1_f32(pg, transform); transform += transform_stride;
                 m7 = svld1_f32(pg, transform); transform += transform_stride;
                //float check1[4];
//                svfloat32_t o0, o1, o2, o3, o4, o5;
                winograd_f6k3_output_transformq1(
                        m0, m1, m2, m3, m4, m5, m6, m7,
                        &o6, &o7, &o8, &o9, &o10, &o11);


	const float*restrict read_ptr = buffer;
	if NNP_LIKELY(row_count == 6 && column_count == 6 && output_stride >= 6) {
                svfloat32x4_t inp0123 =  svcreate4(o0,o1,o2,o3);
                svfloat32x4_t inp4567 =  svcreate4(o6,o7,o8, o9);
                  float  qin0123[4*simd_width], qin4567[4*simd_width];
                svst4(pg, &qin0123[0], inp0123);
                svst4(pg, &qin4567[0], inp4567);
                svfloat32_t qout0, qout1, qout2, qout3, qout4, qout5;
                winograd_f6k3_output_transformq1(
                        svld1_f32( pg, &qin0123[0]),svld1_f32( pg, &qin0123[simd_width]), svld1_f32( pg, &qin0123[2*simd_width]), svld1_f32( pg, &qin0123[3*simd_width]), svld1_f32( pg, &qin4567[0]),svld1_f32( pg, &qin4567[simd_width]), svld1_f32( pg, &qin4567[2*simd_width]), svld1_f32( pg, &qin4567[3*simd_width]),  &qout0, &qout1, &qout2, &qout3, &qout4, &qout5);
                float* output_col0123 = output;
                svst1_f32(pg, output_col0123, qout0); output_col0123 += output_stride;
                svst1_f32(pg, output_col0123, qout1); output_col0123 += output_stride;
                svst1_f32(pg, output_col0123, qout2); output_col0123 += output_stride;
                svst1_f32(pg, output_col0123, qout3); output_col0123 += output_stride;
                svst1_f32(pg, output_col0123, qout4); output_col0123 += output_stride;
                svst1_f32(pg, output_col0123, qout5);

                svfloat32x2_t din01 =  svcreate2(o4,o5);
                svfloat32x2_t din23 =  svcreate2(o10, o11);
                 float  din101[4*simd_width], din123[4*simd_width], din145[4*simd_width], din167[4*simd_width];
                svst2(pg, &din101[0], din01);
                svst2(pg, &din123[0], din23);
		 svfloat32_t dout0, dout1, dout2, dout3, dout4, dout5;
                winograd_f6k3_output_transform1(
                        svld1_f32( pg, &din101[0]), svld1_f32( pg, &din101[2]), svld1_f32( pg, &din101[4]), svld1_f32( pg, &din101[6]), svld1_f32( pg, &din123[0]), svld1_f32( pg, &din123[2]), svld1_f32( pg, &din123[4]), svld1_f32( pg, &din123[6]), &dout0, &dout1, &dout2, &dout3, &dout4, &dout5);
                float* output_col45 = output + simd_width;
                
		 svbool_t pg_new = svwhilelt_b32(i1,simd_width/2);
                svst1_f32(pg_new, output_col45, dout0); output_col45 += output_stride;
                svst1_f32(pg_new, output_col45, dout1); output_col45 += output_stride;
                svst1_f32(pg_new, output_col45, dout2); output_col45 += output_stride;
                svst1_f32(pg_new, output_col45, dout3); output_col45 += output_stride;
                svst1_f32(pg_new, output_col45, dout4); output_col45 += output_stride;
                svst1_f32(pg_new, output_col45, dout5);

	} else {
		NNP_SIMD_ALIGN float block[6][2*simd_width];
                
                svfloat32x4_t inp0123 =  svcreate4(o0,o1,o2,o3);
                svfloat32x4_t inp4567 =  svcreate4(o6,o7,o8, o9);
                  float  qin0123[4*simd_width], qin4567[4*simd_width];
                svst4(pg, &qin0123[0], inp0123);
                svst4(pg, &qin4567[0], inp4567);
		svfloat32_t qout0, qout1, qout2, qout3, qout4, qout5;
		winograd_f6k3_output_transformq1(
                        svld1_f32( pg, &qin0123[0]),svld1_f32( pg, &qin0123[simd_width]), svld1_f32( pg, &qin0123[2*simd_width]), svld1_f32( pg, &qin0123[3*simd_width]), svld1_f32( pg, &qin4567[0]),svld1_f32( pg, &qin4567[simd_width]), svld1_f32( pg, &qin4567[2*simd_width]), svld1_f32( pg, &qin4567[3*simd_width]),  &qout0, &qout1, &qout2, &qout3, &qout4, &qout5);

		svst1_f32(pg, &block[0][0], qout0);
		svst1_f32(pg, &block[1][0], qout1);
		svst1_f32(pg, &block[2][0], qout2);
		svst1_f32(pg, &block[3][0], qout3);
		svst1_f32(pg, &block[4][0], qout4);
		svst1_f32(pg, &block[5][0], qout5);

                svfloat32x2_t din01 =  svcreate2(o4,o5);
                svfloat32x2_t din23 =  svcreate2(o10, o11);
                 float  din101[4*simd_width], din123[4*simd_width], din145[4*simd_width], din167[4*simd_width];
                svst2(pg, &din101[0], din01);
                svst2(pg, &din123[0], din23);
		 svfloat32_t dout0, dout1, dout2, dout3, dout4, dout5;
                winograd_f6k3_output_transform1(
                        svld1_f32( pg, &din101[0]), svld1_f32( pg, &din101[2]), svld1_f32( pg, &din101[4]), svld1_f32( pg, &din101[6]), svld1_f32( pg, &din123[0]), svld1_f32( pg, &din123[2]), svld1_f32( pg, &din123[4]), svld1_f32( pg, &din123[6]), &dout0, &dout1, &dout2, &dout3, &dout4, &dout5);

		 svbool_t pg_new = svwhilelt_b32(i1,simd_width/2);

		svst1_f32(pg_new, &block[0][simd_width], dout0);
		svst1_f32(pg_new, &block[1][simd_width], dout1);
		svst1_f32(pg_new, &block[2][simd_width], dout2);
		svst1_f32(pg_new, &block[3][simd_width], dout3);
		svst1_f32(pg_new, &block[4][simd_width], dout4);
		svst1_f32(pg_new, &block[5][simd_width], dout5);

		for (size_t i = 0; i < row_count; i++) {
			for (size_t j = 0; j < column_count; j++) {
				output[i * output_stride + j] = block[i][j];
			}
		}
	}
/////
	}
}

void nnp_owt8x8_3x3s2_with_bias__neon_intertile(
	const void **restrict transform,
	float *output[restrict static 1],
	const float bias[restrict static 1],
	size_t transform_stride,
	size_t output_stride,
	uint32_t row_count,
	uint32_t column_count)
{
 int simd_width =nnp_hwinfo.sve_simd_width;//nnp_hwinfo.simd_width;
        const int interchannels  = nnp_hwinfo.globalinterchannels;


float *new_data;
        new_data = (float *) malloc(sizeof(float) * 8*8*8*interchannels);

   for(int i=0;i<16;i++)
   {
	 #pragma loop unroll_count(interchannels)
           for(int k=0;k<interchannels;k++){	
                for(int j=0;j<4;j++)
                {
                        new_data[(i*simd_width)+(j+k*4)] =  *(((float *)transform[k])+j);
                }
                transform[k] += transform_stride;
        }
   }


	 for(int i1=0;i1<simd_width;i1+=svcntw())
         {
                svbool_t pg = svwhilelt_b32(i1,simd_width);
                svfloat32_t m0, m1,m2, m3, m4, m5, m6, m7;

                //first iteration
                m0 = svld1_f32(pg, &new_data[0*simd_width]);
                m1 = svld1_f32(pg, &new_data[1*simd_width]);
                m2 = svld1_f32(pg, &new_data[2*simd_width]);
                m3 = svld1_f32(pg, &new_data[3*simd_width]);
                m4 = svld1_f32(pg, &new_data[4*simd_width]);
                m5 = svld1_f32(pg, &new_data[5*simd_width]);
                m6 = svld1_f32(pg, &new_data[6*simd_width]);
                m7 = svld1_f32(pg, &new_data[7*simd_width]);
                svfloat32_t o0, o1, o2, o3, o4, o5;
                winograd_f6k3_output_transformq1_intertile(
                        m0, m1, m2, m3, m4, m5, m6, m7,
                        &o0, &o1, &o2, &o3, &o4, &o5);

        //second iteration
                svfloat32_t o6, o7, o8, o9, o10, o11;
                m0 = svld1_f32(pg, &new_data[8*simd_width]);
                m1 = svld1_f32(pg, &new_data[9*simd_width]);
                m2 = svld1_f32(pg, &new_data[10*simd_width]);
                m3 = svld1_f32(pg, &new_data[11*simd_width]);
                m4 = svld1_f32(pg, &new_data[12*simd_width]);
                m5 = svld1_f32(pg, &new_data[13*simd_width]);
                m6 = svld1_f32(pg, &new_data[14*simd_width]);
                m7 = svld1_f32(pg, &new_data[15*simd_width]);
                winograd_f6k3_output_transformq1_intertile(
                        m0, m1, m2, m3, m4, m5, m6, m7,
                        &o6, &o7, &o8, &o9, &o10, &o11);

                float block[3*simd_width];
                float block1[3*simd_width];
                svfloat32x4_t inp0123 =  svcreate4(o0,o1,o2,o3);
                svfloat32x4_t inp4567 =  svcreate4(o6,o7,o8,o9);
                  float  qin0123[4*simd_width], qin4567[4*simd_width];
                svst4(pg, &qin0123[0], inp0123);
                svst4(pg, &qin4567[0], inp4567);
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



                svfloat32_t qout0, qout1, qout2, qout3, qout4, qout5;

                winograd_f6k3_output_transformq1_intertile(
                        svld1_gather_s32index_f32( pg, &qin0123[0], index1),svld1_gather_s32index_f32( pg, &qin0123[0], index2), svld1_gather_s32index_f32( pg, &qin0123[0], index3), svld1_gather_s32index_f32( pg, &qin0123[0], index4), svld1_gather_s32index_f32( pg, &qin4567[0], index1),svld1_gather_s32index_f32( pg, &qin4567[0], index2), svld1_gather_s32index_f32( pg, &qin4567[0], index3), svld1_gather_s32index_f32( pg, &qin4567[0], index4),  &qout0, &qout1, &qout2, &qout3, &qout4, &qout5);



                svst1_f32(pg, &block[0], qout0);
                svst1_f32(pg, &block[simd_width], qout2);
                svst1_f32(pg, &block[2*simd_width], qout4);

                 svbool_t pg_new = svwhilelt_b32(i1,simd_width/2);

                svfloat32x2_t din01 =  svcreate2(o4,o5);
                svfloat32x2_t din23 =  svcreate2(o10, o11);

                float  din101[4*simd_width], din123[4*simd_width], din145[4*simd_width], din167[4*simd_width];
                svst2(pg, &din101[0], din01);
                svst2(pg, &din123[0], din23);

		int index45_host1[simd_width];
                for(int i=0;i<interchannels;i++)
                {
                        for(int j=0;j<2;j++)
                        {
                                index45_host1[j+i*2] = j+(8*i);
                        }
                }
		svint32_t index01, index02, index03, index04;
                index01 = svld1(pg_new, &index45_host1[0]);
                 index02 = svadd_s32_m(pg, index01, svdup_s32(2));
                index03 = svadd_s32_m(pg, index02, svdup_s32(2));
                index04 = svadd_s32_m(pg, index03, svdup_s32(2));



                 svfloat32_t dout0, dout1, dout2, dout3, dout4, dout5;
                winograd_f6k3_output_transform1_intertile(
                        svld1_gather_s32index_f32( pg_new, &din101[0], index01), svld1_gather_s32index_f32( pg_new, &din101[0], index02), svld1_gather_s32index_f32( pg_new, &din101[0], index03), svld1_gather_s32index_f32( pg_new, &din101[0], index04), svld1_gather_s32index_f32( pg_new, &din123[0], index01), svld1_gather_s32index_f32( pg_new, &din123[0], index02), svld1_gather_s32index_f32( pg_new, &din123[0], index03), svld1_gather_s32index_f32( pg_new, &din123[0], index04), &dout0, &dout1, &dout2, &dout3, &dout4, &dout5);

                svst1_f32(pg_new, &block1[0], dout0);
                svst1_f32(pg_new, &block1[simd_width/2], dout2);
                svst1_f32(pg_new, &block1[2*(simd_width/2)], dout4);
                //////leave here
        
	
/*		   for (size_t i = 0; i < row_count; i++) {
                for (size_t j = 0; j < column_count; j++) {
                          if((j*2) <4){
                                output[0][i * output_stride + j] = block[i*simd_width+(j*2)];
                                output[1][i * output_stride + j] = block[i*simd_width+((j*2)+4)];
                                if(interchannels > 2) {output[2][i * output_stride + j] = block[i*simd_width+((j*2)+8)];}
                                if(interchannels > 3) {output[3][i * output_stride + j] = block[i*simd_width+((j*2)+12)];}
                                if(interchannels > 4) {output[4][i * output_stride + j] = block[i*simd_width+((j*2)+16)];}
                                if(interchannels > 5) {output[5][i * output_stride + j] = block[i*simd_width+((j*2)+20)];}
                                if(interchannels > 6) {output[6][i * output_stride + j] = block[i*simd_width+((j*2)+24)];}
                                if(interchannels > 7) {output[7][i * output_stride + j] = block[i*simd_width+((j*2)+28)];}
                                }
                                else if(((j*2)>3) && ((j*2)<6))
                                {
                                  output[0][i * output_stride + j] = block1[i*(simd_width/2)+ ((j*2)-4)];
                                output[1][i * output_stride + j] = block1[ i *(simd_width/2) +((j*2)-2) ];
                                if(interchannels > 2) {output[2][i * output_stride + j] = block1[ i*(simd_width/2)+((j*2)) ];}
                                if(interchannels > 3) {output[3][i * output_stride + j] =  block1[ i*(simd_width/2)+((j*2) +2)] ;}
                                if(interchannels > 4) {output[4][i * output_stride + j] = block1[ i*(simd_width/2)+((j*2))+4 ];}
                                if(interchannels > 5) {output[5][i * output_stride + j] = block1[ i*(simd_width/2)+((j*2))+6 ];}
                                if(interchannels > 6) {output[6][i * output_stride + j] =  block1[ i*(simd_width/2)+((j*2) +8)] ;}
                                if(interchannels > 7) {output[7][i * output_stride + j] =  block1[ i*(simd_width/2)+((j*2) +10)] ;}

                                }


                }
        }*/


	         for (size_t i = 0; i < row_count; i++) {
		#pragma loop unroll_count(interchannels)
                    for(int k=0;k<interchannels;k++){

                        for (size_t j = 0; j < column_count; j++) {  //original trying to change it 
                        	if(j*2<4)
				{
                                output[k][i * output_stride + j] = block[i*simd_width+((j*2)+k*4)];
                        	}
                    	}
		}

                 #pragma loop unroll_count(interchannels)
                 for(int k=0;k<interchannels;k++)
                  {
                        for (size_t j =0; j < column_count; j++) {  //original trying to change it 
				if(((j*2)>3) && ((j*2)<6))
                                {
                                  output[k][i * output_stride + j] = block1[i*(simd_width/2)+ (((j*2)+k*2)-4)];

                                }
		
                        }
                   }
		
		}

	}
free(new_data);
}

void nnp_owt8x8_3x3s2_with_bias__neon(
	const void *restrict transform,
	float output[restrict static 1],
	const float bias[restrict static 1],
	size_t transform_stride,
	size_t output_stride,
	uint32_t row_count,
	uint32_t column_count)
{
	NNP_SIMD_ALIGN float buffer[8 * 6];
	float*restrict qbuffer = buffer;
	float*restrict dbuffer = buffer + 32;
//	float32x2_t vbias = vreinterpret_f32_u64(vshl_n_u64(vreinterpret_u64_f32(vld1_dup_f32(bias)), 32));
	for (uint32_t col = 0; col < 2; col++) {
		const float32x4_t m0 = vld1q_f32(transform); transform += transform_stride;
		float32x4_t m1 = vld1q_f32(transform); transform += transform_stride;
		/* The only difference in the with_bias vs non with_bias case. */
	//	m1 = vcombine_f32(vadd_f32(vget_low_f32(m1), vbias), vget_high_f32(m1));
	//	vbias = vmov_n_f32(0.0f);
		const float32x4_t m2 = vld1q_f32(transform); transform += transform_stride;
		const float32x4_t m3 = vld1q_f32(transform); transform += transform_stride;
		const float32x4_t m4 = vld1q_f32(transform); transform += transform_stride;
		const float32x4_t m5 = vld1q_f32(transform); transform += transform_stride;
		const float32x4_t m6 = vld1q_f32(transform); transform += transform_stride;
		const float32x4_t m7 = vld1q_f32(transform); transform += transform_stride;
		float32x4_t o0, o1, o2, o3, o4, o5;
		winograd_f6k3_output_transformq(
			m0, m1, m2, m3, m4, m5, m6, m7,
			&o0, &o1, &o2, &o3, &o4, &o5);
		vst1q_f32(qbuffer, o0); qbuffer += 4;
		vst1q_f32(qbuffer, o1); qbuffer += 4;
		vst1q_f32(qbuffer, o2); qbuffer += 4;
		vst1q_f32(qbuffer, o3); qbuffer += 4;
		vst1_f32(dbuffer, vget_low_f32(o4)); dbuffer += 2;
		vst1_f32(dbuffer, vget_low_f32(o5)); dbuffer += 2;
		vst1_f32(dbuffer, vget_high_f32(o4)); dbuffer += 2;
		vst1_f32(dbuffer, vget_high_f32(o5)); dbuffer += 2;
	}

	const float*restrict read_ptr = buffer;
	NNP_SIMD_ALIGN float block[3][8];

	float32x4x4_t qin0123 = vld4q_f32(read_ptr); read_ptr += 16;
	float32x4x4_t qin4567 = vld4q_f32(read_ptr); read_ptr += 16;
	float32x4_t qout0, qout1, qout2, qout3, qout4, qout5;
	winograd_f6k3_output_transformq(
		qin0123.val[0], qin0123.val[1], qin0123.val[2], qin0123.val[3],
		qin4567.val[0], qin4567.val[1], qin4567.val[2], qin4567.val[3],
		&qout0, &qout1, &qout2, &qout3, &qout4, &qout5);
	vst1q_f32(&block[0][0], qout0);
	vst1q_f32(&block[1][0], qout2);
	vst1q_f32(&block[2][0], qout4);

	float32x2x2_t din01 = vld2_f32(read_ptr); read_ptr += 4;
	float32x2x2_t din23 = vld2_f32(read_ptr); read_ptr += 4;
	float32x2x2_t din45 = vld2_f32(read_ptr); read_ptr += 4;
	float32x2x2_t din67 = vld2_f32(read_ptr);
	float32x2_t dout0, dout1, dout2, dout3, dout4, dout5;
	winograd_f6k3_output_transform(
		din01.val[0], din01.val[1], din23.val[0], din23.val[1],
		din45.val[0], din45.val[1], din67.val[0], din67.val[1],
		&dout0, &dout1, &dout2, &dout3, &dout4, &dout5);
	vst1_f32(&block[0][4], dout0);
	vst1_f32(&block[1][4], dout2);
	vst1_f32(&block[2][4], dout4);

	for (size_t i = 0; i < row_count; i++) {
		for (size_t j = 0; j < column_count; j++) {
			output[i * output_stride + j] = block[i][j * 2];
		}
	}
}

void nnp_owt8x8_3x3_with_bias_with_relu__neon(
	const void *restrict transform,
	float output[restrict static 1],
	const float bias[restrict static 1],
	size_t transform_stride, size_t output_stride,
	uint32_t row_count, uint32_t column_count)
{
	NNP_SIMD_ALIGN float buffer[8 * 6];
	float*restrict qbuffer = buffer;
	float*restrict dbuffer = buffer + 32;
	float32x2_t vbias = vreinterpret_f32_u64(vshl_n_u64(vreinterpret_u64_f32(vld1_dup_f32(bias)), 32));
	for (uint32_t col = 0; col < 2; col++) {
		const float32x4_t m0 = vld1q_f32(transform); transform += transform_stride;
		float32x4_t m1 = vld1q_f32(transform); transform += transform_stride;
		/* The only difference in the with_bias vs non with_bias case. */
		m1 = vcombine_f32(vadd_f32(vget_low_f32(m1), vbias), vget_high_f32(m1));
		vbias = vmov_n_f32(0.0f);
		const float32x4_t m2 = vld1q_f32(transform); transform += transform_stride;
		const float32x4_t m3 = vld1q_f32(transform); transform += transform_stride;
		const float32x4_t m4 = vld1q_f32(transform); transform += transform_stride;
		const float32x4_t m5 = vld1q_f32(transform); transform += transform_stride;
		const float32x4_t m6 = vld1q_f32(transform); transform += transform_stride;
		const float32x4_t m7 = vld1q_f32(transform); transform += transform_stride;
		float32x4_t o0, o1, o2, o3, o4, o5;
		winograd_f6k3_output_transformq(
			m0, m1, m2, m3, m4, m5, m6, m7,
			&o0, &o1, &o2, &o3, &o4, &o5);
		vst1q_f32(qbuffer, o0); qbuffer += 4;
		vst1q_f32(qbuffer, o1); qbuffer += 4;
		vst1q_f32(qbuffer, o2); qbuffer += 4;
		vst1q_f32(qbuffer, o3); qbuffer += 4;
		vst1_f32(dbuffer, vget_low_f32(o4)); dbuffer += 2;
		vst1_f32(dbuffer, vget_low_f32(o5)); dbuffer += 2;
		vst1_f32(dbuffer, vget_high_f32(o4)); dbuffer += 2;
		vst1_f32(dbuffer, vget_high_f32(o5)); dbuffer += 2;
	}

	const float*restrict read_ptr = buffer;
	if NNP_LIKELY(row_count == 6 && column_count == 6 && output_stride >= 6) {
		// Fast path to reuse `s` array and write directly into `output`.
		float32x4x4_t qin0123 = vld4q_f32(read_ptr); read_ptr += 16;
		float32x4x4_t qin4567 = vld4q_f32(read_ptr); read_ptr += 16;
		float32x4_t qout0, qout1, qout2, qout3, qout4, qout5;
		winograd_f6k3_output_transformq(
			qin0123.val[0], qin0123.val[1], qin0123.val[2], qin0123.val[3],
			qin4567.val[0], qin4567.val[1], qin4567.val[2], qin4567.val[3],
			&qout0, &qout1, &qout2, &qout3, &qout4, &qout5);
		float* output_col0123 = output;
		const float32x4_t qzero = vmovq_n_f32(0.0f);
		vst1q_f32(output_col0123, neon_reluq_f32(qout0, qzero)); output_col0123 += output_stride;
		vst1q_f32(output_col0123, neon_reluq_f32(qout1, qzero)); output_col0123 += output_stride;
		vst1q_f32(output_col0123, neon_reluq_f32(qout2, qzero)); output_col0123 += output_stride;
		vst1q_f32(output_col0123, neon_reluq_f32(qout3, qzero)); output_col0123 += output_stride;
		vst1q_f32(output_col0123, neon_reluq_f32(qout4, qzero)); output_col0123 += output_stride;
		vst1q_f32(output_col0123, neon_reluq_f32(qout5, qzero));

		float32x2x2_t din01 = vld2_f32(read_ptr); read_ptr += 4;
		float32x2x2_t din23 = vld2_f32(read_ptr); read_ptr += 4;
		float32x2x2_t din45 = vld2_f32(read_ptr); read_ptr += 4;
		float32x2x2_t din67 = vld2_f32(read_ptr);
		float32x2_t dout0, dout1, dout2, dout3, dout4, dout5;
		winograd_f6k3_output_transform(
			din01.val[0], din01.val[1], din23.val[0], din23.val[1],
			din45.val[0], din45.val[1], din67.val[0], din67.val[1],
			&dout0, &dout1, &dout2, &dout3, &dout4, &dout5);
		float* output_col45 = output + 4;
		const float32x2_t dzero = vmov_n_f32(0.0f);
		vst1_f32(output_col45, neon_relu_f32(dout0, dzero)); output_col45 += output_stride;
		vst1_f32(output_col45, neon_relu_f32(dout1, dzero)); output_col45 += output_stride;
		vst1_f32(output_col45, neon_relu_f32(dout2, dzero)); output_col45 += output_stride;
		vst1_f32(output_col45, neon_relu_f32(dout3, dzero)); output_col45 += output_stride;
		vst1_f32(output_col45, neon_relu_f32(dout4, dzero)); output_col45 += output_stride;
		vst1_f32(output_col45, neon_relu_f32(dout5, dzero));
	} else {
		NNP_SIMD_ALIGN float block[6][8];

		float32x4x4_t qin0123 = vld4q_f32(read_ptr); read_ptr += 16;
		float32x4x4_t qin4567 = vld4q_f32(read_ptr); read_ptr += 16;
		float32x4_t qout0, qout1, qout2, qout3, qout4, qout5;
		winograd_f6k3_output_transformq(
			qin0123.val[0], qin0123.val[1], qin0123.val[2], qin0123.val[3],
			qin4567.val[0], qin4567.val[1], qin4567.val[2], qin4567.val[3],
			&qout0, &qout1, &qout2, &qout3, &qout4, &qout5);
		const float32x4_t qzero = vmovq_n_f32(0.0f);
		vst1q_f32(&block[0][0], neon_reluq_f32(qout0, qzero));
		vst1q_f32(&block[1][0], neon_reluq_f32(qout1, qzero));
		vst1q_f32(&block[2][0], neon_reluq_f32(qout2, qzero));
		vst1q_f32(&block[3][0], neon_reluq_f32(qout3, qzero));
		vst1q_f32(&block[4][0], neon_reluq_f32(qout4, qzero));
		vst1q_f32(&block[5][0], neon_reluq_f32(qout5, qzero));

		float32x2x2_t din01 = vld2_f32(read_ptr); read_ptr += 4;
		float32x2x2_t din23 = vld2_f32(read_ptr); read_ptr += 4;
		float32x2x2_t din45 = vld2_f32(read_ptr); read_ptr += 4;
		float32x2x2_t din67 = vld2_f32(read_ptr);
		float32x2_t dout0, dout1, dout2, dout3, dout4, dout5;
		winograd_f6k3_output_transform(
			din01.val[0], din01.val[1], din23.val[0], din23.val[1],
			din45.val[0], din45.val[1], din67.val[0], din67.val[1],
			&dout0, &dout1, &dout2, &dout3, &dout4, &dout5);
		const float32x2_t dzero = vmov_n_f32(0.0f);
		vst1_f32(&block[0][4], neon_relu_f32(dout0, dzero));
		vst1_f32(&block[1][4], neon_relu_f32(dout1, dzero));
		vst1_f32(&block[2][4], neon_relu_f32(dout2, dzero));
		vst1_f32(&block[3][4], neon_relu_f32(dout3, dzero));
		vst1_f32(&block[4][4], neon_relu_f32(dout4, dzero));
		vst1_f32(&block[5][4], neon_relu_f32(dout5, dzero));

		for (size_t i = 0; i < row_count; i++) {
			for (size_t j = 0; j < column_count; j++) {
				output[i * output_stride + j] = block[i][j];
			}
		}
	}
}

void nnp_owt8x8_3x3s2_with_bias_with_relu__neon(
	const void *restrict transform,
	float output[restrict static 1],
	const float bias[restrict static 1],
	size_t transform_stride, size_t output_stride,
	uint32_t row_count, uint32_t column_count)
{
	NNP_SIMD_ALIGN float buffer[8 * 6];
	float*restrict qbuffer = buffer;
	float*restrict dbuffer = buffer + 32;
	float32x2_t vbias = vreinterpret_f32_u64(vshl_n_u64(vreinterpret_u64_f32(vld1_dup_f32(bias)), 32));
	for (uint32_t col = 0; col < 2; col++) {
		const float32x4_t m0 = vld1q_f32(transform); transform += transform_stride;
		float32x4_t m1 = vld1q_f32(transform); transform += transform_stride;
		/* The only difference in the with_bias vs non with_bias case. */
		m1 = vcombine_f32(vadd_f32(vget_low_f32(m1), vbias), vget_high_f32(m1));
		vbias = vmov_n_f32(0.0f);
		const float32x4_t m2 = vld1q_f32(transform); transform += transform_stride;
		const float32x4_t m3 = vld1q_f32(transform); transform += transform_stride;
		const float32x4_t m4 = vld1q_f32(transform); transform += transform_stride;
		const float32x4_t m5 = vld1q_f32(transform); transform += transform_stride;
		const float32x4_t m6 = vld1q_f32(transform); transform += transform_stride;
		const float32x4_t m7 = vld1q_f32(transform); transform += transform_stride;
		float32x4_t o0, o1, o2, o3, o4, o5;
		winograd_f6k3_output_transformq(
			m0, m1, m2, m3, m4, m5, m6, m7,
			&o0, &o1, &o2, &o3, &o4, &o5);
		vst1q_f32(qbuffer, o0); qbuffer += 4;
		vst1q_f32(qbuffer, o1); qbuffer += 4;
		vst1q_f32(qbuffer, o2); qbuffer += 4;
		vst1q_f32(qbuffer, o3); qbuffer += 4;
		vst1_f32(dbuffer, vget_low_f32(o4)); dbuffer += 2;
		vst1_f32(dbuffer, vget_low_f32(o5)); dbuffer += 2;
		vst1_f32(dbuffer, vget_high_f32(o4)); dbuffer += 2;
		vst1_f32(dbuffer, vget_high_f32(o5)); dbuffer += 2;
	}

	const float*restrict read_ptr = buffer;
	NNP_SIMD_ALIGN float block[3][8];

	float32x4x4_t qin0123 = vld4q_f32(read_ptr); read_ptr += 16;
	float32x4x4_t qin4567 = vld4q_f32(read_ptr); read_ptr += 16;
	float32x4_t qout0, qout1, qout2, qout3, qout4, qout5;
	winograd_f6k3_output_transformq(
		qin0123.val[0], qin0123.val[1], qin0123.val[2], qin0123.val[3],
		qin4567.val[0], qin4567.val[1], qin4567.val[2], qin4567.val[3],
		&qout0, &qout1, &qout2, &qout3, &qout4, &qout5);
	const float32x4_t qzero = vmovq_n_f32(0.0f);
	vst1q_f32(&block[0][0], neon_reluq_f32(qout0, qzero));
	vst1q_f32(&block[1][0], neon_reluq_f32(qout2, qzero));
	vst1q_f32(&block[2][0], neon_reluq_f32(qout4, qzero));

	float32x2x2_t din01 = vld2_f32(read_ptr); read_ptr += 4;
	float32x2x2_t din23 = vld2_f32(read_ptr); read_ptr += 4;
	float32x2x2_t din45 = vld2_f32(read_ptr); read_ptr += 4;
	float32x2x2_t din67 = vld2_f32(read_ptr);
	float32x2_t dout0, dout1, dout2, dout3, dout4, dout5;
	winograd_f6k3_output_transform(
		din01.val[0], din01.val[1], din23.val[0], din23.val[1],
		din45.val[0], din45.val[1], din67.val[0], din67.val[1],
		&dout0, &dout1, &dout2, &dout3, &dout4, &dout5);
	const float32x2_t dzero = vmov_n_f32(0.0f);
	vst1_f32(&block[0][4], neon_relu_f32(dout0, dzero));
	vst1_f32(&block[1][4], neon_relu_f32(dout2, dzero));
	vst1_f32(&block[2][4], neon_relu_f32(dout4, dzero));

	for (size_t i = 0; i < row_count; i++) {
		for (size_t j = 0; j < column_count; j++) {
			output[i * output_stride + j] = block[i][j * 2];
		}
	}
}
