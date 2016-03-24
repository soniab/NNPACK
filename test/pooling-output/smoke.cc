#include <gtest/gtest.h>

#include <nnpack.h>

#include <testers/pooling.h>

/*
 * Test that implementation works for a single-channel image with a single-pool
 */

TEST(MaxPooling2x2, single_pool) {
	PoolingTester()
		.inputSize(2, 2)
		.poolingSize(2, 2)
		.poolingStride(2, 2)
		.iterations(100)
		.testOutput();
}

/*
 * Test that implementation works for a single-channel image with few horizontal pools
 */

TEST(MaxPooling2x2, few_horizontal_pools) {
	for (size_t imageWidth = 4; imageWidth <= 50; imageWidth += 2) {
		PoolingTester()
			.inputSize(2, imageWidth)
			.poolingSize(2, 2)
			.poolingStride(2, 2)
			.iterations(100)
			.testOutput();
	}
}

/*
 * Test that implementation works for a single-channel image with few vertical pools
 */

TEST(MaxPooling2x2, few_vertical_pools) {
	for (size_t imageHeight = 4; imageHeight <= 50; imageHeight += 2) {
		PoolingTester()
			.inputSize(imageHeight, 2)
			.poolingSize(2, 2)
			.poolingStride(2, 2)
			.iterations(100)
			.testOutput();
	}
}

/*
 * Test that implementation works for a single-channel image with multiple horizontal and vertical pools
 */

TEST(MaxPooling2x2, large_image) {
	PoolingTester()
		.inputSize(128, 128)
		.poolingSize(2, 2)
		.poolingStride(2, 2)
		.iterations(100)
		.testOutput();
}

/*
 * Test that implementation works for a single-channel image with size which is not perfectly divisible by the pool size
 */

TEST(MaxPooling2x2, indivisible_size) {
	PoolingTester()
		.inputSize(5, 5)
		.poolingSize(2, 2)
		.poolingStride(2, 2)
		.iterations(100)
		.testOutput();
}

/*
 * Test that implementation works for a single-channel image with implicit padding
 */

TEST(MaxPooling2x2, DISABLED_implicit_padding) {
	PoolingTester tester;
	tester.inputSize(24, 24)
		.poolingSize(2, 2)
		.poolingStride(2, 2)
		.iterations(100);
	for (size_t paddingTop = 0; paddingTop < tester.kernelHeight(); paddingTop++) {
		for (size_t paddingRight = 0; paddingRight < tester.kernelWidth(); paddingRight++) {
			for (size_t paddingLeft = 0; paddingLeft < tester.kernelWidth(); paddingLeft++) {
				for (size_t paddingBottom = 0; paddingBottom < tester.kernelHeight(); paddingBottom++) {
					tester.inputPadding(paddingTop, paddingRight, paddingLeft, paddingBottom)
						.testOutput();
				}
			}
		}
	}
}

/*
 * Test that implementation can handle small non-unit batch_size
 */

TEST(MaxPooling2x2, small_batch) {
	PoolingTester tester;
	tester.inputSize(12, 12)
		.poolingSize(2, 2)
		.poolingStride(2, 2)
		.iterations(100);
	for (size_t batchSize = 2; batchSize <= 5; batchSize++) {
		tester.batchSize(batchSize)
			.testOutput();
	}
}

/*
 * Test that implementation can handle small non-unit number of channels
 */

TEST(MaxPooling2x2, few_channels) {
	PoolingTester tester;
	tester.inputSize(12, 12)
		.poolingSize(2, 2)
		.poolingStride(2, 2)
		.iterations(100);
	for (size_t channels = 2; channels <= 5; channels++) {
		tester.channels(channels)
			.testOutput();
	}
}

int main(int argc, char* argv[]) {
	const enum nnp_status init_status = nnp_initialize();
	assert(init_status == nnp_status_success);
	setenv("TERM", "xterm-256color", 0);
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
