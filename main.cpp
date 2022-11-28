#include "lib.h"

int main(void) {

	float P0[] = { 718.856f, 0.0f, 607.1928f, 0.0f, 0.0f, 718.856f, 185.2157f, 0.0f,0.0f,0.0f,1.0f, 0.0f };
	float P1[] = { 718.856f, 0.0f, 607.1928f, -386.1448f, 0.0f, 718.856f, 185.2157f, 0.0f,0.0f,0.0f,1.0f, 0.0f };
	Mat P_left(3, 4, cv::DataType<float>::type, P0);
	Mat P_right(3, 4, cv::DataType<float>::type, P1);
	std::vector<int> dynamic_classes = {8,9,10};
	std::string folder_left = "../lil_dataset/00/image_0/";
	std::string folder_right = "../lil_dataset/00/image_1";
	std::string segment_folder = "../lil_dataset/00/segment/dataset";
	std::string input  = "optimized_results.txt";
	//VisualNoDynamic(folder_left, segment_folder, folder_right, input, P_left, P_right, dynamic_classes);
	EstimateAndOptimize(folder_left, folder_right, input, P_left, P_right);
	return 0;
}
