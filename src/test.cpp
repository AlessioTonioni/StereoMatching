#include <opencv2/opencv.hpp>
#include <opencv2\stereo.hpp>
#include "stereo_functions.hpp"
#include <numeric>

int main(int argc, char**argv){
	int p1 = 100;
	int p2 = 1000;

	int maxDisp = 32;
	int blocksize = 7;
	std::string left_image = "data/tsukuba/scene1.row3.col1.ppm";
	std::string right_image = "data/tsukuba/scene1.row3.col2.ppm";

	cv::Mat left = cv::imread(left_image, CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat right = cv::imread(right_image, CV_LOAD_IMAGE_GRAYSCALE);


	cv::imshow("Left", left);
	//cv::imshow("Right",right);
	cv::waitKey();

	cv::Ptr<cv::StereoBM> pippo = cv::StereoBM::create(maxDisp, blocksize);
	cv::Ptr<cv::StereoSGBM> pluto = cv::StereoSGBM::create(0, maxDisp, blocksize, p1, p2);
	cv::Mat disparity_pippo;
	cv::Mat disparity_pluto;
	std::cout << "Opencv Block Matcher" << std::endl;
	pippo->compute(left, right, disparity_pippo);
	std::cout << "Opencv SGM " << std::endl;
	pluto->compute(left, right, disparity_pluto);
	cv::normalize(disparity_pippo, disparity_pippo, 0, 255, cv::NORM_MINMAX, CV_8UC1);
	cv::normalize(disparity_pluto, disparity_pluto, 0, 255, cv::NORM_MINMAX, CV_8UC1);
	cv::imshow("BlockMatcher_Opencv", disparity_pippo);
	cv::imshow("SGBM_Opencv", disparity_pluto);
	cv::waitKey(0);

	std::cout << "My fixed window" << std::endl;

	std::unique_ptr<stereo::costs::costFunction> c = std::make_unique<stereo::costs::truncatedAbsoluteDifference>(25);
	//std::unique_ptr<stereo::costs::costFunction> c = std::make_unique<stereo::costs::absoluteDifference>();
	//std::unique_ptr<stereo::costs::costFunction> c = std::make_unique<stereo::costs::squarredDifference>();

	cv::Mat DSI = stereo::disp::fixedWindowBF(left, right, stereo::disp::getDisparityRange(maxDisp), blocksize / 2, c);
	//cv::Mat DSI_right = stereo::disp::fixedWindowBF(right, left, stereo::disp::getDisparityRange(maxDisp, false), blocksize / 2, c);

	std::cout << "Confidence" << std::endl;
	//cv::Mat confidence = stereo::confidence::matchingScoreMeasure(DSI);
	//cv::Mat confidence = stereo::confidence::curvature(DSI);
	//cv::Mat confidence = stereo::confidence::peakRatioNaive(DSI);
	cv::Mat confidence = stereo::confidence::winnerMarginNaive(DSI);
	//cv::Mat confidence = stereo::confidence::leftRightCheck(DSI, DSI_right);

	cv::Mat disp = stereo::disp::getDisparity(DSI);
	std::cout << "Scanline optimization" << std::endl;
	DSI = stereo::sgm::sgmOptimization(DSI, p1, p2);
	std::cout << "Done" << std::endl;
	cv::Mat disp2 = stereo::disp::getDisparity(DSI);
	
	cv::normalize(disp, disp, 0, 255, cv::NORM_MINMAX, CV_8UC1);
	cv::normalize(disp2, disp2, 0, 255, cv::NORM_MINMAX, CV_8UC1);
	cv::imshow("disp", disp);
	cv::imshow("disp_scaline", disp2);
	cv::imshow("confidence", confidence);
	cv::waitKey();
}