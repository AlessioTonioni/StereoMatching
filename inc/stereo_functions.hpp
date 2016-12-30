#ifndef STEREO_FUNCTIONS_H_
#define STEREO_FUNCTIONS_H_

#include <opencv2/opencv.hpp>
#include <thread>
#include <functional>
#include <memory>

namespace stereo{
	namespace costs{
		class costFunction{
		public:
			costFunction(){};
			virtual float operator()(float a, float b) = 0;
			virtual float operator()(const cv::Mat& a, const cv::Mat& b)=0;
		};

		class absoluteDifference: public costFunction{
		public:
			absoluteDifference(){};
			float operator()(float a, float b);
			float operator()(const cv::Mat& a, const cv::Mat& b);
		};

		class squarredDifference:public costFunction{
		public:
			squarredDifference(){};
			float operator()(float a, float b);
			float operator()(const cv::Mat& a, const cv::Mat& b);
		};

		class truncatedAbsoluteDifference:public costFunction{
		private:
			float truncation;
		public:
			truncatedAbsoluteDifference(float trunc) :truncation(trunc){};

			float operator()(float a, float b);
			float operator()(const cv::Mat& a, const cv::Mat& b);
		};
	}

	namespace disp{
		std::vector<int> getDisparityRange(const int maxDisp, const bool isLeftRight=true);
		cv::Mat singlePixelDisparity(const cv::Mat& left, const cv::Mat& right, std::vector<int> disparities, const std::unique_ptr<costs::costFunction>& cost_function);
		cv::Mat fixedWindowDisparity(const cv::Mat& left, const cv::Mat& right, std::vector<int> disparities, int support, const std::unique_ptr<costs::costFunction>& cost_function);
		cv::Mat fixedWindowBF(const cv::Mat& left, const cv::Mat& right, std::vector<int> disparities, int support,const std::unique_ptr<costs::costFunction>& cost_function);
		cv::Mat getDisparity(const cv::Mat & DSI);
	}

	namespace sgm{
		void scanlineOptimization(cv::Mat& result,const cv::Mat& DSI, const cv::Vec2i & previousPosition, const float p1, const float p2);
		cv::Mat sgmOptimization(const cv::Mat& DSI, const float p1, const float p2);
	}

	namespace confidence{
		cv::Mat matchingScoreMeasure(const cv::Mat& DSI);
		cv::Mat curvature(const cv::Mat& DSI);
		cv::Mat peakRatioNaive(const cv::Mat& DSI);
		cv::Mat winnerMarginNaive(const cv::Mat& DSI);
		cv::Mat leftRightCheck(const cv::Mat& DSI_left, const cv::Mat& DSI_right);
	}
}
#endif