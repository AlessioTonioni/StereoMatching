#include "stereo_functions.hpp"
#include <numeric>

//========================================COST FUCNTION=========================================================
float stereo::costs::absoluteDifference::operator()(float a, float b){
	return std::abs(a - b);
}
float stereo::costs::absoluteDifference::operator()(const cv::Mat& a, const cv::Mat& b){
	return cv::sum(cv::abs(a - b))[0];
}

float stereo::costs::squarredDifference::operator()(float a, float b){
	return std::pow((a - b), 2);
}
float stereo::costs::squarredDifference::operator()(const cv::Mat& a, const cv::Mat& b){
	cv::Mat a_f, b_f, t;
	a.convertTo(a_f, cv::DataType<int>::type);
	b.convertTo(b_f, cv::DataType<int>::type);
	cv::pow(a_f - b_f, 2, t);
	return cv::sum(t)[0];
}

float stereo::costs::truncatedAbsoluteDifference::operator()(float a, float b){
	float ad = std::abs(a-b);
	return (ad<this->truncation) ? ad : this->truncation;
}
float stereo::costs::truncatedAbsoluteDifference::operator()(const cv::Mat& a, const cv::Mat& b){
	cv::Mat pars = cv::abs(a - b);
	for (int r = 0; r < pars.rows; r++){
		for (int c = 0; c < pars.cols; c++)
			pars.at<uchar>(r, c) = (pars.at<uchar>(r, c) < this->truncation) ? pars.at<uchar>(r, c) : this->truncation;
	}
	return cv::sum(pars)[0];
}


//=========================================DISPARITY=============================================================
std::vector<int> stereo::disp::getDisparityRange(const int maxDisp,const bool isLeftRight){
	std::vector<int> result;
	for (int i = 0; i < maxDisp; i++){
		int val = isLeftRight ? -i : i;
		result.push_back(val);
	}
	return result;
}

cv::Mat stereo::disp::getDisparity(const cv::Mat& DSI){
    int rows = DSI.size[0];
    int cols = DSI.size[1];
    int maxDisp = DSI.size[2];
    cv::Mat disp(cv::Size(cols,rows),CV_8UC1);
    for(int r=0; r<rows; r++){
        for(int c=0; c<cols; c++){
            int disp_value=0;
            float best_cost = std::numeric_limits<float>::max();
            for(int delta=0; delta<maxDisp; delta++){
                float cost = DSI.at<float>(r,c,delta);
                if(cost<best_cost){
                    best_cost=cost;
                    disp_value = delta;
                }
            }
            disp.at<uchar>(r,c)=disp_value;
        }
    }
    return disp;
}

cv::Mat stereo::disp::singlePixelDisparity(const cv::Mat& left, const cv::Mat &right, std::vector<int> disparities, const std::unique_ptr<stereo::costs::costFunction>& cost_function){
    int dims[3]={left.rows, left.cols, disparities.size()};
    cv::Mat DSI(3,dims,CV_32F);

	for (int r = 0; r<left.rows; r++){
		for (int c = 0; c<left.cols; c++){
			float left_value = left.at<uchar>(r, c);
			for (int i = 0; i < disparities.size(); i++){
				int delta = disparities[i];
				if (c + delta > 0 && c+delta<right.cols)
					DSI.at<float>(r, c, i) = (*cost_function)(left_value, (float)right.at<uchar>(r, c + delta));
				else
					DSI.at<float>(r, c, i) = std::numeric_limits<float>::max();
			}
		}
	}
	return DSI;
   
}

cv::Mat stereo::disp::fixedWindowDisparity(const cv::Mat& left, const cv::Mat &right, std::vector<int> disparities, int support, const std::unique_ptr<stereo::costs::costFunction>& cost_function){
    int dims[3]={left.rows, left.cols, disparities.size()};

    cv::Mat DSI(3,dims,CV_32F);

	for (int r = support; r<left.rows - support; r++){
		for (int c = support; c<left.cols - support; c++){
			cv::Mat left_value = left(cv::Range(r - support, r + support + 1), cv::Range(c - support, c + support + 1));
			for (int i = 0; i < disparities.size(); i++){
				int delta = disparities[i];
				if (c + delta - support >= 0 && c+delta+support<right.cols && c + support<right.cols && r + support<right.rows){
					cv::Mat right_value = right(cv::Range(r - support, r + support + 1), cv::Range(c + delta - support, c + delta + support + 1));
					DSI.at<float>(r, c, i) = (*cost_function)(left_value, right_value);
				}
				else
					DSI.at<float>(r, c, i) = (float)std::numeric_limits<float>::max();
			}
		}
	}
	return DSI;
}

cv::Mat stereo::disp::fixedWindowBF(const cv::Mat& left, const cv::Mat &right, std::vector<int> disparities, int support, const std::unique_ptr<stereo::costs::costFunction>& cost_function){
	int dims[3] = { left.rows, left.cols, disparities.size() };
	cv::Mat DSI(3, dims, CV_32F);

	cv::Mat cache = cv::Mat::zeros(cv::Size(disparities.size(),left.cols), cv::DataType<float>::type);
	std::vector<float> lastUTop(disparities.size(),0);
	std::vector<float> lastUBottom(disparities.size(), 0);
	for (int r = support; r < left.rows - support; r++){
		for (int c = support; c < left.cols - support; c++){
			cv::Mat left_value = left(cv::Range(r - support, r + support + 1), cv::Range(c - support, c + support + 1));
			for (int i = 0; i < disparities.size(); i++){
				int delta = disparities[i];
				if (c + delta - support >= 0 && c + delta + support<right.cols && c + support<right.cols && r + support<right.rows){
					if (r == support){
						cv::Mat right_value = right(cv::Range(r - support, r + support + 1), cv::Range(c + delta - support, c + delta + support + 1));
						float cost = (*cost_function)(left_value, right_value);
						DSI.at<float>(r, c, i) = cost;
						cache.at<float>(c, i) = cost;
					}
					else if (c-support+delta==0 || c-support==0){
						float last_cost = cache.at<float>(c, i);
						cv::Mat previous_row_left = left(cv::Range(r - support - 1, r - support), cv::Range(c - support, c + support + 1));
						cv::Mat previous_row_right = right(cv::Range(r - support - 1, r - support), cv::Range(c +delta - support, c+delta + support + 1));
						cv::Mat new_row_left = left(cv::Range(r + support, r + support + 1), cv::Range(c - support, c + support + 1));
						cv::Mat new_row_right = right(cv::Range(r + support, r + support + 1), cv::Range(c +delta - support, c +delta + support + 1));
						float old_cost = (*cost_function)(previous_row_left, previous_row_right);
						float new_cost = (*cost_function)(new_row_left, new_row_right);
						lastUTop[i] = old_cost;
						lastUBottom[i] = new_cost;
						float new_val = cache.at<float>(c, i) + new_cost-old_cost;
						DSI.at<float>(r, c, i) = new_val;
						cache.at<float>(c, i) = new_val;
					}
					else {
						float a_diff = (*cost_function)(left.at<uchar>(r - support, c - support - 1), right.at<uchar>(r - support, c - support + delta -1));
						float b_diff = (*cost_function)(left.at<uchar>(r - support, c + support), right.at<uchar>(r - support, c + support + delta));
						float c_diff = (*cost_function)(left.at<uchar>(r + support, c + support), right.at < uchar>(r + support, c + support + delta));
						float d_diff = (*cost_function)(left.at<uchar>(r + support, c - support - 1), right.at<uchar>(r + support, c - support + delta -1));
						lastUTop[i] = lastUTop[i] + b_diff - a_diff;
						lastUBottom[i] = lastUBottom[i] + c_diff - d_diff;
						float new_cost = cache.at<float>(c, i) + lastUBottom[i] -lastUTop[i];
						DSI.at<float>(r, c, i) = new_cost;
						cache.at<float>(c, i) = new_cost;
					}
				}
				else
					DSI.at<float>(r, c, i) = (float)std::numeric_limits<float>::max();
			}
		}
	}
	return DSI;
}


//===========================================SGM==================================================================
void stereo::sgm::scanlineOptimization(cv::Mat& result, const cv::Mat& DSI, const cv::Vec2i & previousPosition, const float p1, const float p2){
	DSI.copyTo(result);
	bool left_right = previousPosition[0] <= 0;
	bool top_down = previousPosition[1] <= 0;
	int startRow = (top_down) ? 0 : DSI.size[0];
	int startCol = (left_right) ? 0 : DSI.size[1];

	for (int r = startRow; std::abs(r-startRow)<DSI.size[0];){
		for (int c = startCol; std::abs(c-startCol)<DSI.size[1];){
			for (int d = 0; d < DSI.size[2]; d++){

				int pr = r + previousPosition[1];
				int pc = c + previousPosition[0];
				if (pr >= 0 && pc >= 0 && pr<DSI.size[0] && pc<DSI.size[1]){
					float min = std::numeric_limits<float>::max();
					for (int pd = 0; pd < DSI.size[2]; pd++){
						float penality = (pd - d == 0) ? 0 : (std::abs(pd - d) == 1) ? p1 : p2;
						float t = result.at<float>(pr, pc, pd) + penality;
						min = (min < t) ? min : t;
					}
					result.at<float>(r, c, d) = result.at<float>(r, c, d) + min;
				}
			}
			c = (left_right) ? c + 1 : c - 1;
		}
		r = (top_down) ? r + 1 : r - 1;
	}
}

cv::Mat stereo::sgm::sgmOptimization(const cv::Mat& DSI, const float p1, const float p2){
	//ovest->est
	cv::Mat lr;
	std::thread t_lr(std::bind(scanlineOptimization,std::ref(lr),std::ref(DSI), cv::Vec2i(-1, 0), p1, p2));
	//est->ovest
	cv::Mat rl;
	std::thread t_rl(std::bind(scanlineOptimization,std::ref(rl),std::ref(DSI), cv::Vec2i(1, 0), p1, p2));
	//nord->sud
	cv::Mat tb;
	std::thread t_tb(std::bind(scanlineOptimization,std::ref(tb),std::ref(DSI), cv::Vec2i(0, -1), p1, p2));
	//sud->nord
	cv::Mat bt;
	std::thread t_bt(std::bind(scanlineOptimization,std::ref(bt),std::ref(DSI), cv::Vec2i(0, 1), p1, p2));
	//nord/ovest->sud/est
	cv::Mat tl;
	std::thread t_tl(std::bind(scanlineOptimization,std::ref(tl),std::ref(DSI), cv::Vec2i(-1, -1), p1, p2));
	//sud/est->nord/ovest
	cv::Mat br;
	std::thread t_br(std::bind(scanlineOptimization,std::ref(br),std::ref(DSI), cv::Vec2i(1, 1), p1, p2));
	//nord/est->sud/ovest
	cv::Mat tr;
	std::thread t_tf(std::bind(scanlineOptimization,std::ref(tr),std::ref(DSI), cv::Vec2i(1, -1), p1, p2));
	//sud/ovest->nord/est
	cv::Mat bl;
	std::thread t_bl(std::bind(scanlineOptimization,std::ref(bl),std::ref(DSI), cv::Vec2i(-1, 1), p1, p2));

	t_lr.join();
	t_rl.join();
	t_tb.join();
	t_bt.join();
	t_tl.join();
	t_br.join();
	t_tf.join();
	t_bl.join();
	return lr + rl + tb + bt + tl + tr + bl + br;
}


//==========================================CONFIDENCE========================================================
cv::Mat stereo::confidence::matchingScoreMeasure(const cv::Mat& DSI){
	int rows = DSI.size[0];
	int cols = DSI.size[1];
	int maxDisp = DSI.size[2];
	cv::Mat conf(cv::Size(cols, rows), cv::DataType<float>::type);
	for (int r = 0; r<rows; r++){
		for (int c = 0; c<cols; c++){
			float best_cost = std::numeric_limits<float>::max();
			for (int delta = 0; delta<maxDisp; delta++){
				float cost = DSI.at<float>(r, c, delta);
				best_cost = (best_cost < cost) ? best_cost : cost;
			}
			conf.at<float>(r, c) = -best_cost;
		}
	}
	conf = conf(cv::Range(0, rows), cv::Range(maxDisp, cols));
	cv::normalize(conf, conf, 0, 255, cv::NORM_MINMAX, CV_8UC1);
	return conf;
}

cv::Mat stereo::confidence::curvature(const cv::Mat& DSI){
	int rows = DSI.size[0];
	int cols = DSI.size[1];
	int maxDisp = DSI.size[2];
	cv::Mat conf(cv::Size(cols, rows), cv::DataType<float>::type);
	for (int r = 0; r<rows; r++){
		for (int c = 0; c<cols; c++){
			int disp_value = 0;
			float best_cost = std::numeric_limits<float>::max();
			for (int delta = 0; delta<maxDisp; delta++){
				float cost = DSI.at<float>(r, c, delta);
				if (cost < best_cost) {
					best_cost = cost;
					disp_value = delta;
				}
			}
			float previous_cost = (disp_value - 1 >= 0)?DSI.at<float>(r, c, disp_value - 1) : 0;
			float next_cost = (disp_value + 1 < maxDisp) ? DSI.at<float>(r, c, disp_value + 1) : 0;
			conf.at<float>(r, c) = -2*best_cost+previous_cost+next_cost ;
		}
	}
	conf = conf(cv::Range(0, rows), cv::Range(maxDisp, cols));
	cv::normalize(conf, conf, 0, 255, cv::NORM_MINMAX, CV_8UC1);
	return conf;
}

cv::Mat stereo::confidence::peakRatioNaive(const cv::Mat & DSI){
	int rows = DSI.size[0];
	int cols = DSI.size[1];
	int maxDisp = DSI.size[2];
	cv::Mat conf(cv::Size(cols, rows), cv::DataType<float>::type);
	for (int r = 0; r<rows; r++){
		for (int c = 0; c<cols; c++){
			float best_cost = std::numeric_limits<float>::max();
			float second_best = std::numeric_limits<float>::max();
			for (int delta = 0; delta<maxDisp; delta++){
				float cost = DSI.at<float>(r, c, delta);
				if (cost < best_cost){
					second_best = best_cost;
					best_cost = cost;
				} else if (cost<second_best){
					second_best = cost;
				}
				
			}
			float val = (best_cost!=0)? second_best / best_cost:1;
			conf.at<float>(r, c) = val;

		}
	}
	conf = conf(cv::Range(0, rows), cv::Range(maxDisp, cols));
	cv::normalize(conf, conf, 0, 255, cv::NORM_MINMAX, CV_8UC1);
	return conf;
}

cv::Mat stereo::confidence::winnerMarginNaive(const cv::Mat& DSI){
	int rows = DSI.size[0];
	int cols = DSI.size[1];
	int maxDisp = DSI.size[2];
	cv::Mat conf(cv::Size(cols, rows), cv::DataType<float>::type);
	for (int r = 0; r<rows; r++){
		for (int c = 0; c<cols; c++){
			float best_cost = std::numeric_limits<float>::max();
			float second_best = std::numeric_limits<float>::max();
			float total_cost = 0;
			for (int delta = 0; delta<maxDisp; delta++){
				float cost = DSI.at<float>(r, c, delta);
				if (cost < best_cost){
					second_best = best_cost;
					best_cost = cost;
				}
				else if (cost<second_best){
					second_best = cost;
				}
				total_cost += cost;

			}
			float val = (total_cost != 0) ? (second_best - best_cost)/total_cost : 0;
			conf.at<float>(r, c) = val;

		}
	}
	conf = conf(cv::Range(0, rows), cv::Range(maxDisp, cols));
	cv::normalize(conf, conf, 0, 255, cv::NORM_MINMAX, CV_8UC1);
	return conf;
}

cv::Mat stereo::confidence::leftRightCheck(const cv::Mat& DSI_left, const cv::Mat& DSI_right){
	int rows = DSI_left.size[0];
	int cols = DSI_right.size[1];
	int maxDisp = DSI_right.size[2];
	cv::Mat conf(cv::Size(cols, rows), cv::DataType<float>::type);
	for (int r = 0; r<rows; r++){
		for (int c = 0; c<cols; c++){
			int disp_left = -1;
			float best_cost = std::numeric_limits<float>::max();
			for (int delta = 0; delta<maxDisp; delta++){
				float cost = DSI_left.at<float>(r, c, delta);
				disp_left = (cost < best_cost) ? delta : disp_left;
				best_cost = (best_cost < cost) ? best_cost : cost;
				
			}
			int disp_right = -1;
			best_cost = std::numeric_limits<float>::max();
			for (int delta = 0; delta<maxDisp; delta++){
				float cost = DSI_left.at<float>(r, c-disp_left, delta);
				disp_right = (cost < best_cost) ? delta : disp_right;
				best_cost = (best_cost < cost) ? best_cost : cost;

			}
			conf.at<float>(r, c) = -std::abs(disp_left-disp_right);
		}
	}
	conf = conf(cv::Range(0, rows), cv::Range(maxDisp, cols));
	cv::normalize(conf, conf, 0, 255, cv::NORM_MINMAX, CV_8UC1);
	return conf;
}