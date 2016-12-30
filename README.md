# StereoMatching
Implementation of some stereo matching algorithms and confidence measures using OpenCV:
## stereo::costs 
namespace qith three different cost function, absolute difference, truncated absolute difference and squarred difference

## stereo::disp
namespace with function to compute a DSI given left and right image or to obtain a disparity map given a DSI:
* singlePixeDisparity: compute cost function on couples of pixel taken from left and right image.
* fixedWindowDisparity: compute cost function agregating cost in a square support surrounding the current pixel
* fixedWindowBG: as the normal fixed window, but speeded-up using box filtering
* getDisparityRange: returns a vector to be used with the funciton above given the max disparity allowed and if we are matching left to right or right to left
* getDisparity: given a DSI produces a disparity map normalized between 0 and 255

## stereo::sgm
namespace with function used to apply the sgm[1] optimization alghoritm
* scanlineOptimization: recompute cost across a single scanline identified by previousPosition
* sgmOptimization: apply sgm alghorithm using 8 different scanline 

## stereo::confidence
namespace with some confidence measure for stereo matching, given a DSI the produce a grey scale image with area with low confidence in black
* matchingScoreMeasure
* curvature
* peakRatio
* winnerMargin
* leftRightCheck

[1]  Hirschmuller, H. (2005). Accurate and Efficient Stereo Processing by Semi Global Matching and Mutual Information. CVPR .
