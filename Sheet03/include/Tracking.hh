#ifndef TRACKING_HH_
#define TRACKING_HH_

#include <opencv2/core/core.hpp>
#include <vector>
#include "Types.hh"
#include "AdaBoost.hh"

#define _objectWindow_width 121
#define _objectWindow_height 61

#define _searchWindow_width 61
#define _searchWindow_height 61

// use 30/15 for overlapping negative examples and 120/60 for non-overlapping negative examples
#define _displacement_x 30 //120
#define _displacement_y 15 //60

#define _sample_count 5

class Tracker{
private:
        void computeHistogram(const cv::Mat& image, const cv::Point& p, Vector& histogram);
        void generateTrainingData(std::vector<Example>& data, const std::vector<cv::Mat>& imageSequence, const std::vector<cv::Point>& referencePoints);
        void loadImage(const std::string& imageFile, cv::Mat& image);
        void loadTrainFrames(const char* trainDataFile, std::vector<cv::Mat>& imageSequence,
                        std::vector<cv::Point>& referencePoints);
        void loadTestFrames(const char* testDataFile, std::vector<cv::Mat>& imageSequence, cv::Point& startingPoint);
        void findBestMatch(const cv::Mat& image, cv::Point& lastPosition, AdaBoost& adaBoost);
        void drawTrackedFrame(cv::Mat& image, cv::Point& position);
        u32 getRandomDisplacement(u32 min = -10, u32 max = 10);
public:
        int track(const char* trainFName, const char* testFName, u32 adaBoostIterations);        
};


#endif /* TRACKING_HH_ */
