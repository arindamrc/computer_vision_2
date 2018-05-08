#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <sstream>
#include <random>
#include "Tracking.hh"


void Tracker::computeHistogram(const cv::Mat& image, const cv::Point& p, Vector& histogram)
{
        f32 half_width = _objectWindow_width / 2.0;
        f32 half_height = _objectWindow_height / 2.0;
        int histSize = 256;
        float range[] = {0, 256} ;
        const float* histRange = {range};
        cv::Rect roi(
                p.x - half_width, 
                p.y - half_height, 
                _objectWindow_width, 
                _objectWindow_height
        );
        cv::Mat patch = image(roi);
        cv::cvtColor(patch, patch, CV_BGR2GRAY);
        histogram.resize(histSize);
        cv::Mat hist;
        cv::calcHist(&patch, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, true, false);
        // cv::normalize(hist, hist, 1, 0, cv::NORM_L1); // normalize?
        if (hist.isContinuous()) {
                histogram.assign((float*)hist.datastart, (float*)hist.dataend);
        } else {
                hist.copyTo(histogram);
        }
}

void Tracker::drawTrackedFrame(cv::Mat& image, cv::Point& position)
{
        cv::rectangle(image, cv::Point(position.x - _objectWindow_width / 2, position.y - _objectWindow_height / 2),
			cv::Point(position.x + _objectWindow_width / 2, position.y + _objectWindow_height / 2), 0, 3);
	cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
	cv::imshow("Display window", image);
	//std::sleep(1);
	cv::waitKey(0);
}

void Tracker::findBestMatch(const cv::Mat& image, cv::Point& lastPosition, AdaBoost& adaBoost)
{
        
}

u32 Tracker::getRandomDisplacement(u32 min, u32 max)
{
        return (max - min) * rand() + min;
}


void Tracker::generateTrainingData(std::vector<Example>& data, const std::vector<cv::Mat>& imageSequence, const std::vector<cv::Point>& referencePoints)
{
        int lbl_positive = 1;
        int lbl_negative = 0;
        for(int i = 0; i < imageSequence.size(); i++){
                // positive example
                cv::Point pt_pos = referencePoints[i];
                
                // negative examples
                cv::Point pt_neg1(
                        pt_pos.x - _displacement_x + getRandomDisplacement(), 
                        pt_pos.y - _displacement_y + getRandomDisplacement()
                );
                cv::Point pt_neg2(
                        pt_pos.x - _displacement_x + getRandomDisplacement(), 
                        pt_pos.y + _displacement_y + getRandomDisplacement()
                );
                cv::Point pt_neg3(
                        pt_pos.x + _displacement_x + getRandomDisplacement(), 
                        pt_pos.y - _displacement_y + getRandomDisplacement()
                );
                cv::Point pt_neg4(
                        pt_pos.x + _displacement_x + getRandomDisplacement(), 
                        pt_pos.y + _displacement_y + getRandomDisplacement()
                );
                
                Vector hist_pos, hist_neg1, hist_neg2, hist_neg3, hist_neg4;
                
                computeHistogram(imageSequence[i], pt_pos, hist_pos);
                computeHistogram(imageSequence[i], pt_neg1, hist_neg1);
                computeHistogram(imageSequence[i], pt_neg2, hist_neg2);
                computeHistogram(imageSequence[i], pt_neg3, hist_neg3);
                computeHistogram(imageSequence[i], pt_neg4, hist_neg4);
                
                data[i].attributes = hist_pos;
                data[i].label = lbl_positive;
                
                data[i+1].attributes = hist_neg1;
                data[i+1].label = lbl_negative;
                data[i+2].attributes = hist_neg2;
                data[i+2].label = lbl_negative;
                data[i+3].attributes = hist_neg3;
                data[i+3].label = lbl_negative;
                data[i+4].attributes = hist_neg4;
                data[i+4].label = lbl_negative;
        }
}

void Tracker::loadImage(const std::string& imageFile, cv::Mat& image)
{
}

void Tracker::loadTestFrames(const char* testDataFile, std::vector<cv::Mat>& imageSequence, cv::Point& startingPoint)
{
}

void Tracker::loadTrainFrames(const char* trainDataFile, std::vector<cv::Mat>& imageSequence, std::vector<cv::Point>& referencePoints)
{
        std::ifstream f(trainDataFile);
        std::string prefix = "./nemo/";
	std::string frame_name;
        u32 x,y;
        while(f >> frame_name >> x >> y){
               frame_name = prefix + frame_name;
               cv::Mat img = cv::imread(frame_name);
               cv::Point p(x,y);
               imageSequence.push_back(img);
               referencePoints.push_back(p);
        }
}


int Tracker::track(const char* trainFName, const char* testFName, u32 adaBoostIterations)
{
	// load the training frames
	std::vector<cv::Mat> imageSequence;
	std::vector<cv::Point> referencePoints;
	loadTrainFrames(trainFName, imageSequence, referencePoints);

	// generate gray-scale histograms from the training frames:
	// one positive example per frame (_objectWindow_width x _objectWindow_height window around reference point for object)
	// four negative examples per frame (with _displacement_{x/y} + small random displacement from reference point)
	std::vector<Example> trainingData;
	generateTrainingData(trainingData, imageSequence, referencePoints);

	// initialize AdaBoost and train a cascade with the extracted training data
	AdaBoost adaBoost(adaBoostIterations);
	adaBoost.initialize(trainingData);
	adaBoost.trainCascade(trainingData);

	// log error rate on training set
	u32 nClassificationErrors = 0;
	for (u32 i = 0; i < trainingData.size(); i++) {
		u32 label = adaBoost.classify(trainingData.at(i).attributes);
		nClassificationErrors += (label == trainingData.at(i).label ? 0 : 1);
	}
	std::cout << "Error rate on training set: " << (f32)nClassificationErrors / (f32)trainingData.size() << std::endl;

	// load the test frames and the starting position for tracking
	std::vector<Example> testImages;
	cv::Point lastPosition;
	loadTestFrames(testFName, imageSequence, lastPosition);

	// for each frame...
	for (u32 i = 0; i < imageSequence.size(); i++) {
		// ... find the best match in a window of size
		// _searchWindow_width x _searchWindow_height around the last tracked position
		findBestMatch(imageSequence.at(i), lastPosition, adaBoost);

		// draw the result
		drawTrackedFrame(imageSequence.at(i), lastPosition);
	}

	return 0;
}

