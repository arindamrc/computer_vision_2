#include <opencv2/highgui/highgui.hpp>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <sstream>
#include "Tracking.hh"


void Tracker::computeHistogram(const cv::Mat& image, const cv::Point& p, Vector& histogram)
{
}

void Tracker::drawTrackedFrame(cv::Mat& image, cv::Point& position)
{
}

void Tracker::findBestMatch(const cv::Mat& image, cv::Point& lastPosition, AdaBoost& adaBoost)
{
}

void Tracker::generateTrainingData(std::vector<Example>& data, const std::vector<cv::Mat>& imageSequence, const std::vector<cv::Point>& referencePoints)
{
}

void Tracker::loadImage(const std::string& imageFile, cv::Mat& image)
{
}

void Tracker::loadTestFrames(const char* testDataFile, std::vector<cv::Mat>& imageSequence, cv::Point& startingPoint)
{
}

void Tracker::loadTrainFrames(const char* trainDataFile, std::vector<cv::Mat>& imageSequence, std::vector<cv::Point>& referencePoints)
{
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

