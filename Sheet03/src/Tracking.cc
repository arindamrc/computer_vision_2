#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <sstream>
#include <random>
#include <limits>
#include "Tracking.hh"


void Tracker::computeHistogram(const cv::Mat& image, const cv::Point& p, Vector& histogram)
{
        f32 half_width = _objectWindow_width / 2.0;
        f32 half_height = _objectWindow_height / 2.0;
        int histSize = 256;
        float range[] = {0, 256} ;
        const float* histRange = {range};
        u32 tl_x = (p.x - half_width) > 0 ? (p.x - half_width) : 0;
        u32 tl_y = (p.y - half_height) > 0 ? (p.y - half_height) : 0;
        u32 br_x = (p.x + half_width) < image.cols ? (p.x + half_width) : (image.cols - 1);
        u32 br_y = (p.y + half_height) < image.rows ? (p.y + half_height) : (image.rows - 1);
        
        cv::Point pt_tl(tl_x, tl_y);
        cv::Point pt_br(br_x, br_y);
        
        cv::Rect roi(pt_tl, pt_br);
        
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
        u32 tl_x = (lastPosition.x - (0.5 * _searchWindow_width)) > 0 ? (lastPosition.x - (0.5 * _searchWindow_width)) : 0;
        u32 tl_y = (lastPosition.y - (0.5 * _searchWindow_height)) > 0 ? (lastPosition.y - (0.5 * _searchWindow_height)) : 0;
        u32 lbl_pos = 1;
        f32 max_confidence = -std::numeric_limits<f32>::min();
        cv::Point mostLikelyPt;
        for(int c = tl_x; c < tl_x + _searchWindow_width; c++){
                for(int r = tl_y; r < tl_y + _searchWindow_height; r++){
                        Vector histogram;
                        u32 x = (c < image.cols) ? c : image.cols;
                        u32 y = (r < image.rows) ? r : image.rows;
                        cv::Point windowPt(x,y);
                        computeHistogram(image, windowPt, histogram);
                        f32 confidence = adaBoost.confidence(histogram, lbl_pos);
//                         std::cout << confidence << "\t";
                        if(confidence > max_confidence){
                                max_confidence = confidence;
                                mostLikelyPt = windowPt;
                        }
                }
                std::cout << max_confidence << std::endl;
        }
        lastPosition = mostLikelyPt;
}

u32 Tracker::getRandomDisplacement(u32 min, u32 max)
{
        f32 r = min + rand() % (max - min + 1);
        return r;
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
                
                Example ex_pos;
                ex_pos.attributes = hist_pos;
                ex_pos.label = lbl_positive;
                data.push_back(ex_pos);
                
                Example ex_neg1;
                ex_neg1.attributes = hist_neg1;
                ex_neg1.label = lbl_negative;
                data.push_back(ex_neg1);
                Example ex_neg2;
                ex_neg2.attributes = hist_neg2;
                ex_neg2.label = lbl_negative;
                data.push_back(ex_neg2);
                Example ex_neg3;
                ex_neg3.attributes = hist_neg3;
                ex_neg3.label = lbl_negative;
                data.push_back(ex_neg3);
                Example ex_neg4;
                ex_neg4.attributes = hist_neg4;
                ex_neg4.label = lbl_negative;
                data.push_back(ex_neg4);
        }
}

void Tracker::loadImage(const std::string& imageFile, cv::Mat& image)
{
}

void Tracker::loadTestFrames(const char* testDataFile, std::vector<cv::Mat>& imageSequence, cv::Point& startingPoint)
{
        std::ifstream f(testDataFile);
        std::string prefix = "./nemo/";
	std::string frame_name;
        u32 x,y;
        f >> x >> y; // starting co-ords
        startingPoint.x = x;
        startingPoint.y = y;
        while(f >> frame_name){
               frame_name = prefix + frame_name;
               cv::Mat img = cv::imread(frame_name);
               imageSequence.push_back(img);
        }
}

void Tracker::loadTrainFrames(const char* trainDataFile, std::vector<cv::Mat>& imageSequence, std::vector<cv::Point>& referencePoints)
{
        std::ifstream f(trainDataFile);
        std::string prefix = "./nemo/";
	std::string frame_name;
        u32 x,y;
        while(f >> frame_name >> x >> y){
//                 std::cout << frame_name << "," << x << "," << y << std::endl; 
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
        std::cout << "Training frames loaded..." << std::endl;

	// generate gray-scale histograms from the training frames:
	// one positive example per frame (_objectWindow_width x _objectWindow_height window around reference point for object)
	// four negative examples per frame (with _displacement_{x/y} + small random displacement from reference point)
	std::vector<Example> trainingData;
	generateTrainingData(trainingData, imageSequence, referencePoints);
        std::cout << "Training data generated..." << std::endl;

	// initialize AdaBoost and train a cascade with the extracted training data
	AdaBoost adaBoost(adaBoostIterations);
	adaBoost.initialize(trainingData);
	adaBoost.trainCascade(trainingData);
        std::cout << "Training completed..." << std::endl;

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
                std::cout << "lastPosition: " << lastPosition << std::endl;
		findBestMatch(imageSequence.at(i), lastPosition, adaBoost);
		// draw the result
		drawTrackedFrame(imageSequence.at(i), lastPosition);
	}

	return 0;
}

