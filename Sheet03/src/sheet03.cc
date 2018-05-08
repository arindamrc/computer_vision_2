#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>
#include <string.h>
#include <fstream>
#include <cmath>
#include <vector>
#include <stdlib.h>
#include "Types.hh"
#include "AdaBoost.hh"
#include "Tracking.hh"

using namespace cv;
using namespace std;

void readData(const char* filename, std::vector<Example>& data) {
	std::ifstream f(filename);
	u32 dimension, nObservations;
	f >> dimension;
	f >> nObservations;
	data.resize(nObservations);
	for (u32 i = 0; i < nObservations; i++) {
		f >> data.at(i).label;
		data.at(i).attributes.resize(dimension);
		for (u32 d = 0; d < dimension; d++) {
			f >> data.at(i).attributes.at(d);
		}
	}
	f.close();
}

void violaAndJones(Mat& img_gray, Mat& img_color){
        CascadeClassifier cascade;
        const float scale_factor(1.2f);
        const int min_neighbors(3);

        if (cascade.load("./face-model.xml")) {
                equalizeHist(img_gray, img_gray);
                vector<Rect> objs;
                cascade.detectMultiScale(img_gray, objs, scale_factor, min_neighbors);

                for (int n = 0; n < objs.size(); n++) {
                        rectangle(img_color, objs[n], Scalar(255,0,0), 8);
                }
                imshow("Viola Jones", img_color);
                waitKey(0);
        }
        else{
                std::cerr << "Unable to load face models" << std::endl;
        }
}

void nr1(){
        string images[3] = {
                "./img1.jpg", 
                "./img2.jpg", 
                "./img3.jpg"
        };
        for(int i = 0; i < images->size(); i++){
               Mat img_gray = imread(images[i], CV_LOAD_IMAGE_GRAYSCALE); 
               Mat img_color = imread(images[i], CV_LOAD_IMAGE_COLOR); 
               violaAndJones(img_gray, img_color);
        }
}

void nr2(const char* trainFile, const char* testFile) {
        u32 adaBoostIterations[] = {1,2,3,4,5,10,15,20,30,40,50,80,100,150,250,500};
	std::vector<Example> trainingData;
	std::vector<Example> testData;
	readData(trainFile, trainingData);
	readData(testFile, testData);

        for(int k : adaBoostIterations){
                // train cascade of weak classifiers
                AdaBoost adaBoost(k);
                adaBoost.initialize(trainingData);
                adaBoost.trainCascade(trainingData);

                // classification on test data
                u32 nClassificationErrors = 0;
                for (u32 i = 0; i < testData.size(); i++) {
                        u32 c = adaBoost.classify(testData.at(i).attributes);
                        nClassificationErrors += (c == testData.at(i).label ? 0 : 1);
                }
                f32 accuracy = 1.0 - (f32) nClassificationErrors / (f32) testData.size();

                std::cout << "k= " << k << std::endl;
                std::cout << "Classified " << testData.size() << " examples." << std::endl;
                std::cout << "Accuracy: " << accuracy << " (" << testData.size() - nClassificationErrors << "/" << testData.size() << ")" << std::endl;
//                 std::cout << k << "," << accuracy << std::endl;
        }
}

void nr3(){
      Tracker tracker;
      tracker.track("./nemo/frames.train","./nemo/frames.test", 200);
}


int main(int argc, char* argv[])
{
//         nr1();
//         nr2("./splice/splice.train", "./splice/splice.test");
        nr3();
        return 0;
}
