#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <string.h>
#include <fstream>
#include <cmath>
#include <stdlib.h>
#include "Types.hh"
#include "AdaBoost.hh"

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

int nr2(const char* trainFile, const char* testFile, u32 adaBoostIterations) {
	std::vector<Example> trainingData;
	std::vector<Example> testData;
	readData(trainFile, trainingData);
	readData(testFile, testData);

	// train cascade of weak classifiers
	AdaBoost adaBoost(adaBoostIterations);
	adaBoost.initialize(trainingData);
	adaBoost.trainCascade(trainingData);

	// classification on test data
	u32 nClassificationErrors = 0;
	for (u32 i = 0; i < testData.size(); i++) {
		u32 c = adaBoost.classify(testData.at(i).attributes);
		nClassificationErrors += (c == testData.at(i).label ? 0 : 1);
	}
	f32 accuracy = 1.0 - (f32) nClassificationErrors / (f32) testData.size();

	std::cout << "Classified " << testData.size() << " examples." << std::endl;
	std::cout << "Accuracy: " << accuracy << " (" << testData.size() - nClassificationErrors << "/" << testData.size() << ")" << std::endl;
	return 0;
}


int main(int argc, char* argv[])
{
    // TODO implement your solution here

    // feel free to create additional .cpp files for other classes if needed

    return 0;
}
