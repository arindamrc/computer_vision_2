/*
 * AdaBoost.cc
 *
 *  Created on: Apr 25, 2014
 *      Author: richard
 */

#include "AdaBoost.hh"
#include <iostream>
#include <cmath>
#include <random>

AdaBoost::AdaBoost(u32 nIterations) :
	nIterations_(nIterations)
{}

void AdaBoost::normalizeWeights() {

}

void AdaBoost::updateWeights(const std::vector<Example>& data, const std::vector<u32>& classAssignments, u32 iteration) {

}

f32 AdaBoost::weightedErrorRate(const std::vector<Example>& data, const std::vector<u32>& classAssignments) {

}

void AdaBoost::initWeakClassifier(Stump& stump)
{
        u32 attribute = rand() % static_cast<u32>(this->attrCount_ - 1);
        f32 max = this->attrRange_[attribute].max;
        f32 min = this->attrRange_[attribute].min;
        f32 threshold = (max - min) * ((((float) rand()) / (float) RAND_MAX)) + min;
        stump.initialize(attribute, threshold);
}


void AdaBoost::initialize(std::vector<Example>& data) {
        this->attrCount_ = data[0].attributes.size();
        this->dataCount_ = data.size();
	// initialize weak classifiers
	for (u32 iteration = 0; iteration < nIterations_; iteration++) {
		weakClassifier_.push_back(Stump());
	}
	for (u32 attr = 0; attr < this->attrCount_; attr++){
                this->attrRange_.push_back(Range());
        }
	// initialize classifier weights
	classifierWeights_.resize(nIterations_);
	// initialize weights
	weights_.resize(data.size());
        // find the range of values for each attribute
	for (u32 i = 0; i < data.size(); i++) {
		weights_.at(i) = 1.0 / data.size();
                for (u32 j = 0; j < this->attrCount_; j++){
                        if(data[i].attributes[j] > this->attrRange_[j].max){
                                this->attrRange_[j].max = data[i].attributes[j];
                        }
                        if(data[i].attributes[j] < this->attrRange_[j].min){
                                this->attrRange_[j].min = data[i].attributes[j];
                        }
                }
	}
}

void AdaBoost::trainCascade(std::vector<Example>& data) {
        for(int t = 0; t < nIterations_; t++){
                this->initWeakClassifier(weakClassifier_[t]);
        }
}

u32 AdaBoost::classify(const Vector& v) {

}

f32 AdaBoost::confidence(const Vector& v, u32 k) {

}
