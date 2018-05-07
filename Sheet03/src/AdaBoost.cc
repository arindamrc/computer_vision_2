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
        f32 sum = 0.0f;
        for(int i = 0; i < this->weights_.size(); i++){
                sum += this->weights_[i];
        }
        for(int i = 0; i < this->weights_.size(); i++){
                this->weights_[i] /= sum;
        }
}

void AdaBoost::updateWeights(const std::vector<Example>& data, const std::vector<u32>& classAssignments, u32 iteration) {
        for(int i = 0; i < data.size(); i++){
                this->weights_[i] = this->weights_[i] * pow(this->classifierWeights_[iteration], 1 - abs(classAssignments[i] - data[i].label));
        }
}

f32 AdaBoost::weightedErrorRate(const std::vector<Example>& data, const std::vector<u32>& classAssignments) {
        f32 epsilon = 0.0;
        for(int i = 0; i < data.size(); i++){
                if(classAssignments[i] != data[i].label){
                        epsilon += this->weights_[i];
                }
        }
        return (epsilon/(1 - epsilon));
}

void AdaBoost::initWeakClassifier(Stump& stump)
{
        u32 attribute = rand() % static_cast<u32>(this->attrCount_ - 1);
        f32 max = this->attrRange_[attribute].max;
        f32 min = this->attrRange_[attribute].min;
        f32 threshold = (max - min) * ((((float) rand()) / (float) RAND_MAX)) + min;
        u32 classLabelLt = (rand() > RAND_MAX/2) ? 0 : 1;
        stump.initialize(this->attrCount_, attribute, threshold, classLabelLt);
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
                std::vector<u32> classAssignments = weakClassifier_[t].train(data,weights_);
                this->classifierWeights_[t] = this->weightedErrorRate(data, classAssignments);
                this->updateWeights(data, classAssignments, t);
                this->normalizeWeights();
        }
}

u32 AdaBoost::classify(const Vector& v) {
        f32 confidence_0 = this->confidence(v, 0);
        f32 confidence_1 = this->confidence(v, 1);
        if(confidence_0 > confidence_1){
                return 0;
        }else{
                return 1;
        }
}

f32 AdaBoost::confidence(const Vector& v, u32 k) {
        f32 sum = 0.0f;
        for(int t = 0; t  < nIterations_; t++){
                if(weakClassifier_[t].classify(v) == k){
                        sum += log(1.0/this->classifierWeights_[t]);
                }
        }
        return sum;
}
