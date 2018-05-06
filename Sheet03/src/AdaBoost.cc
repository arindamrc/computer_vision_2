/*
 * AdaBoost.cc
 *
 *  Created on: Apr 25, 2014
 *      Author: richard
 */

#include "AdaBoost.hh"
#include <iostream>
#include <cmath>

AdaBoost::AdaBoost(u32 nIterations) :
	nIterations_(nIterations)
{}

void AdaBoost::normalizeWeights() {

}

void AdaBoost::updateWeights(const std::vector<Example>& data, const std::vector<u32>& classAssignments, u32 iteration) {

}

f32 AdaBoost::weightedErrorRate(const std::vector<Example>& data, const std::vector<u32>& classAssignments) {

}

void AdaBoost::initialize(std::vector<Example>& data) {
	// initialize weak classifiers
	for (u32 iteration = 0; iteration < nIterations_; iteration++) {
		weakClassifier_.push_back(Stump());
	}
	// initialize classifier weights
	classifierWeights_.resize(nIterations_);
	// initialize weights
	weights_.resize(data.size());
	for (u32 i = 0; i < data.size(); i++) {
		//weights_.at(i) = ?;
	}
}

void AdaBoost::trainCascade(std::vector<Example>& data) {

}

u32 AdaBoost::classify(const Vector& v) {

}

f32 AdaBoost::confidence(const Vector& v, u32 k) {

}
