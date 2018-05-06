/*
 * NearestMeanClassifier.cc
 *
 *  Created on: Apr 25, 2014
 *      Author: richard
 */

#include "WeakClassifier.hh"
#include <cmath>
#include <iostream>

/*
 * Stump
 */

Stump::Stump() :
		dimension_(0),
		splitAttribute_(0),
		splitValue_(0),
		classLabelLeft_(0),
		classLabelRight_(0)
{}

void Stump::initialize(u32 dimension) {
	dimension_ = dimension;
}

f32 Stump::weightedGain(const std::vector<Example>& data, const Vector& weights, u32 splitAttribute, f32 splitValue, u32& resultingLeftLabel) {

}

void Stump::train(const std::vector<Example>& data, const Vector& weights) {

}

u32 Stump::classify(const Vector& v) {

}

void Stump::classify(const std::vector<Example>& data, std::vector<u32>& classAssignments) {

}
