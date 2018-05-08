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

void Stump::initialize(u32 dimension, u32 splitAttribute, f32 threshold, u32 classLabelLt) {
	dimension_ = dimension;
        splitAttribute_ = splitAttribute;
        splitValue_ = threshold;
        classLabelLeft_ = classLabelLt;
        classLabelRight_ = 1 - classLabelLt;
}

/*
 * train this classifier on all training samples and return the results as a vector.
 */
std::vector<u32> Stump::train(const std::vector<Example>& data, const Vector& weights) {
        std::vector<u32> classAssignments;
        for(int i = 0; i < data.size(); i++){
                u32 classLabel = this->classify(data[i].attributes);
                classAssignments.push_back(classLabel);
        }
        return classAssignments;
}

/*
 * Classify a sample. Result is either 0 or 1.
 */
u32 Stump::classify(const Vector& v) {
        f32 attrValue = v[this->splitAttribute_];
        if(attrValue < this->splitValue_){
                return this->classLabelLeft_;
        }else{
                return this->classLabelRight_;
        }
}
