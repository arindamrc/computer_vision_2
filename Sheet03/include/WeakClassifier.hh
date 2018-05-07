/*
 * NearestMeanClassifier.hh
 *
 *  Created on: Apr 25, 2014
 *      Author: richard
 */

#ifndef NEARESTMEANCLASSIFIER_HH_
#define NEARESTMEANCLASSIFIER_HH_

#include <vector>
#include "Types.hh"

class Stump
{
private:
	u32 dimension_;
	u32 splitAttribute_;
	f32 splitValue_;
	u32 classLabelLeft_;
	u32 classLabelRight_;

public:
	Stump();
	void initialize(u32 dimension, u32 splitAttribute, f32 threshold, u32 classLabelLt);
	std::vector<u32> train(const std::vector<Example>& data, const Vector& weights);
	u32 classify(const Vector& v);
};

#endif /* NEARESTMEANCLASSIFIER_HH_ */
