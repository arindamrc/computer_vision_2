/*
 * AdaBoost.hh
 *
 *  Created on: Apr 25, 2014
 *      Author: richard
 */

#ifndef ADABOOST_HH_
#define ADABOOST_HH_
#include <vector>
#include <string.h>
#include <limits>
#include "Types.hh"
#include "WeakClassifier.hh"

class Range{

public:
        f32 min = std::numeric_limits<float>::max();
        f32 max = std::numeric_limits<float>::min();
};

class AdaBoost
{
private:
	u32 nIterations_, attrCount_, dataCount_;
	Vector weights_;
	std::vector<Stump> weakClassifier_;
	Vector classifierWeights_;
        std::vector<Range> attrRange_;

	void normalizeWeights();
	void updateWeights(const std::vector<Example>& data, const std::vector<u32>& classAssignments, u32 iteration);
	f32 weightedErrorRate(const std::vector<Example>& data, const std::vector<u32>& classAssignments);
        void initWeakClassifier(Stump& stump);
        
public:
	AdaBoost(u32 nIterations);
	void initialize(std::vector<Example>& data);
	void trainCascade(std::vector<Example>& data);
	u32 classify(const Vector& v);
	f32 confidence(const Vector& v, u32 k);
};


#endif /* ADABOOST_HH_ */
