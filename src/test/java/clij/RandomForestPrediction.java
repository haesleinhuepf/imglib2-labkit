
package clij;

import weka.core.Instance;

import java.util.List;
import java.util.stream.Collectors;

public class RandomForestPrediction implements SimpleClassifier {

	private final MyRandomForest forest;

	private final List<RandomTreePrediction> trees;

	private final MajorityClassifier majority;

	public RandomForestPrediction(MyRandomForest forest) {
		this.forest = forest;
		this.trees = forest.trees().stream().map(RandomTreePrediction::new).collect(Collectors
			.toList());
		this.majority = new MajorityClassifier(trees);
	}

	@Override
	public double[] distributionForInstance(Instance instance) {
		return majority.distributionForInstance(instance);
	}
}
