
package clij;

import weka.core.Instance;

import java.util.List;
import java.util.stream.Collectors;

public class RandomForestPrediction implements SimpleClassifier {

	private final MyRandomForest forest;

	private final List<RandomTreePrediction> trees;

	private final MajorityClassifier majority;

	private final int numberOfClasses;

	private final int numberOfTrees;

	private final int[][][] nodeIndicies;

	private final double[][] nodeThresholds;

	private final double[][][] classProbabilities;

	public RandomForestPrediction(MyRandomForest forest, int numberOfClasses) {
		this.forest = forest;
		this.trees = forest.trees().stream().map(RandomTreePrediction::new).collect(Collectors
			.toList());
		this.majority = new MajorityClassifier(trees);
		this.numberOfClasses = numberOfClasses;
		this.numberOfTrees = trees.size();
		int maxNumberOfNodes = trees.stream().mapToInt(x -> x.numberOfNodes).max().getAsInt();
		int maxNumberOfLeafs = trees.stream().mapToInt(x -> x.numberOfLeafs).max().getAsInt();
		this.nodeIndicies = new int[numberOfTrees][maxNumberOfNodes][3];
		this.nodeThresholds = new double[numberOfTrees][maxNumberOfNodes];
		this.classProbabilities = new double[numberOfTrees][maxNumberOfLeafs][numberOfClasses];
		for (int j = 0; j < numberOfTrees; j++) {
			RandomTreePrediction tree = trees.get(j);
			for (int i = 0; i < tree.numberOfNodes; i++) {
				nodeIndicies[j][i][0] = tree.attributeIndicies[i];
				nodeIndicies[j][i][1] = tree.smallerChild[i];
				nodeIndicies[j][i][2] = tree.biggerChild[i];
				nodeThresholds[j][i] = tree.threshold[i];
			}
			for (int i = 0; i < tree.leafCount; i++)
				for (int k = 0; k < numberOfClasses; k++)
					classProbabilities[j][i][k] = tree.classProbabilities[i][k];
		}
	}

	@Override
	public double[] distributionForInstance(Instance instance) {
		double[] distribution = new double[numberOfClasses];
		for (int j = 0; j < numberOfTrees; j++) {
			addDistributionForTree(instance, j, distribution);
		}
		return ArrayUtils.normalize(distribution);
	}

	private void addDistributionForTree(Instance instance, int tree, double[] distribution) {
		int nodeIndex = 0;
		while (nodeIndex >= 0) {
			int attributeIndex = nodeIndicies[tree][nodeIndex][0];
			double attributeValue = instance.value(attributeIndex);
			int b = attributeValue < nodeThresholds[tree][nodeIndex] ? 1 : 2;
			nodeIndex = nodeIndicies[tree][nodeIndex][b];
		}
		int leafIndex = -1 - nodeIndex;
		ArrayUtils.add(classProbabilities[tree][leafIndex], distribution);
	}
}
