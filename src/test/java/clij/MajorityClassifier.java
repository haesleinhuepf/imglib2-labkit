
package clij;

import weka.core.Instance;

import java.util.Arrays;
import java.util.List;

public class MajorityClassifier implements SimpleClassifier {

	protected final List<? extends SimpleClassifier> trees;

	public MajorityClassifier(List<? extends SimpleClassifier> trees) {
		this.trees = trees;
	}

	public double[] distributionForInstance(Instance instance) {
		double[] result = new double[2];
		setZero(result);
		for (SimpleClassifier tree : trees)
			sum(tree.distributionForInstance(instance), result);
		return normalize(result);
	}

	private static void setZero(double[] result) {
		Arrays.fill(result, 0);
	}

	private static double[] sum(double[] a, double[] b) {
		for (int i = 0; i < a.length; i++)
			b[i] += a[i];
		return b;
	}

	private static double[] normalize(double[] values) {
		double sum = sum(values);
		for (int i = 0; i < values.length; i++) {
			values[i] /= sum;
		}
		return values;
	}

	private static double sum(double[] values) {
		double sum = 0;
		for (double value : values)
			sum += value;
		return sum;
	}

}
