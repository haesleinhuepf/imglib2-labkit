
package clij;

import hr.irb.fastRandomForest.FastRandomForest;
import net.imglib2.util.Cast;
import weka.core.Instance;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class MyRandomForest implements SimpleClassifier {

	private final List<MyRandomTree> trees;

	public MyRandomForest(FastRandomForest original) {
		Object bagger = ReflectionUtils.getPrivateField(original, "m_bagger");
		Object[] trees = Cast.unchecked(ReflectionUtils.getPrivateField(bagger, "m_Classifiers"));
		this.trees = Collections.unmodifiableList(Stream.of(trees).map(MyRandomTree::new).collect(
			Collectors.toList()));
	}

	public List<MyRandomTree> trees() {
		return trees;
	}

	public int classifyInstance(Instance instance) {
		return findMax(distributionForInstance(instance));
	}

	private int findMax(double[] doubles) {
		int maxIndex = 0;
		double max = doubles[maxIndex];
		for (int i = 1; i < doubles.length; i++) {
			if (max < doubles[i]) {
				maxIndex = i;
				max = doubles[maxIndex];
			}
		}
		return maxIndex;
	}

	public double[] distributionForInstance(Instance instance) {
		double[] result = new double[2];
		setZero(result);
		for (MyRandomTree tree : trees)
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
