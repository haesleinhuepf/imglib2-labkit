
package clij;

import weka.core.Instance;

public interface SimpleClassifier {

	double[] distributionForInstance(final Instance instance);
}
