
package clij;

public class ArrayUtils {

	public static int findMax(double[] doubles) {
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
}
