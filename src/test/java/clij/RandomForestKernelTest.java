
package clij;

import ij.IJ;
import ij.ImagePlus;
import net.haesleinhuepf.clij.CLIJ;
import net.haesleinhuepf.clij.clearcl.ClearCLBuffer;
import net.haesleinhuepf.clij.coremem.enums.NativeTypeEnum;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.type.numeric.RealType;
import net.imglib2.view.Views;

import java.util.HashMap;
import java.util.Map;

public class RandomForestKernelTest {

	public static void main(String... args) {
		CLIJ clij = CLIJ.getInstance();
		try {
			int width = 3;
			int height = 1;
			int numberOfTrees = 2;
			int numberOfFeatures = 1;
			int numberOfClasses = 2;
			int numberOfNodes = 2;
			int numberOfLeafs = 3;
			ClearCLBuffer distributions = clij.create(new long[] { width, height, numberOfClasses },
				NativeTypeEnum.Float);
			ImagePlus src = ImageJFunctions.wrapFloat(ArrayImgs.floats(
				new float[] { 41, 43, 45 },
				width, height, numberOfFeatures), "src");
			ImagePlus thresholds = ImageJFunctions.wrapFloat(ArrayImgs.floats(
				new float[] {
					42, 44,

					43.5f, 0,
				},
				1, numberOfNodes, numberOfTrees), "thresholds");
			ImagePlus probabilities = ImageJFunctions.wrapFloat(ArrayImgs.floats(
				new float[] {
					2, 2,
					3, 0,
					4, 4,

					0, 1,
					0, 0,
					0, 0
				},
				numberOfClasses, numberOfLeafs, numberOfTrees), "probabilities");
			ImagePlus indices = ImageJFunctions.wrap(ArrayImgs.floats(
				new float[] {
					0, -1, 1,
					0, -2, -3,

					0, -1, -2,
					0, 0, 0
				}, 3, numberOfNodes, numberOfTrees), "indices");

			randomForest(clij,
				distributions,
				clij.push(src),
				clij.push(thresholds),
				clij.push(probabilities),
				clij.push(indices),
				numberOfTrees,
				numberOfClasses);

			RandomAccessibleInterval<? extends RealType<?>> result = clij.pullRAI(distributions);
			Views.iterable(result).forEach(System.out::println);
		}
		catch (Throwable t) {
			t.printStackTrace();
		}
		finally {
			clij.close();
		}
	}

	private static void randomForest(CLIJ clij,
		ClearCLBuffer distributions,
		ClearCLBuffer src,
		ClearCLBuffer thresholds,
		ClearCLBuffer probabilities,
		ClearCLBuffer indices,
		int numberOfTrees,
		int numberOfClasses)
	{
		long[] globalSizes = { src.getWidth(), src.getHeight() };
		Map<String, Object> parameters = new HashMap<>();
		parameters.put("src", src);
		parameters.put("dst", distributions);
		parameters.put("thresholds", thresholds);
		parameters.put("probabilities", probabilities);
		parameters.put("indices", indices);
		parameters.put("num_trees", numberOfTrees);
		parameters.put("num_classes", numberOfClasses);
		clij.execute(ClijDemo.class, "random_forest.cl", "random_forest", globalSizes,
			parameters);
	}
}
