
package clij;

import ij.ImagePlus;
import net.haesleinhuepf.clij.CLIJ;
import net.haesleinhuepf.clij.clearcl.ClearCLBuffer;
import net.haesleinhuepf.clij.coremem.enums.NativeTypeEnum;
import net.haesleinhuepf.clij.kernels.Kernels;
import net.imglib2.FinalInterval;
import net.imglib2.Interval;
import net.imglib2.util.Intervals;
import net.imglib2.view.Views;

import java.util.HashMap;
import java.util.Map;

public class ClijDemo {

	public static void main(String... args) {

		try {
			CLIJ clij = CLIJ.getInstance();
			ImagePlus input = new ImagePlus("/home/arzt/Documents/Datasets/Example/small-3d-stack.tif");
			input.show();
			ClearCLBuffer inputCl = clij.push(input);
			int numChannels = 10;
			ClearCLBuffer outputCl = clij.createCLBuffer(new long[] { inputCl.getWidth(), inputCl
				.getHeight(), inputCl.getDepth() * numChannels }, NativeTypeEnum.Float);
			for (int i = 0; i < numChannels; i++) {
				Interval sourceInterval = interval(inputCl);
				FinalInterval destinationInterval = Intervals.translate(sourceInterval, i * inputCl
					.getDepth(), 2);
				copy(clij, inputCl, sourceInterval, outputCl, destinationInterval);
			}
			ImagePlus output = clij.pull(outputCl);
			output.setDisplayRange(0, 255);
			output.show();
		}
		catch (Throwable t) {
			t.printStackTrace();
		}
	}

	private static void copy(CLIJ clij, ClearCLBuffer inputCl, Interval sourceInterval,
		ClearCLBuffer outputCl, Interval destinationInterval)
	{
		long[] globalSizes = Intervals.dimensionsAsLongArray(sourceInterval);
		Map<String, Object> parameters = new HashMap<>();
		parameters.put("src", inputCl);
		parameters.put("dst", outputCl);
		parameters.put("src_offset_x", (int) sourceInterval.min(0));
		parameters.put("src_offset_y", (int) sourceInterval.min(1));
		parameters.put("src_offset_z", (int) sourceInterval.min(2));
		parameters.put("dst_offset_x", (int) destinationInterval.min(0));
		parameters.put("dst_offset_y", (int) destinationInterval.min(1));
		parameters.put("dst_offset_z", (int) destinationInterval.min(2));
		clij.execute(ClijDemo.class, "copy_with_offset.cl", "copy_with_offset", globalSizes,
			parameters);
	}

	private static Interval interval(ClearCLBuffer inputCl) {
		return new FinalInterval(inputCl.getDimensions());
	}

	private static void crop(CLIJ clij, ClearCLBuffer inputCl, ClearCLBuffer outputCl) {
		HashMap<String, Object> parameters = new HashMap<>();
		parameters.put("src", inputCl);
		parameters.put("dst", outputCl);
		parameters.put("start_x", 0);
		parameters.put("start_y", 0);
		parameters.put("start_z", 0);
		clij.execute(Kernels.class, "duplication.cl", "crop_3d", parameters);
	}
}
